"""Difference of submodular minimization (DSM) attack"""
import copy
import math
import os
import time
import logging
import gc
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Any, Literal
import numpy as np
import torch
from scipy.optimize import linprog
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from ..dataset import PromptDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult
from ..lm_utils import prepare_conversation, TokenMergeError, generate_ragged_batched, get_flops, get_disallowed_ids, filter_suffix, with_max_batchsize
from ..types import Conversation
from .submodular_utils import EneSubmodularSetFnReduction, EneReductionMap, DR_submodular_decomposition, SetFnReduction, pgm_lovasz, dca_dsm


@dataclass
class DCAConfig:
    """Config for the DCA optimizer."""
    hessian_upperbd: float | Literal["hessian_upperbd_at_zero"] = "hessian_upperbd_at_zero" # runs PGM in that case
    dsm_cache_dir: str | None = None
    outer_tol: float = 1e-5
    inner_gap_tol: float = 1e-4
    # num_outer_steps: will be set to num_steps / num_inner_steps 
    num_inner_steps: int = 1
    inner_solver: str = "pgm"
    tie_break: Literal["random"] | None = None  
    # TODO: might want to also try using random tie breaking in PGM when used as inner solver, can potentially speed it up?

@dataclass
class PGMConfig:
    """Config for the PGM optimizer."""
    L: float | Literal["singletons", "normalize", "polyak"] = "polyak"  
    tie_break: Literal["random"] | None = None 


@dataclass
class DSMConfig:
    """Config for the DSM attack."""
    # not strictly necessary, but good for type checking config, clarity,
    # and providing defaults if not specified in attacks.yaml
    name: str = "dsm"
    type: str = "discrete"
    version: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    placement: str = "suffix"
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    num_steps: int = 1
    lm_reg_weight: float = 0.0  # weight on -log p(x|q) when using reg_ce
    pgm_config: PGMConfig = field(default_factory=PGMConfig)
    optimizer: Literal["pgm", "dca"] = "pgm"  # "pgm" or "dca"
    dca_config: DCAConfig = field(default_factory=DCAConfig)
    allow_non_ascii: bool = False
    allow_special: bool = False
    filter_ids: bool = True

@dataclass
class DSMAttackStepResult(AttackStepResult):
    unfiltered_loss: float # discrete_obj_values + F_0
    continuous_loss: float
    duality_gaps: List[float] | float # inner_duality_gaps for DCA, duality_gap for PGM
    # store these info for DCA, set to None for PGM. Later might want to store a separate result for each inner step of DCA.
    inner_discrete_values: List[float] | None = None
    inner_discrete_values_filtered: List[float] | None = None
    inner_continuous_values: List[float] | None = None
    inner_times: List[float] | None = None  
    inner_flops: List[int] | None = None


def _masked_cross_entropy(
    shift_logits: Tensor,
    shift_labels: Tensor,
    logit_mask: Tensor,
) -> Tensor:
    """Cross-entropy loss on selected tokens.

    Args:
        shift_logits: logits of shape (batch_size, seq_len - 1, vocab_size)
        shift_labels: labels of shape (batch_size, seq_len - 1); logits of token
            i predicts label i (token i+1)
        logit_mask: mask where to apply loss to of shape (seq_len -1,),
            shared across the batch

    Returns:
        loss: Tensor of shape (batch_size,)
    """
    batch_size, _, vocab_size = shift_logits.shape
    if not logit_mask.any():
        raise ValueError(
            "logit_mask selects no positions; cross-entropy is undefined. "
            "Check target_mask / attack_mask and conversation layout."
        )
    sel_logits = shift_logits[:, logit_mask, :].contiguous()  # (batch_size, num_selected_tokens, vocab_size)
    sel_labels = shift_labels[:, logit_mask].contiguous()  # (batch_size, num_selected_tokens)
    flat_loss = torch.nn.functional.cross_entropy(
        sel_logits.view(-1, vocab_size),  # flatten since cross-entropy expects class dimension to be 1
        sel_labels.view(-1),
        reduction="none",
    )
    loss = flat_loss.view(batch_size, -1).mean(dim=-1)
    return loss


@torch.no_grad()
def compute_loss(
    # logits: Tensor, #keeping this in case want to revert to logits input and do fwd pass elsewhere
    model: PreTrainedModel,
    attack_ids: Tensor,
    original_tokens: Tensor,
    target_mask: torch.BoolTensor,
    attack_mask: torch.BoolTensor,
    lm_reg_weight: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Computes the cross-entropy loss on target tokens (-log p(y|q,x)) plus
    language-model regularizer lm_reg_weight * cross-entropy loss on attack
    tokens (-log p(x|q))

    Args:
        model: PreTrainedModel
        attack_ids: the attack token ids to evaluate. Tensor of shape
            (batch_size, n_optim_tokens)
        original_tokens: token ids for the full conversation. Tensor of shape
            (seq_len,)
        target_mask: bool mask of shape (seq_len,); True on tokens to apply
            loss to (target tokens shifted by one to the left).
        #TODO: maybe simpler to not shift target_mask earlier
        attack_mask: bool mask of shape (seq_len,); True on attack tokens.
        lm_reg_weight: Multiplier for the language-model regularizer.

    Returns:
        loss: Tensor of shape (batch_size,)
        flops: Tensor of shape (batch_size,), number of flops for the forward pass per attack_id`
        Same scalar estimate for all since same sequence length.

    """
    # TODO: if we revert to logits inputs, put back description logits: logits outputs for the full conversation. Tensor of shape (batch_size, seq_len, vocab_size)

    input_ids = original_tokens.unsqueeze(0).repeat(attack_ids.shape[0], 1)  # (batch_size, seq_len)
    input_ids[:, attack_mask] = attack_ids
    # TODO: add KV caching as done in GCG.
    # use float32 for logits to avoid issues in optimization with lower precision
    # logits device can differ from masks when using several GPUs, move it to same device
    logits = model(input_ids).logits.to(dtype=torch.float32)
    logits = logits.to(device=model.device)
    flops = get_flops(model, input_ids.shape[1], 0, "forward") # flops estimate for one attack_id

    # logits of token i-1 predicts token i
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    tgt_logit_mask = target_mask[:-1]

    loss = _masked_cross_entropy(shift_logits, shift_labels, tgt_logit_mask)
    if lm_reg_weight > 0.0:
        assert attack_mask is not None, "attack_mask is required when lm_reg_weight > 0.0"
        atk_logit_mask = attack_mask[1:]  # shift to the left
        reg_loss = _masked_cross_entropy(shift_logits, shift_labels, atk_logit_mask)
        loss += lm_reg_weight * reg_loss


    # TODO: If we add KV caching, maybe add these lines as done in GCG compute_candidates_loss to free memory?
    # Should check if this is actually helpful.
    # del outputs
    # gc.collect()
    # torch.cuda.empty_cache()

    return loss, torch.tensor(flops, device=loss.device, dtype=loss.dtype).expand_as(loss)

def compute_loss_with_max_batchsize(
    model: PreTrainedModel,
    attack_ids: Tensor,
    original_tokens: Tensor,
    target_mask: torch.BoolTensor,
    attack_mask: torch.BoolTensor,
    lm_reg_weight: float = 0.0,
) -> Tuple[Tensor, int]:
    """Wrap compute_loss in with_max_batchsize only if batch_size is large enough to trigger OOM error
    to avoid unnecessary overhead of with_max_batchsize if batch_size is small.
    I did not encounter OOM error with eval_chain and eval_neighbors which have batch_size n*b and 2*n*b 
    respectively (2*n*b = 1040 for Llama-3.2-1B-Instruct with n=20). 
    I did get OOM error with eval_all_pairs which uses batch_size n*b*(n*b-1)/2.
    """
    batch_size = attack_ids.shape[0]
    compute_loss_fn = lambda attack_ids: compute_loss(model, attack_ids, original_tokens, target_mask, attack_mask, lm_reg_weight)
    if batch_size > 2**11: # adjust threshold as needed
        loss, flops = with_max_batchsize(compute_loss_fn, attack_ids)
        logging.info(f"flops output of with_max_batchsize has shape: {flops.shape}")
    else:
        loss, flops = compute_loss_fn(attack_ids)
    return loss, flops.sum().item()


def _find_min_gap_permutation(embedding_matrix: Tensor) -> Tuple[Tensor, float]:
    r"""Sort rows of embedding matrix based on their jth coordinate in non-decreasing order, 
    for j \in [d] with the largest minimal gap between adjacent rows, i.e.,  
    \max_{j \in [d]} \min_{i \in [k-1]} (E_{\sigma^j_{i+1}, j} - E_{\sigma^j_i, j}), 
    where \sigma^j is such that E_{\sigma_k, j} \geq \ldots \geq E_{\sigma_0, j}.
    Return reordered embedding matrix and the corresponding permutation.
    """
    k, d = embedding_matrix.shape
    max_min_gap = -float("inf")
    best_perm = torch.arange(k, device=embedding_matrix.device)
    best_j = None
    for j in range(d):
        perm = embedding_matrix[:, j].argsort(stable=True)
        gaps = embedding_matrix[perm, j].diff()
        min_gap = gaps.min().item()
        if min_gap > max_min_gap:
            max_min_gap = min_gap
            best_perm = perm
            best_j = j
    logging.info(f"Max min gap: {max_min_gap:.6g}, achieved at j = {best_j}")

    return best_perm, max_min_gap


def _randomly_permute_embeddings(embedding_matrix: Tensor) -> Tuple[Tensor, Tensor, float, Tensor]:
    """
    Generate a random unit vector w in R^d, and permute the rows of the embedding matrix 
    according to the non-decreasing order of their projections onto w.
    """
    k, d = embedding_matrix.shape
    max_retries = 1  # increase if needed
    # we need to use float64 precision, otherwise couldn't find valid w even after 100 attempts 
    # for Llama-3.2-1B-Instruct, 1st conversation in adv_behaviors 
    # this likely will lead again to large H(x) values... 
    for _ in range(max_retries):
        w = torch.randn(d, dtype=torch.float64, device=embedding_matrix.device)
        w = w / w.norm()
        projections = embedding_matrix.double() @ w
        if projections.unique().numel() == k: 
            logging.info(f"Found a random direction w with distinct projections for all {k} rows.")
            break
    else:
        raise ValueError(
            f"Could not find a random direction w with distinct projections for all {k} rows "
            f"after {max_retries} attempts."
        )

    perm = projections.argsort(stable=True)
    permuted_embedding_projections = projections[perm]
    min_gap = permuted_embedding_projections.diff().min().item()
    logging.info(f"Min gap achieved with w: {min_gap:.6g}")
    return w, perm, min_gap, permuted_embedding_projections


def _valid_embeddings(
    model: PreTrainedModel,
    valid_token_ids: Tensor,
    device: torch.device | str | None = None,
) -> Tensor:
    embedding_layer = model.get_input_embeddings()
    E = embedding_layer.weight[valid_token_ids].detach().float()
    if hasattr(embedding_layer, "embed_scale"):
        E = E * embedding_layer.embed_scale.float()
    if device is not None:
        E = E.to(device)
    return E


def _permuted_valid_projections(
    model: PreTrainedModel,
    valid_token_ids: Tensor,
    perm: Tensor,
    w: Tensor,
) -> Tensor:
    E = _valid_embeddings(model, valid_token_ids, device=perm.device)
    return (E @ w.to(device=perm.device, dtype=E.dtype))[perm]

def _find_embeddings_dual_cone_w(
    model: PreTrainedModel,
    valid_token_ids: Tensor,
    solve_lp: bool = False,
    seed: int = 0,
    save_file: str | None = None,
) -> Tuple[Tensor, float, Tensor, Tensor, Tensor | None]:
    """Find a vector w in R^d in the interior of the dual cone of differences of
    adjacent embedding vectors.

    Let U be the matrix with rows {E_{i + 1} - E_i : i in V} where 
    E_i is the i-th row of the model embedding matrix (restricted to valid_token_ids) 
    Solve the LP problem:

        max_{t >= 0, w in [-1, 1]^d} t  subject to  U w >= t

    If save_file is set, cache results to that path.
    #TODO: update docstring to reflect new version of permuting embeddings
    """

    embedding_matrix = _valid_embeddings(model, valid_token_ids, device="cpu")

    k, d = embedding_matrix.shape
    # n_unique_rows = np.unique(embedding_matrix, axis=0).shape[0]
    # assert n_unique_rows == k, (f"Embedding matrix has {k - n_unique_rows} duplicate row(s).")

    # perm, min_gap = _find_min_gap_permutation(embedding_matrix)
    torch.manual_seed(seed) # reset seed to ensure reproducibility of resulting w, perm for a given seed
    w, perm, min_gap, permuted_embedding_projections = _randomly_permute_embeddings(embedding_matrix)
    embedding_matrix = embedding_matrix[perm]
    perm = perm.to(device=model.device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(k, device=model.device)

    # We can simply use random w, but probably better to use w that maximizes the min gap 
    # for this permuted embedding matrix. 
    # LP took > 3hrs to solve, so for now let's use random w.
    # TODO: can try to solve problem with SVM instead of LP

    if solve_lp:
        embedding_matrix = embedding_matrix.numpy()
        # Solve LP with linprog: min c^T x subject to A_ub x <= b_ub, x in bounds.
        # x = [w_0, ..., w_{d-1}, t], c = [0, ..., 0, -1], A_ub = [-U, 1], b_ub = 0, 
        # bounds = [-1, 1]^d x [0, None]. 
        A_ub = np.empty((k-1, d + 1), dtype=np.float32)
        A_ub[:, :d] = embedding_matrix[:-1] - embedding_matrix[1:]
        A_ub[:, d] = 1.0
        del embedding_matrix
        
        c = np.zeros(d + 1, dtype=np.float32)
        c[-1] = -1.0
        b_ub = np.zeros(k - 1, dtype=np.float32)
        bounds = [(-1.0, 1.0)] * d + [(0.0, None)]

        logging.info(
            f"Solving LP with {d + 1} variables and {k-1} constraints"
        )
        lp_result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"disp": True})  # set disp to False when done debugging
        if not lp_result.success:
            raise RuntimeError(f"LP failed: {lp_result.message}")

        t_opt = float(-lp_result.fun)
        assert t_opt >= 0.0, "t* should be non-negative."
        assert t_opt >= min_gap, "t* should be greater than or equal to the min gap."
        w_opt = torch.tensor(lp_result.x[:-1], dtype=torch.float32, device=model.device)
        w_opt = w_opt / t_opt # can recover t_opt from ||w_opt||_\infty = 1/t_opt
        if t_opt == 0.0:
            logging.info("Did not find w in the interior of the dual cone, t* = 0.0.")
        else:
            logging.info(f"Found w in the interior of the dual cone with t* = {t_opt:.6g}.")

        lambdas = -lp_result.ineqlin.marginals  # dual variables / Lagrange multipliers
        if not (lambdas >= 0.0).all():
            logging.warning("Lambdas are not non-negative.")
        if abs(lambdas.sum() - 1.0) > 1e-12:
            logging.warning(f"Lambdas do not sum to 1.")
        
        permuted_embedding_projections = None # maybe compute them here too?
    else:
        lp_result = None
        t_opt = min_gap
        w_opt = w.to(device=model.device) / t_opt # can recover t_opt = min_gap from ||w_opt||_2 = 1/t_opt
        permuted_embedding_projections = permuted_embedding_projections.to(device=model.device) / t_opt 

    if save_file is not None:
        os.makedirs(os.path.dirname(f"{save_file}/{seed}.pt"), exist_ok=True)
        torch.save(
            {"w_opt_scaled": w_opt, "t_opt": t_opt, "lp_result": lp_result, "perm": perm, "inv_perm": inv_perm, "min_gap": min_gap},
            save_file,
        )

    return w_opt, perm, inv_perm, permuted_embedding_projections

class DSMAttack(Attack):
    def __init__(self, config: DSMConfig):
        super().__init__(config)

    @torch.no_grad()
    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, dataset: PromptDataset) -> AttackResult:
        # TODO: add time tracking
        # --- Prepare Conversations ---
        tokens, attack_masks, target_masks, conversations = self._prepare_dataset(dataset, tokenizer)
        logging.info(f"Prepared {len(conversations)} conversations for attack")

        # --- Build Valid Vocab ---
        self._build_valid_vocab(tokenizer, model)

        # --- Find w to use in DR-submodular decomposition ---
        if self.config.optimizer == "dca":
            model_name_safe = model.name_or_path.replace("/", "-")
            save_file = f"{self.config.dca_config.dsm_cache_dir}/{model_name_safe}/embeddings_dual_cone_w/{self.config.seed}.pt"
            if os.path.exists(save_file):
                logging.info(f"Loading w found in the dual cone of forward differences of embedding vectors from {save_file}")
                cache = torch.load(save_file,  map_location=model.device, weights_only=False)
                self._embeddings_dual_cone_w = cache["w_opt_scaled"]
                self._embeddings_perm = cache["perm"]
                self._embeddings_inv_perm = cache["inv_perm"]
                self._permuted_embedding_projections = None # will be computed below
            else:
                logging.info(f"Searching for w in the interior of the dual cone of forward differences of embedding vectors and saving it to {save_file}")
                time_start = time.time()
                self._embeddings_dual_cone_w, self._embeddings_perm, self._embeddings_inv_perm, self._permuted_embedding_projections = _find_embeddings_dual_cone_w(
                    model, self.valid_token_ids, seed=self.config.seed, save_file=save_file
                )
                time_end = time.time()
                logging.info(f"Time taken to find w: {time_end - time_start:.2f} seconds")
            if self._permuted_embedding_projections is None:
                self._permuted_embedding_projections = _permuted_valid_projections(model, self.valid_token_ids, self._embeddings_perm, self._embeddings_dual_cone_w)
        else:
            # define identity embedding permutation to be used by PGM
            # TODO: it's interesting to check if PGM performs better with DCA's embedding permutation.
            self._embeddings_perm  = torch.arange(self.valid_vocab_size, device=model.device)
            self._embeddings_inv_perm = torch.arange(self.valid_vocab_size, device=model.device)
            

        runs = []
        for idx, conversation in enumerate(conversations):
            runs.append(self._attack_single_conversation(model, tokenizer, conversation, tokens[idx], attack_masks[idx], target_masks[idx], idx))

        return AttackResult(runs=runs)

    def _attack_single_conversation(self, model, tokenizer, conversation, tokens, attack_mask, target_mask, idx) -> SingleAttackRunResult:
        #TODO: Compute the KV Cache for tokens that appear before the optimized tokens as done in GCG.
        #TODO: add early stopping if exact match found as done in GCG.
        #TODO: move things like building loss_fn, filter_fn, initialization to separate functions
        logging.info(f"Starting attack for conversation: {conversation}")
        t_start = time.time()
        # --- Optimize Attack ---
        device = model.device
        logging.info(f"model device: {device}")
        tokens = tokens.to(device)
        attack_mask = attack_mask.to(device)
        target_mask = target_mask.to(device)
        n_optim_tokens = int(attack_mask.sum().item())
        # Initialize with the token ids of optim_str_init
        # TODO: experiment with different initial solutions (see notes.md)
        optim_ids_init = tokens[attack_mask].detach().clone().unsqueeze(0) # (1, n_optim_tokens)
        reduced_ids_init = self.valid_token_id_to_reduced_idx[optim_ids_init]
        invalid_optim_ids = optim_ids_init[reduced_ids_init == -1]
        if invalid_optim_ids.numel() > 0:
            raise ValueError(
                f"Initial attack ids contains {invalid_optim_ids.numel()} not allowed token id(s) "
                f"e.g. {invalid_optim_ids[:5].tolist()}."
            )

        # apply inverse embedding permutation to reduced initial attack ids
        inv_perm_ids_init = self._embeddings_inv_perm[reduced_ids_init]

        # define loss_fn over V^n where V = {0, 1, ..., valid_vocab_size - 1} and n = n_optim_tokens
        # on the permuted embedding matrix. Need to apply embeddings_perm (defined on V^n, so should be applied first)  
        # and map back to original token ids
        loss_fn = lambda attack_ids: compute_loss_with_max_batchsize(
            model, self.valid_token_ids[self._embeddings_perm[attack_ids]], tokens, target_mask, attack_mask, self.config.lm_reg_weight
        )
        zero_attack_ids = torch.zeros_like(inv_perm_ids_init)
        F_0, F_0_flops = loss_fn(zero_attack_ids)
        logging.info(f"Loss at zero F(0): {F_0.item():.4f}") # this changes with permutation of embeddings
        # normalize F(0) = 0
        def F_batch(attack_ids):
            loss, flops = loss_fn(attack_ids)
            return loss - F_0, flops

        # define filter function
        filter_fn = None
        filter_zero = False
        if self.config.filter_ids: 
            if self.config.placement == "suffix":
                filter_fn = lambda attack_ids: filter_suffix(tokenizer, conversation, [[None, self.valid_token_ids[self._embeddings_perm[attack_ids]].cpu()]], False)
                # check if zero_attack_ids is reachable
                retained_idx = filter_fn(zero_attack_ids)
                if not retained_idx:
                    filter_zero = True
                    logging.warning("Zero attack ids is not reachable from any input string. Will not round to zero during optimization.") 
            else:
                # TODO: adapt filter function for other placements
                raise ValueError(f"Filtering for {self.config.placement} placement not supported yet.")

        F_set_batch = EneSubmodularSetFnReduction(F_batch, self.valid_vocab_size, n_optim_tokens, device, filter_fn, filter_zero)
       
        # TODO: have a common clean interface for optimizers 
        if self.config.optimizer == "pgm":
            # run PGM with initial optim_ids as initial solution (assume F is approximately submodular)       
            _, _, discrete_obj_values, discrete_obj_values_filtered, continuous_obj_values, duality_gaps, discrete_sols_filtered, \
            times, flops = pgm_lovasz(
                F_set_batch,
                inv_perm_ids_init,
                self.config.num_steps,
                self.config.pgm_config.L,
                tie_break=self.config.pgm_config.tie_break,
                gap_tol=None,
            )

            plot_pgm_curves(discrete_obj_values, discrete_obj_values_filtered, continuous_obj_values, duality_gaps, F_0.item())

        elif self.config.optimizer == "dca":
            dca_config = self.config.dca_config
            if dca_config.hessian_upperbd == "hessian_upperbd_at_zero":
                logging.info(f"DR-submodular decomposition using Hessian upper bound at zero") 
                F_singleton_vals, flops_F_singletons = F_set_batch.eval_singletons()
                model_name_safe = model.name_or_path.replace("/", "-")
                save_file = f"{dca_config.dsm_cache_dir}/{model_name_safe}/hessian_upperbd_at_zero_{idx}.pt"
                if os.path.exists(save_file):
                    logging.info(f"Loading Hessian upper bound at zero from {save_file}")
                    cache = torch.load(save_file, map_location=device)
                    hessian_upperbd = cache["hessian_upperbd"]
                    flops_hessian_bd = cache["flops"]
                    time_hessian_bd = cache["time_taken"]
                else:
                    logging.info(f"Computing Hessian upper bound at zero and saving to {save_file}")
                    hessian_upperbd, flops_hessian_bd, time_taken = F_set_batch.hessian_upperbd_at_zero(F_singleton_vals, save_file=save_file)
                    logging.info(f"Time taken: {time_taken}")
                    time_hessian_bd = 0 # time already included      

                L_F, flops_L_F = F_set_batch.singletons_L_bound(F_singleton_vals) # flops_L_F=0 when singleton_vals are provided
            else:
                hessian_upperbd = dca_config.hessian_upperbd
                L_F, flops_L_F = F_set_batch.singletons_L_bound() 
                logging.info(f"DR-submodular decomposition using scalar Hessian upper bound {hessian_upperbd}") 

            
            # TODO: run DCA for more num_outer_steps if not converged and actual number of inner steps ran in total < num_steps
            num_outer_steps = self.config.num_steps // dca_config.num_inner_steps
            assert num_outer_steps >=1, "num_outer_steps = num_steps // num_inner_steps must be at least 1."
            # decompose F into the difference of two DR-submodular functions G and H
            G_batch, H_batch = DR_submodular_decomposition(
                F_set_batch.lattice_fn,
                hessian_upperbd,
                self._permuted_embedding_projections,
            )
            G_set_batch = SetFnReduction(G_batch, F_set_batch.map, filter_fn, filter_zero)
            H_set_batch = SetFnReduction(H_batch, F_set_batch.map, filter_fn, filter_zero)
      
            # H_set is a monotone non-increasing function so L_H = - H_set([n] x [b]) = - H((k-1) 1) where k = valid_vocab_size
            # TODO: add flops_L_H, flops_L_F, flops_hessian_bd, flops_F_singletons to flops count of first step?
            H_max, flops_L_H= H_batch(torch.full((1, n_optim_tokens), self.valid_vocab_size - 1, dtype=torch.long, device=device))
            L_H = -H_max.item()
            L_G = L_F + L_H

            # run DCA with initial optim_ids as initial solution
            discrete_obj_values, discrete_obj_values_filtered, continuous_obj_values, discrete_sols_filtered, times, flops, \
            inner_discrete_values, inner_discrete_values_filtered, inner_continuous_values, inner_duality_gaps, inner_times, inner_flops = \
            dca_dsm(
                F_set_batch,
                G_set_batch,
                H_set_batch,
                inv_perm_ids_init,
                num_outer_steps,
                dca_config.num_inner_steps,
                dca_config.inner_solver,
                outer_tol=dca_config.outer_tol,
                inner_gap_tol=dca_config.inner_gap_tol,
                tie_break=dca_config.tie_break,
                L_G=L_G,
            )
            
            for i in range(len(inner_discrete_values)): # plot pgm curves for each outer iteration
                plot_pgm_curves(inner_discrete_values[i], inner_discrete_values_filtered[i], inner_continuous_values[i], inner_duality_gaps[i], F_0.item(), outer_step=i)

        else:
            raise ValueError(f"Optimizer {self.config.optimizer} not supported. Must be 'pgm' or 'dca'.")

        # Drop steps with no valid filtered solution
        valid_idx = [i for i in range(len(discrete_obj_values_filtered)) if math.isfinite(discrete_obj_values_filtered[i])]
        if not valid_idx:
            raise ValueError("Every optimization step has no valid filtered solution.")
        discrete_sols_filtered = discrete_sols_filtered[valid_idx]
        # discrete_obj_values_filtered = [discrete_obj_values_filtered[i] for i in valid_idx]
        # times = [times[i] for i in valid_idx]
        # flops = [flops[i] for i in valid_idx]

        best_sol_idx_filtered = min(range(len(valid_idx)), key=lambda i: discrete_obj_values_filtered[valid_idx[i]]) 
        flops[valid_idx[0]] += F_0_flops

        # map back to original token ids and decode to strings
        optim_ids = self.valid_token_ids[self._embeddings_perm[discrete_sols_filtered]]
        optim_strings = tokenizer.batch_decode(optim_ids.cpu())  # decode handles batching in v5.3+, keeping batch_decode to support older versions
        losses = [val + F_0.item() for val in discrete_obj_values_filtered]
        unfiltered_losses = [val + F_0.item() for val in discrete_obj_values]
        continuous_losses = [val + F_0.item() for val in continuous_obj_values]

        logging.info(
            f"Optimization loop completed. Best attack (step {valid_idx[best_sol_idx_filtered]}): {optim_strings[best_sol_idx_filtered][:80]!s}. "
            f"Optimization time: {time.time() - t_start:.2f}s."
        )
        # logging.info(f"Optimization loop completed. Best attack: {optim_strings[-1][:80]} with loss: {losses[-1]}." # for now we're not saving best loss

        # --- Generate Completions ---
        # get tokens of attack conversations with optimized attack strings and empty assistant content
        prompt_token_list = []
        attack_conversations = []
        for idx, attack in enumerate(optim_strings):
            try:
                parts, attack_conversation = self._prepare_single_conversation(conversation, tokenizer, attack, generation=True)
            except TokenMergeError: 
                if self.config.filter_ids:
                    raise ValueError(f"TokenMergeError encountered for attack: {attack} at step {idx}. This should not happen when filtering is enabled.")
                else:
                    logging.warning(f"TokenMergeError encountered for attack: {attack} at step {idx}. Skipping it.")
                    valid_idx.pop(idx)
                    continue

            prompt_token_list.append(torch.cat(parts[:5]))
            attack_conversations.append(attack_conversation)

        optim_strings = [optim_strings[i] for i in valid_idx]
        t_start_gen = time.time()
        completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=prompt_token_list,  # Generate from the prompt tokens
            # embedding_list=embedding_list, # Or generate from the prompt embeddings
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=len(optim_strings),  # change to size of the full dataset if we switch to batched optimization
        )
        t_end_gen = time.time()
        gen_time_total = t_end_gen - t_start_gen
        logging.info(
            f"Generated {len(completions)}x{self.config.generation_config.num_return_sequences} completions. "
            f"Generation time: {gen_time_total:.2f}s."
        )

        t_end = time.time() 

        # --- Assemble Results ---
        # model_completions, model_input, and model_input_tokens have only valid steps aligned with optim_strings
        # all other results lists have results for all steps including invalid ones
        steps_results = []
        for i in range(len(optim_strings)):
            step_result = DSMAttackStepResult(
                step=valid_idx[i],
                model_completions=completions[i],
                time_taken=times[valid_idx[i]],
                loss=losses[valid_idx[i]],
                unfiltered_loss=unfiltered_losses[valid_idx[i]],
                continuous_loss=continuous_losses[valid_idx[i]],
                flops=flops[valid_idx[i]],
                model_input=attack_conversations[i],
                model_input_tokens=prompt_token_list[i].tolist(),
                inner_discrete_values=inner_discrete_values[valid_idx[i]] if self.config.optimizer == "dca" else None,
                inner_discrete_values_filtered=inner_discrete_values_filtered[valid_idx[i]] if self.config.optimizer == "dca" else None,
                inner_continuous_values=inner_continuous_values[valid_idx[i]] if self.config.optimizer == "dca" else None,
                duality_gaps=inner_duality_gaps[valid_idx[i]] if self.config.optimizer == "dca" else duality_gaps[valid_idx[i]],
                inner_times=inner_times[valid_idx[i]] if self.config.optimizer == "dca" else None,
                inner_flops=inner_flops[valid_idx[i]] if self.config.optimizer == "dca" else None,
            )
            steps_results.append(step_result)

        run_result = SingleAttackRunResult(
            original_prompt=conversation,
            steps=steps_results,
            total_time=t_end - t_start + time_hessian_bd if self.config.optimizer == "dca" else 0,
        )
        return run_result


    def _build_valid_vocab(self, tokenizer, model):
        # get disallowed_ids as done in GCG
        not_allowed_ids = get_disallowed_ids(tokenizer, self.config.allow_non_ascii, self.config.allow_special).to(model.device)
        num_embeddings = model.get_input_embeddings().weight.size(0)
        # drop disallowed_ids >= num_embeddings; some models like gemma-3 add extra tokens that do not have embeddings
        self.not_allowed_ids = not_allowed_ids[not_allowed_ids < num_embeddings]
        self.vocab_size = num_embeddings
        logging.info(f"Number of embeddings: {num_embeddings}, Tokenizer vocab size: {len(tokenizer)}")  # to check if they match

        # get valid token ids to map from V = {0, 1, ..., valid_vocab_size - 1} to V_original = {0, 1, ..., vocab_size - 1}
        valid_tokens_mask = torch.ones(self.vocab_size, dtype=torch.bool, device=model.device)
        if self.not_allowed_ids is not None and self.not_allowed_ids.numel() > 0:
            valid_tokens_mask[self.not_allowed_ids.to(model.device)] = False

        self.valid_token_ids = torch.nonzero(valid_tokens_mask, as_tuple=False).squeeze(1)
        self.valid_vocab_size = self.valid_token_ids.numel()

        # build inverse map: V_original -> V or -1 if disallowed
        self.valid_token_id_to_reduced_idx = torch.full(
            (self.vocab_size,), -1, dtype=torch.long, device=model.device
        )
        self.valid_token_id_to_reduced_idx[self.valid_token_ids] = torch.arange(
            self.valid_vocab_size, device=model.device, dtype=torch.long
        )

        logging.info(
            f"Valid vocab size: {self.valid_vocab_size} (excluded {int(self.not_allowed_ids.numel())} ids)"
        )


    # copied from PGDDiscreteAttack. Added assert for single-turn conversation and removed padding.
    # if we're not doing batched optimization, no point preparing full dataset, can call _prepare_single_conversation
    # inside _attack_single_conversation. For now let's keep this in case we switch to batched optimization.
    def _prepare_dataset(self, dataset, tokenizer) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Conversation]]:
        all_tokens = []
        all_attack_masks = []
        all_target_masks = []
        all_conversations = []

        for conversation in dataset:
            assert len(conversation) == 2, "DSM attack currently assumes single-turn conversation."

            all_conversations.append(conversation)
            try:
                parts, _ = self._prepare_single_conversation(
                    conversation, tokenizer, self.config.optim_str_init
                )
            except TokenMergeError:
                logging.warning("TokenMergeError encountered, retrying with added space.")
                parts, _ = self._prepare_single_conversation(
                    conversation, tokenizer, " " + self.config.optim_str_init
                )

            pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks = parts
            tokens = torch.cat(parts)

            # TODO: do we really need to use masks in our case? maybe better to store ids as in GCG?
            # build attack_mask (tokens to optimize) and target_mask (tokens to apply loss to)
            attack_mask = torch.zeros_like(tokens, dtype=torch.bool)
            offset = pre_toks.size(0)
            attack_mask[offset : offset + attack_prefix_toks.size(0)] = True
            offset += attack_prefix_toks.size(0) + prompt_toks.size(0)
            attack_mask[offset : offset + attack_suffix_toks.size(0)] = True

            target_mask = torch.zeros_like(tokens, dtype=torch.bool)
            target_start_idx = len(tokens) - target_toks.size(0)
            target_mask[target_start_idx:] = True
            # TODO: maybe better to shift when computing loss and not here for clarity?
            # unless we need this shifted version elsewhere?
            target_mask = target_mask.roll(-1, 0)  # shift to the left
            target_mask[-1] = False

            all_tokens.append(tokens)
            all_attack_masks.append(attack_mask)
            all_target_masks.append(target_mask)

        # remove padding for now since we're not doing batched optimization.
        # TODO: add padding back if we switch to batched optimization, but not here inside attack_batch and just sort here
        # we also will need an attention_mask in the forward pass as done in PGD Discrete in that case.
        # all_tokens = pad_sequence(all_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        # all_target_masks = pad_sequence(all_target_masks, batch_first=True)
        # all_attack_masks = pad_sequence(all_attack_masks, batch_first=True)

        return all_tokens, all_attack_masks, all_target_masks, all_conversations

    def _prepare_single_conversation(
        self, conversation, tokenizer, optim_str, generation=False
    ) -> Tuple[
        tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor],
        Conversation,
    ]:
        # insert optimizable string optim_str in user content according to placement and get tokens of conversation split into six parts
        assistant_content = conversation[1]["content"] if not generation else ""
        if self.config.placement == "suffix":
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + optim_str},
                {"role": "assistant", "content": assistant_content},
            ]
        elif self.config.placement == "prefix":
            attack_conversation = [
                {"role": "user", "content": optim_str + conversation[0]["content"]},
                {"role": "assistant", "content": assistant_content},
            ]
        elif self.config.placement == "prefix_suffix":
            raise ValueError(f"Prefix_suffix placement not supported yet for DSM attack.")
            # attack_conversation = [
            #     {"role": "user", "content": optim_str_prefix + conversation[0]["content"] + optim_str_suffix},
            #     {"role": "assistant", "content": assistant_content},
            # ]
        elif self.config.placement == "prompt":
            attack_conversation = copy.deepcopy(conversation)
            if generation:
                # matches _reconstruct_attack_conversation in PGDDiscreteAttack
                # TODO: not sure why they re-add original prompt, ask authors
                attack_conversation[0]["content"] = optim_str + attack_conversation[0]["content"]
                attack_conversation[1]["content"] = ""
            else:
                # matches _prepare_single_conversation in PGDDiscreteAttack
                # initial optim_str is not used here
                conversation = copy.deepcopy(conversation)
                conversation[0]["content"] = ""  # the whole prompt is optimized
        else:
            raise ValueError(f"Invalid placement: {self.config.placement}")
        parts = prepare_conversation(tokenizer, conversation, attack_conversation)[0]  # assumes single-turn conversation

        return parts, attack_conversation

def plot_pgm_curves(discrete_obj_values, discrete_obj_values_filtered, continuous_obj_values, duality_gaps, F_0=0.0, outer_step=None):
    # figure will be saved in Hydra run directory ${root_dir}/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}/

    discrete_obj_values = [val + F_0 for val in discrete_obj_values]
    discrete_obj_values_filtered = [val + F_0 for val in discrete_obj_values_filtered]
    continuous_obj_values = [val + F_0 for val in continuous_obj_values]

    steps_axis = range(len(discrete_obj_values))
    fig, (ax_obj, ax_gap) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_obj.plot(steps_axis, discrete_obj_values, label=r"Discrete $F(x^t)$", marker="o", ms=3)
    ax_obj.plot(steps_axis, discrete_obj_values_filtered, label=r"Filtered Discrete $F(\tilde{x}^t)$", marker="x", ms=3)
    ax_obj.plot(steps_axis, continuous_obj_values, label=r"Lovasz $f_L(X^t)$", marker="s", ms=3)
    ax_obj.set_ylabel("Objective")
    ax_obj.legend(loc="best")
    ax_obj.grid(True, alpha=0.3)
    ax_gap.plot(steps_axis, duality_gaps, color="black", label="Duality gap", marker="^", ms=3)
    ax_gap.set_xlabel("PGM iteration")
    ax_gap.set_ylabel("Duality gap")
    ax_gap.grid(True, alpha=0.3)
    if outer_step is not None:
        fig.suptitle(f"PGM objective values and duality gap for DCA outer step {outer_step}")
        filename = f"pgm_curves_dca_step_{outer_step}.png"
    else:
        fig.suptitle("PGM objective values and duality gap")
        filename = "pgm_curves.png"
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)