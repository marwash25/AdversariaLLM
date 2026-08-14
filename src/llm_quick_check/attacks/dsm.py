"""Difference of submodular minimization (DSM) attack"""
import copy
from math import isfinite
from pathlib import Path
import sys
import time
import logging
import gc
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Literal
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from ..dataset import PromptDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult
from ..lm_utils import prepare_conversation, TokenMergeError, generate_ragged_batched, get_flops, get_disallowed_ids, filter_suffix, with_max_batchsize
from ..types import Conversation
from .submodular_utils import EneSubmodularSetFnReduction, EneReductionMap, DR_submodular_decomposition, SetFnReduction, pgm_lovasz, dca_dsm
from .embeddings_dual_cone import _find_embeddings_dual_cone_w, _sorted_valid_projections


@dataclass
class DCAConfig:
    """Config for the DCA optimizer."""
    hessian_upperbd: float | Literal["hessian_upperbd_at_zero"] = "hessian_upperbd_at_zero" # runs PGM in that case
    dsm_cache_dir: str = str(Path(__file__).resolve().parent / "dsm_cache")
    outer_tol: float = 1e-5
    inner_gap_tol: float = 1e-4
    # num_outer_steps: will be set to num_steps / num_inner_steps 
    num_inner_steps: int = 1
    inner_solver: Literal["pgm"] = "pgm"
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
            save_path = Path(self.config.dca_config.dsm_cache_dir) / model_name_safe / f"embeddings_dual_cone_w_seed{self.config.dca_config.seed_w}.pt"
            
            if not self.config.dca_config.overwrite_cache and save_path.exists():
                    logging.info(f"Loading w found in the dual cone of forward differences of embedding vectors from {save_path}")
                    cache = torch.load(save_path,  map_location=model.device, weights_only=False)
                    self._embeddings_dual_cone_w = cache["w_opt_scaled"]
                    self._embeddings_perm = cache["perm"]
                    self._embeddings_inv_perm = cache["inv_perm"]
                    self._sorted_embedding_projections = None # will be computed below
            else:
                logging.info(f"Searching for w in the interior of the dual cone of forward differences of embedding vectors and saving it to {save_path}")
                dual_cone_config = self.config.dca_config.dual_cone_config
                time_start = time.time()
                self._embeddings_dual_cone_w, self._embeddings_perm, self._embeddings_inv_perm, self._sorted_embedding_projections = _find_embeddings_dual_cone_w(
                    model, self.valid_token_ids, init_w = dual_cone_config.init_w, solver=dual_cone_config.solver, solver_config=dual_cone_config.solver_config, 
                    seed=self.config.dca_config.seed_w, save_file=save_path
                )
                time_end = time.time()
                logging.info(f"Time taken to find w: {time_end - time_start:.2f} seconds")
                return AttackResult(runs=[]) #TODO: remove when done testing

            if self._sorted_embedding_projections is None:
                self._sorted_embedding_projections = _sorted_valid_projections(model, self.valid_token_ids, self._embeddings_perm, self._embeddings_dual_cone_w)
        else:
            # define identity embedding permutation to be used by PGM
            # TODO: it's interesting to check if PGM performs better with DCA's embedding permutation.
            self._embeddings_perm  = torch.arange(self.valid_vocab_size, device=model.device)
            self._embeddings_inv_perm = torch.arange(self.valid_vocab_size, device=model.device)
            

        runs = []
        for idx, conversation in enumerate(conversations):
            stable_idx = int(dataset.idx[idx].item()) # conversation index in the original dataset (before shuffle)
            runs.append(self._attack_single_conversation(model, tokenizer, conversation, tokens[idx], attack_masks[idx], target_masks[idx], stable_idx))

        return AttackResult(runs=runs)

    def _attack_single_conversation(self, model, tokenizer, conversation, tokens, attack_mask, target_mask, stable_idx) -> SingleAttackRunResult:
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
            time_hessian_bd = 0
            if dca_config.hessian_upperbd == "hessian_upperbd_at_zero":
                logging.info(f"DR-submodular decomposition using Hessian upper bound at zero") 
                F_singleton_vals, flops_F_singletons = F_set_batch.eval_singletons() 
                model_name_safe = model.name_or_path.replace("/", "-")
                # F changes with permutation of embeddings, which is fixed per seed, so we need to recompute hessian_upperbd for each seed
                # TODO: we also need to recompute if anything else changes F, e.g., _embeddings_perm, optim_str_init, lm_reg_weight, normalized flag, 
                # attack placement, etc. We can store in saved file and validate on load. For now, these are fixed.
                save_path = Path(dca_config.dsm_cache_dir) / model_name_safe / f"hessian_upperbd_at_zero_idx{stable_idx}_seed{self.config.seed}.pt"
                if not dca_config.overwrite_cache and save_path.exists():
                    logging.info(f"Loading Hessian upper bound at zero from {save_path}")
                    cache = torch.load(save_path, map_location=device)
                    hessian_upperbd = cache["hessian_upperbd"]
                    flops_hessian_bd = cache["flops"]
                    time_hessian_bd = cache["time_taken"]
                else:
                    logging.info(f"Computing Hessian upper bound at zero and saving to {save_path}")
                    hessian_upperbd, flops_hessian_bd, time_taken = F_set_batch.hessian_upperbd_at_zero(singleton_vals=F_singleton_vals, save_file=save_path)
                    logging.info(f"Time taken to compute Hessian upper bound at zero: {time_taken}")

                L_F, flops_L_F = F_set_batch.singletons_L_bound(F_singleton_vals) # flops_L_F=0 when singleton_vals are provided
            else:
                hessian_upperbd = torch.tensor(dca_config.hessian_upperbd, device=device, dtype=torch.float32)
                L_F, flops_L_F = F_set_batch.singletons_L_bound() 
                logging.info(f"DR-submodular decomposition using scalar Hessian upper bound {hessian_upperbd}") 

            
            # TODO: run DCA for more num_outer_steps if not converged and actual number of inner steps ran in total < num_steps
            num_outer_steps = self.config.num_steps // dca_config.num_inner_steps
            assert num_outer_steps >=1, "num_outer_steps = num_steps // num_inner_steps must be at least 1."
            # decompose F into the difference of two DR-submodular functions G and H
            G_batch, H_batch = DR_submodular_decomposition(
                F_set_batch.lattice_fn,
                hessian_upperbd,
                self._sorted_embedding_projections,
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
        valid_idx = [i for i in range(len(discrete_obj_values_filtered)) if isfinite(discrete_obj_values_filtered[i])]
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
            f"Optimization loop completed. Best valid attack (step {valid_idx[best_sol_idx_filtered]}): {optim_strings[best_sol_idx_filtered][:80]!s}. "
            f"Optimization time: {time.time() - t_start:.2f}s."
        )
        # logging.info(f"Optimization loop completed. Best attack: {optim_strings[-1][:80]} with loss: {losses[-1]}." # for now we're not saving best loss

        # --- Generate Completions ---
        # get tokens of attack conversations with optimized attack strings and empty assistant content
        prompt_token_list = []
        attack_conversations = []
        gen_valid_idx: list[int] = []
        gen_optim_strings: list[str] = []

        for i, attack in enumerate(optim_strings):
            try:
                parts, attack_conversation = self._prepare_single_conversation(conversation, tokenizer, attack, generation=True)
            except TokenMergeError: 
                if self.config.filter_ids:
                    raise ValueError(f"TokenMergeError encountered for attack: {attack} at step {valid_idx[i]}. This should not happen when filtering is enabled.")
                else:
                    logging.warning(f"TokenMergeError encountered for attack: {attack} at step {valid_idx[i]}. Skipping it.")
                    continue

            # keep track of non-skipped attacks and their indices
            gen_valid_idx.append(valid_idx[i])
            gen_optim_strings.append(attack)
            prompt_token_list.append(torch.cat(parts[:5]))
            attack_conversations.append(attack_conversation)

        valid_idx = gen_valid_idx
        optim_strings = gen_optim_strings
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
            total_time=t_end - t_start + (time_hessian_bd if self.config.optimizer == "dca" else 0),
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