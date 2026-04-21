"""Dummy attack for testing/learning implementation of new attacks"""
import copy
import time
import logging
import sys
from tqdm import trange
from typing import Dict, List, Optional, Tuple, Callable, Any
import torch
from math import log2
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from ..dataset import PromptDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult
from ..lm_utils import prepare_conversation, TokenMergeError, generate_ragged_batched, get_flops, get_disallowed_ids
from ..types import Conversation


@dataclass
class DummyConfig: 
    """Config for the Dummy attack."""
    # not strictly necessary, but good for type checking config, clarity, 
    # and providing defaults if not specified in attacks.yaml
    name: str = "dummy"
    type: str = "discrete"
    version: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    placement: str = "suffix"
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    num_steps: int = 1
    lm_reg_weight: float = 0.0  # weight on -log p(x|q) when using reg_ce
    allow_non_ascii: bool = False
    allow_special: bool = False


def _masked_cross_entropy(
    shift_logits: Tensor,
    shift_labels: Tensor,
    logit_mask: Tensor,
) -> Tensor:
    """Cross-entropy loss on selected tokens.

    Args:
        shift_logits: logits of shape (batch_size, seq_len - 1, vocab_size)
        shift_labels: labels of shape (batch_size, seq_len - 1)
        logits of token i predicts label i (token i+1)
        logit_mask: mask where to apply loss to of shape (seq_len -1,), shared across the batch

    Returns:
        loss: Tensor of shape (batch_size,)
    """
    batch_size, _, vocab_size = shift_logits.shape  
    sel_logits = shift_logits[:, logit_mask, :].contiguous()  # (batch_size, num_selected_tokens, vocab_size)
    sel_labels = shift_labels[:, logit_mask].contiguous()     # (batch_size, num_selected_tokens)
    flat_loss = torch.nn.functional.cross_entropy(
        sel_logits.view(-1, vocab_size), #flatten since cross-entropy expects class dimension to be 1
        sel_labels.view(-1),
        reduction="none",
    )
    loss = flat_loss.view(batch_size, -1).mean(dim=-1)
    return loss


@torch.no_grad()
def compute_loss(
    # logits: Tensor, #keeping this in case want to revert to logits input and do fwd pass elsewhere
    model: PreTrainedModel,
    tokens: Tensor,
    target_mask: torch.BoolTensor,
    attack_mask: Optional[torch.BoolTensor] = None,
    lm_reg_weight: float = 0.0,
) -> Tuple[Tensor, int]:
    """Computes the cross-entropy loss on target tokens (-log p(y|q,x)) plus language-model regularizer
    lm_reg_weight * cross-entropy loss on attack tokens (-log p(x|q))

    Args:
        model: PreTrainedModel
        tokens: token ids for the full conversation. Tensor of shape (batch_size, seq_len)
        target_mask: bool mask of shape (seq_len,); True on tokens to apply loss to (target tokens shifted by one to the left).
        #TODO: maybe simpler to not shift target_mask earlier
        attack_mask: bool mask of shape (seq_len,); True on attack tokens. Optional if lm_reg_weight == 0.0.
        lm_reg_weight: Multiplier for the language-model regularizer.

    Returns:
        loss: Tensor of shape (batch_size,)
        flops: int, number of flops for the forward pass for the full batch

    """
    # TODO: if we revert to logits inputs, put back description logits: logits outputs for the full conversation. Tensor of shape (batch_size, seq_len, vocab_size)
    logits = model(tokens).logits
    flops = get_flops(model, tokens.numel(), 0, "forward") 
    # logits of token i-1 predicts token i
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    tgt_logit_mask= target_mask[:-1]

    loss = _masked_cross_entropy(shift_logits, shift_labels, tgt_logit_mask)
    if lm_reg_weight > 0.0:
        assert attack_mask is not None, "attack_mask is required when lm_reg_weight > 0.0"
        atk_logit_mask = attack_mask[1:] #shift to the left 
        reg_loss = _masked_cross_entropy(shift_logits, shift_labels, atk_logit_mask)
        loss += lm_reg_weight * reg_loss
    
    return loss, flops

def loss_fn(attack_ids: Tensor):
    """
    Args: 
        attack_ids: the attack token ids to evaluate. Tensor of shape (batch_size, n_optim_tokens)
    """
    # TODO: do KV caching as done in GCG.
    # we need to map attack_ids to batch of tokens to provide as input to compute_loss (or move foward pass here)

    return 0
    
# TODO: move SubmodularSetFnReduction to separate file
class SubmodularSetFnReduction:
    """Given a DR-submodular discrete function F: V^n -> R, where V = {0, 1, ..., k - 1} and k = 2^t,
    provides reduction to a submodular set function F_set: 2^([n] x [t]) -> R and its Lovasz extension subgradient computation.
    """
    def __init__(self, F_batch: Callable[[Tensor], Any], k: int, n: int, device: torch.device):
        # F_batch should be a function that takes a batch of inputs in V^n (Tensor of shape (batch_size, n)) 
        # and evaluates F for each input.
        self.F_batch = F_batch
        self.device = device 
        self.k = k
        self.n = n
        self.F_set_batch = self.set_function_reduction()
        # TODO: normalize F(emptyset) = 0
        
    def __call__(self, rows: Tensor, cols: Tensor) -> Tensor:
        return self.F_set_batch(rows, cols)

    def set_function_reduction(self) -> Callable[[Tensor, Tensor], Any]:
        """Reduction from F to F_set using binary representation of subsets, i.e.,
        F_set(S) = F(x), where X = J_S is the matrix with 1 at indices in S, 0 elsewhere, 
        and x is the vector such that each x_i is the int with binary representation X[i, :]
        Least significant bit is at column index 0 (bit index matches column index).
        For simplicity, will use rows and cols indices as input to F_set instead of a set of tupples
        # TODO: modify this if needed
        """
        # TODO: test this even if not used so far. For now assume vocab_size is a power of 2
        self.t = int(log2(self.k))
        assert self.k == 2 ** self.t, "k must be a power of 2"
        self.powers = (1 << torch.arange(self.t, dtype=torch.long, device=self.device)) # more efficient than 2**torch.arange(t)
        
        def F_set_batch(rows: Tensor, cols: Tensor) -> Tensor:
            """Compute F_set(S^i) for the sets S^i = {(rows[i, j], cols[i, j]) for j in range(rows.shape(1))}
            Args:
                rows: Tensor of shape (batch_size, n)
                cols: Tensor of shape (batch_size, n)
            """
            assert rows.shape == cols.shape, "rows and cols must have the same shape"
            x = torch.zeros((rows.shape[0], self.n), dtype=torch.long, device=self.device)
            x.scatter_add_(1, rows, self.powers[cols]) # x[i, rows[i, j]] += powers[cols[i, j]] for all i, j
            return self.F_batch(x)

        return F_set_batch

    def subgradient_lovasz_extension(self, X: Tensor, tie_breaker: Optional[Tensor] = None):
        # TODO: adjust docstring 
        """Compute a subgradient of the Lovasz extension of self.F_set using Edmonds' greedy algorithm.
        Args:
            X: Tensor of shape (n, t) in [0,1]^n x t 
            tie_breaker: Tensor of shape (n, t) used to break ties when sorting X.flatten(). If not provided, original order is used.
        """
        # TODO: for now assume x is a 2D tensor, not sure if there's a reason to vectorize it 
        if tie_breaker is None:
            sorted_idx = torch.argsort(X.flatten(), descending=True, stable=True)
        else: 
            sorted_idx = torch.argsort(tie_breaker.flatten(), descending=True, stable=True) 
            sorted_idx = sorted_idx[torch.argsort(X.flatten()[sorted_idx], stable=True)]

        rows, cols = torch.unravel_index(sorted_idx, X.shape)
        # map sets S^i = {(rows[1], cols[1]), ..., (rows[i], cols[i])} to x^i in V^n and stack them in x_chain
        # more efficient than calling F_set on S^i's which would compute each x^i separately
        x = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        x_chain = [] # no need to evaluate F(0) since F is normalized #TODO: normalize earlier
        for i in rows.shape[0]:
            x[rows[i]] += self.powers[cols[i]] #TODO: maybe we have a class where we store powers to not have to recompute them
            x_chain.append(x)
        
        # compute F(x^i) for all x^i's
        Fvalues = self.F_batch(x_chain) # TODO: need to implement F that takes batch of attack indices and computes loss for them

        return Fvalues



 


class DummyAttack(Attack):
    def __init__(self, config: DummyConfig):
        super().__init__(config)

    @torch.no_grad()
    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, dataset: PromptDataset) -> AttackResult:
        # TODO: add time tracking
        # --- 1. Prepare Conversations ---
        tokens, attack_masks, target_masks, conversations = self._prepare_dataset(dataset, tokenizer)
        logging.info(f"Prepared {len(conversations)} conversations for attack")

        # get disallowed_ids as done in GCG
        not_allowed_ids = get_disallowed_ids(tokenizer, self.config.allow_non_ascii, self.config.allow_special).to(model.device)
        num_embeddings = model.get_input_embeddings().weight.size(0) 
        # drop disallowed_ids >= num_embeddings; some models like gemma-3 add extra tokens that do not have embeddings
        self.not_allowed_ids = not_allowed_ids[not_allowed_ids < num_embeddings]
        self.vocab_size = num_embeddings

        runs = []
        for idx, conversation in enumerate(dataset):
            runs.append(self._attack_single_conversation(model, tokenizer, conversation, tokens[idx], attack_masks[idx], target_masks[idx]))     
           
        return AttackResult(runs=runs)


    def _attack_single_conversation(self, model, tokenizer, conversation, tokens, attack_mask, target_mask) -> SingleAttackRunResult:
        #TODO: Compute the KV Cache for tokens that appear before the optimized tokens as done in GCG.
        logging.info(f"Starting attack for conversation: {conversation}")
        t_start = time.time()
        # --- 2. Optimize attack ---
        # TODO: Implement optimization loop here.
        # TODO: compute loss for initial optim_str. GCG does that in init_buffer 
        # it doesn't create a AttackStepResult for it but it uses it for initialization of best loss and best optim_ids
        # so to be consistent with it and other attacks I won't do that either
        device = model.device
        tokens = tokens.unsqueeze(0).to(device) 
        attack_mask = attack_mask.to(device)
        target_mask = target_mask.to(device)
        losses = []
        times = []
        flops = []
        optim_strings = [self.config.num_steps] if self.config.num_steps == 0 else []
        for i in (pbar := trange(self.config.num_steps, file=sys.stdout)):
            current_loss, time_for_step, optim_ids, optim_str, flops_for_step = self._single_step(model, tokenizer, conversation, tokens, attack_mask, target_mask)
            losses.append(current_loss)
            times.append(time_for_step)
            # TODO: add flops for prefill and init to initial step flops as done in GCG if we do prefill/init? 
            flops.append(flops_for_step) 
            optim_strings.append(optim_str)
            pbar.set_postfix({"Loss": current_loss, "Current Attack": optim_str[:80]})

        logging.info(f"Optimization loop completed."
        # logging.info(f"Optimization loop completed. Best attack: {optim_strings[-1][:80]} with loss: {losses[-1]}." # for now we're not saving best loss
        f"Optimization time: {time.time() - t_start:.2f}s.")

        # --- 3. Generate Completions --- 
        # get tokens of attack conversations with otimized attack strings and empty assistant content
        prompt_token_list = []
        attack_conversations = []
        for attack in optim_strings:
            parts, attack_conversation = self._prepare_single_conversation(conversation, tokenizer, attack, generation = True)
            prompt_token_list.append(torch.cat(parts[:5]))
            attack_conversations.append(attack_conversation)

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
            initial_batch_size=len(optim_strings), # change to size of the full dataset if we switch to batched optimization
        )
        t_end_gen = time.time()
        gen_time_total = t_end_gen - t_start_gen
        logging.info(f"Generated {len(completions)}x{self.config.generation_config.num_return_sequences} completions. "
         f"Generation time: {gen_time_total:.2f}s.")

        t_end = time.time()

        # --- 4. Assemble Results ---
        steps_results = []
        for i in range(self.config.num_steps):
            step_result = AttackStepResult(
                step=i,
                model_completions=completions[i],
                time_taken=times[i],
                loss=losses[i],
                flops=flops[i],
                model_input=attack_conversations[i],
                model_input_tokens=prompt_token_list[i].tolist(),
            )
            steps_results.append(step_result)

        run_result = SingleAttackRunResult(
            original_prompt=conversation,
            steps=steps_results,
            total_time=t_end - t_start,
        )
        return run_result

    def _single_step(self, model, tokenizer, conversation, tokens, attack_mask, target_mask) -> Tuple[float, float, torch.Tensor, str, int]:
        # TODO: Implement single step of the attack.
      
        t_start_step = time.time()
        optim_str = self.config.optim_str_init
        optim_ids = tokens[:, attack_mask]
        loss, loss_flops = compute_loss(model, tokens, target_mask, attack_mask, self.config.lm_reg_weight)
        current_loss = loss.item()
        # TODO: check if optim_ids is reachable using filter_suffix as done in GCG.
        time_for_step =  time.time() - t_start_step
        flops_for_step = loss_flops + 0
        return current_loss, time_for_step, optim_ids, optim_str, flops_for_step

    # copied from PGDDiscreteAttack. Added assert for single-turn conversation and removed padding.
    # if we're not doing batched optimization, no point preparing full dataset, can call _prepare_single_conversation
    # inside _attack_single_conversation. For now let's keep this in case we switch to batched optimization.
    def _prepare_dataset(self, dataset, tokenizer) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Conversation]]:
        all_tokens = []
        all_attack_masks = []
        all_target_masks = []
        all_conversations = []

        for conversation in dataset:
            assert len(conversation) == 2, "Dummy attack currently assumes single-turn conversation."

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
            attack_mask[offset:offset + attack_prefix_toks.size(0)] = True
            offset += attack_prefix_toks.size(0) + prompt_toks.size(0)
            attack_mask[offset:offset + attack_suffix_toks.size(0)] = True

            target_mask = torch.zeros_like(tokens, dtype=torch.bool)
            target_start_idx = len(tokens) - target_toks.size(0)
            target_mask[target_start_idx:] = True
            # TODO: maybe better to shift when computing loss and not here for clarity?
            # unless we need this shifted version elsewhere?
            target_mask = target_mask.roll(-1, 0) # shift to the left 
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
 
    

    def _prepare_single_conversation(self, conversation, tokenizer, optim_str, generation = False
    ) -> Tuple[tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor], Conversation]:
        # insert optimizable string optim_str in user content according to placement and get tokens of conversation split into six parts
        assistant_content = conversation[1]["content"] if not generation else ""
        if self.config.placement == "suffix":
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + optim_str},
                {"role": "assistant", "content": assistant_content}
            ]
        elif self.config.placement == "prefix":
            attack_conversation = [
                {"role": "user", "content": optim_str + conversation[0]["content"]},
                {"role": "assistant", "content": assistant_content}
            ]
        elif self.config.placement == "prefix_suffix":
            attack_conversation = [
                {"role": "user", "content": optim_str + conversation[0]["content"] + optim_str},
                {"role": "assistant", "content": assistant_content}
            ]
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
        parts = prepare_conversation(tokenizer, conversation, attack_conversation)[0] # assumes single-turn conversation

        return parts, attack_conversation