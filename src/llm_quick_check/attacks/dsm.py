"""Difference of submodular minimization (DSM) attack"""
import copy
import time
import logging
import sys
from tqdm import trange
from typing import List, Optional, Tuple, Callable, Any
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
from .submodular_utils import EneSubmodularSetFnReduction, pgm_lovasz


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
        shift_labels: labels of shape (batch_size, seq_len - 1); logits of token
            i predicts label i (token i+1)
        logit_mask: mask where to apply loss to of shape (seq_len -1,),
            shared across the batch

    Returns:
        loss: Tensor of shape (batch_size,)
    """
    batch_size, _, vocab_size = shift_logits.shape
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
) -> Tuple[Tensor, int]:
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
        flops: int, number of flops for the forward pass for the full batch

    """
    # TODO: if we revert to logits inputs, put back description logits: logits outputs for the full conversation. Tensor of shape (batch_size, seq_len, vocab_size)

    input_ids = original_tokens.unsqueeze(0).repeat(attack_ids.shape[0], 1)  # (batch_size, seq_len)
    input_ids[:, attack_mask] = attack_ids
    # TODO: add KV caching as done in GCG.
    logits = model(input_ids).logits
    flops = get_flops(model, input_ids.numel(), 0, "forward")

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

    return loss, flops


class DSMAttack(Attack):
    def __init__(self, config: DSMConfig):
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
        logging.info(f"Number of embeddings: {num_embeddings}, Tokenizer vocab size: {len(tokenizer)}")  # to check if they match

        runs = []
        for idx, conversation in enumerate(conversations):
            runs.append(self._attack_single_conversation(model, tokenizer, conversation, tokens[idx], attack_masks[idx], target_masks[idx]))

        return AttackResult(runs=runs)

    def _attack_single_conversation(self, model, tokenizer, conversation, tokens, attack_mask, target_mask) -> SingleAttackRunResult:
        #TODO: Compute the KV Cache for tokens that appear before the optimized tokens as done in GCG.
        #TODO: add early stopping if exact match found as done in GCG.
        logging.info(f"Starting attack for conversation: {conversation}")
        t_start = time.time()
        # --- 2. Optimize attack ---
        # TODO: Implement optimization loop here.
        # TODO: compute loss for initial optim_str. GCG does that in init_buffer
        # it doesn't create a AttackStepResult for it but it uses it for initialization of best loss and best optim_ids
        # so to be consistent with it and other attacks I won't do that either
        device = model.device
        tokens = tokens.to(device)
        attack_mask = attack_mask.to(device)
        target_mask = target_mask.to(device)
        n_optim_tokens = int(attack_mask.sum().item())

        optim_strings: List[str] = [self.config.optim_str_init] if self.config.num_steps == 0 else []
        # Initialize with the token ids of optim_str_init
        optim_ids = tokens[attack_mask].detach().clone().unsqueeze(0)

        loss_fn = lambda attack_ids: compute_loss(model, attack_ids, tokens, target_mask, attack_mask, self.config.lm_reg_weight)
        F_0, F_0_flops = loss_fn(torch.zeros_like(optim_ids))
        # normalize F(0) = 0
        def F_batch(attack_ids):
            loss, flops = loss_fn(attack_ids)
            return loss - F_0, flops
        F_set_batch = EneSubmodularSetFnReduction(F_batch, self.vocab_size, n_optim_tokens, device)
       
        # run PGM with initial optim_ids as initial solution (assume F is approximately submodular)       
        discrete_obj_values, continuous_obj_values, duality_gaps, discrete_sols, times, flops = \
            pgm_lovasz(F_set_batch, optim_ids, self.config.num_steps, 'singletons', gap_tol=None)

        # TODO: check if optim_ids is reachable using filter_suffix as done in GCG.
        # for i in (pbar := trange(self.config.num_steps, file=sys.stdout)):
        #     current_loss, time_for_step, optim_ids, optim_str, flops_for_step = self._single_step(optim_ids, F_set_batch)
        #     losses.append(current_loss)
        #     times.append(time_for_step)
        #     # TODO: add flops for prefill and init to initial step flops as done in GCG if we do prefill/init?
        #     flops.append(flops_for_step)
        #     optim_strings.append(optim_str)
        #     pbar.set_postfix({"Loss": current_loss, "Current Attack": optim_str[:80]})

        logging.info(
            "Optimization loop completed. "
            f"Optimization time: {time.time() - t_start:.2f}s."
        )
        # logging.info(f"Optimization loop completed. Best attack: {optim_strings[-1][:80]} with loss: {losses[-1]}." # for now we're not saving best loss

        # --- 3. Generate Completions ---
        # get tokens of attack conversations with otimized attack strings and empty assistant content
        prompt_token_list = []
        attack_conversations = []
        for attack in optim_strings:
            parts, attack_conversation = self._prepare_single_conversation(conversation, tokenizer, attack, generation=True)
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
            initial_batch_size=len(optim_strings),  # change to size of the full dataset if we switch to batched optimization
        )
        t_end_gen = time.time()
        gen_time_total = t_end_gen - t_start_gen
        logging.info(
            f"Generated {len(completions)}x{self.config.generation_config.num_return_sequences} completions. "
            f"Generation time: {gen_time_total:.2f}s."
        )

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

    # def _single_step(self, optim_ids: Tensor, F_set_batch: Callable[[List[Tensor], List[Tensor]], Tuple[Tensor, int]]) -> Tuple[float, float, torch.Tensor, str, int]:
    #     """Single step of the attack.
    #     Args:
    #         optim_ids: Current attack token ids. Tensor of shape
    #             (n_optim_tokens,)
    #         F_set_batch: Submodular set function that computes the loss for a batch of attack sets

    #     """

    #     t_start_step = time.time()
    #     optim_str = self.config.optim_str_init
    #     loss, loss_flops = F_set_batch.F_batch(optim_ids)
    #     current_loss = loss.item()
    #     time_for_step = time.time() - t_start_step
    #     flops_for_step = loss_flops + 0
    #     return current_loss, time_for_step, optim_ids, optim_str, flops_for_step

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
            attack_conversation = [
                {"role": "user", "content": optim_str + conversation[0]["content"] + optim_str},
                {"role": "assistant", "content": assistant_content},
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
        parts = prepare_conversation(tokenizer, conversation, attack_conversation)[0]  # assumes single-turn conversation

        return parts, attack_conversation
