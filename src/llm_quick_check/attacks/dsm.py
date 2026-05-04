"""Difference of submodular minimization (DSM) attack"""
import copy
import time
import logging
import sys
import matplotlib.pyplot as plt
from tqdm import trange
from typing import List, Tuple, Callable, Any
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from ..dataset import PromptDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult
from ..lm_utils import prepare_conversation, TokenMergeError, generate_ragged_batched, get_flops, get_disallowed_ids, filter_suffix
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
    pgm_L: float | str = 'singletons' 
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
        # --- Prepare Conversations ---
        tokens, attack_masks, target_masks, conversations = self._prepare_dataset(dataset, tokenizer)
        logging.info(f"Prepared {len(conversations)} conversations for attack")

        # --- Build Valid Vocab ---
        self._build_valid_vocab(tokenizer, model)

        runs = []
        for idx, conversation in enumerate(conversations):
            runs.append(self._attack_single_conversation(model, tokenizer, conversation, tokens[idx], attack_masks[idx], target_masks[idx]))

        return AttackResult(runs=runs)

    def _attack_single_conversation(self, model, tokenizer, conversation, tokens, attack_mask, target_mask) -> SingleAttackRunResult:
        #TODO: Compute the KV Cache for tokens that appear before the optimized tokens as done in GCG.
        #TODO: add early stopping if exact match found as done in GCG.
        logging.info(f"Starting attack for conversation: {conversation}")
        t_start = time.time()
        # --- Optimize Attack ---
        # TODO: Implement optimization loop here.
        # TODO: compute loss for initial optim_str. GCG does that in init_buffer
        # it doesn't create a AttackStepResult for it but it uses it for initialization of best loss and best optim_ids
        # so to be consistent with it and other attacks I won't do that either
        device = model.device
        tokens = tokens.to(device)
        attack_mask = attack_mask.to(device)
        target_mask = target_mask.to(device)
        n_optim_tokens = int(attack_mask.sum().item())
        # Initialize with the token ids of optim_str_init
        # TODO: experiment with different initial solutions (see notes.md)
        optim_ids_init = tokens[attack_mask].detach().clone().unsqueeze(0) # (1, n_optim_tokens)
        optim_ids_reduced = self.valid_token_id_to_reduced_idx[optim_ids_init]
        invalid_optim_ids = optim_ids_init[optim_ids_reduced == -1]
        if invalid_optim_ids.numel() > 0:
            raise ValueError(
                f"Initial attack ids contains {invalid_optim_ids.numel()} not allowed token id(s) "
                f"e.g. {invalid_optim_ids[:5].tolist()}."
            )
    

        # define loss_fn over V^n where V = {0, 1, ..., valid_vocab_size - 1} and n = n_optim_tokens
        loss_fn = lambda attack_ids: compute_loss(
            model, self.valid_token_ids[attack_ids], tokens, target_mask, attack_mask, self.config.lm_reg_weight
        )
        zero_attack_ids = torch.zeros_like(optim_ids_reduced)
        F_0, F_0_flops = loss_fn(zero_attack_ids)
        logging.info(f"Loss at zero F(0): {F_0.item():.4f}")
        # normalize F(0) = 0
        def F_batch(attack_ids):
            loss, flops = loss_fn(attack_ids)
            return loss - F_0, flops

        # define filter function
        filter_fn = None
        filter_zero = False
        if self.config.filter_ids: 
            if self.config.placement == "suffix":
                filter_fn = lambda attack_ids: filter_suffix(tokenizer, conversation, [[None, self.valid_token_ids[attack_ids].cpu()]])
                try:  # check if zero_attack_ids is reachable
                    filter_fn(zero_attack_ids)
                except RuntimeError:
                    filter_zero = True
                    logging.warning("Zero attack ids is not reachable from any input string. Will not round to zero during optimization.") 
            else:
                # TODO: adapt filter function for other placements
                raise ValueError(f"Filtering for {self.config.placement} placement not supported yet.")

        F_set_batch = EneSubmodularSetFnReduction(F_batch, self.valid_vocab_size, n_optim_tokens, device, filter_fn, filter_zero)
       
        # run PGM with initial optim_ids as initial solution (assume F is approximately submodular)       
        best_sol_idx, discrete_obj_values, continuous_obj_values, duality_gaps, discrete_sols, times, flops = \
            pgm_lovasz(F_set_batch, optim_ids_reduced, self.config.num_steps, self.config.pgm_L, gap_tol=None)

        plot_pgm_curves(discrete_obj_values, continuous_obj_values, duality_gaps)

        flops[0] += F_0_flops

        # map back to original token ids and decode to strings
        optim_ids = self.valid_token_ids[discrete_sols]
        optim_strings = tokenizer.batch_decode(optim_ids.cpu())  # decode handles batching in v5.3+, keeping batch_decode to support older versions
        losses = [val + F_0.item() for val in discrete_obj_values]

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
            f"Optimization loop completed. Best attack (step {best_sol_idx}): {optim_strings[best_sol_idx][:80]!s}. "
            f"Optimization time: {time.time() - t_start:.2f}s."
        )
        # logging.info(f"Optimization loop completed. Best attack: {optim_strings[-1][:80]} with loss: {losses[-1]}." # for now we're not saving best loss

        # --- Generate Completions ---
        # get tokens of attack conversations with otimized attack strings and empty assistant content
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
                    optim_strings.pop(idx)
                    losses.pop(idx)
                    times.pop(idx)
                    flops.pop(idx)
                    continue

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

        # --- Assemble Results ---
        #TODO: If we want to also store continuou loss and duality gap, we can create subclasses of AttackStepResult for that.
        steps_results = []
        for i in range(len(optim_strings)):
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
        self.valid_vocab_size = int(self.valid_token_ids.numel())

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

def plot_pgm_curves(discrete_obj_values, continuous_obj_values, duality_gaps):
    steps_axis = range(len(discrete_obj_values))
    fig, (ax_obj, ax_gap) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_obj.plot(steps_axis, discrete_obj_values, label=r"Discrete $F(x^t)$", marker="o", ms=3)
    ax_obj.plot(steps_axis, continuous_obj_values, label=r"Lovasz $f_L(X^t)$", marker="s", ms=3)
    ax_obj.set_ylabel("Objective")
    ax_obj.legend(loc="best")
    ax_obj.grid(True, alpha=0.3)
    ax_gap.plot(steps_axis, duality_gaps, color="C2", label="Duality gap", marker="^", ms=3)
    ax_gap.set_xlabel("PGM iteration")
    ax_gap.set_ylabel("Duality gap")
    ax_gap.grid(True, alpha=0.3)
    fig.suptitle("DSM PGM trace")
    fig.tight_layout()
    fig.savefig("dsm_pgm_curves.png", dpi=150)
    plt.close(fig)