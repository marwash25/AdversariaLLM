"""Dummy attack for testing/learning implementation of new attacks"""
import copy
import time
import logging
from typing import Dict, List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from ..dataset import PromptDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult
from ..lm_utils import prepare_conversation, TokenMergeError, generate_ragged_batched


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

class DummyAttack(Attack):
    def __init__(self, config: DummyConfig):
        super().__init__(config)

    @torch.no_grad()
    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, dataset: PromptDataset) -> AttackResult:
        # TODO: add time tracking
        # --- 1. Prepare Conversations ---
        tokens, attack_masks, target_masks, conversations = self._prepare_dataset(dataset, tokenizer)
        logging.info(f"Prepared {len(conversations)} conversations for attack")

        # --- 2. Optimize attack ---
        runs = []
        for conversation in dataset:
            runs.append(self._attack_single_conversation(model, tokenizer, conversation))     

        # --- 3. Generate Completions --- 
        t_start_gen = time.time()
        completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=prompt_token_tensors_list,  # Generate from the prompt tokens
            # embedding_list=embedding_list, # Or generate from the prompt embeddings 
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=len(attack_strings), # change to size of the full dataset if we switch to batched optimization
        )
        t_end_gen = time.time()
        gen_time_total = t_end_gen - t_start_gen

        # --- 4. Assemble Results ---
           
        return AttackResult(runs=runs)



    # Things I would need for optimization attack:

    # get_disallowed_ids as done in for example in PGD_discrete, GCG, RandomSearch.
    # from PGD discrete:
    # disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)
    # from GCG:
    # self.not_allowed_ids = get_disallowed_ids(tokenizer, self.config.allow_non_ascii, self.config.allow_special).to(model.device)

    def _attack_single_conversation(self, model, tokenizer, conversation) -> SingleAttackRunResult:
        #TODO: Compute the KV Cache for tokens that appear before the optimized tokens as done in GCG.
        run = []
        return run

    # copied from PGDDiscreteAttack. Added assert for single-turn conversation and removed padding.
    # if we're not doing batched optimization, no point preparing full dataset, can call _prepare_single_conversation
    # inside _attack_single_conversation. For now let's keep this in case we switch to batched optimization.
    def _prepare_dataset(self, dataset, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        #TODO: do we want to keep padding here?
        all_tokens = []
        all_attack_masks = []
        all_target_masks = []
        all_conversations = []

        for conversation in dataset:
            assert len(conversation) == 2, "Dummy attack currently assumes single-turn conversation."

            all_conversations.append(conversation)
            try:
                tokens, attack_mask, target_mask = self._prepare_single_conversation(
                    conversation, tokenizer, self.config.optim_str_init
                )
            except TokenMergeError:
                logging.warning("TokenMergeError encountered, retrying with added space.")
                tokens, attack_mask, target_mask = self._prepare_single_conversation(
                    conversation, tokenizer, " " + self.config.optim_str_init
                )

            all_tokens.append(tokens)
            all_attack_masks.append(attack_mask)
            all_target_masks.append(target_mask)

        # remove padding for now since we're not doing batched optimization. 
        # TODO: add padding back but not here if we switch to batched optimization.
        # all_tokens = pad_sequence(all_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        # all_target_masks = pad_sequence(all_target_masks, batch_first=True)
        # all_attack_masks = pad_sequence(all_attack_masks, batch_first=True)

        return all_tokens, all_attack_masks, all_target_masks, all_conversations
 
    

    # copied from PGDDiscreteAttack
    def _prepare_single_conversation(self, conversation, tokenizer, optim_str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        # insert optimizable string optim_str in user content according to placement, tokenize conversation, then build attack_mask (tokens to optimize) 
        # and target_mask (tokens to apply loss to)
        if self.config.placement == "suffix":
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + optim_str},
                {"role": "assistant", "content": conversation[1]["content"]}
            ]
        elif self.config.placement == "prefix":
            attack_conversation = [
                {"role": "user", "content": optim_str + conversation[0]["content"]},
                {"role": "assistant", "content": conversation[1]["content"]}
            ]
        elif self.config.placement == "prefix_suffix":
            attack_conversation = [
                {"role": "user", "content": optim_str + conversation[0]["content"] + optim_str},
                {"role": "assistant", "content": conversation[1]["content"]}
            ]
        elif self.config.placement == "prompt": # the whole prompt is optimized
            attack_conversation = copy.deepcopy(conversation)
            conversation = copy.deepcopy(conversation)
            conversation[0]["content"] = ""
        else:
            raise ValueError(f"Invalid placement: {self.config.placement}")
        parts = prepare_conversation(tokenizer, conversation, attack_conversation)[0] # assumes single-turn conversation
        pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks = parts

        tokens = torch.cat(parts)

        attack_mask = torch.zeros_like(tokens, dtype=torch.bool)
        offset = pre_toks.size(0)
        attack_mask[offset:offset + attack_prefix_toks.size(0)] = True
        offset += attack_prefix_toks.size(0) + prompt_toks.size(0)
        attack_mask[offset:offset + attack_suffix_toks.size(0)] = True

        target_mask = torch.zeros_like(tokens, dtype=torch.bool)
        target_start_idx = len(tokens) - target_toks.size(0)
        target_mask[target_start_idx:] = True
        target_mask = target_mask.roll(-1, 0)
        target_mask[-1] = False

        return tokens, attack_mask.long(), target_mask.long()