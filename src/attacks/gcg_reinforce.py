"""WIP GCG REINFORCE attack implementation.

This implements the REINFORCE version of the GCG attack, which uses a judge model
to provide rewards for generated completions and optimizes using REINFORCE gradients.

@article{geisler2025reinforce,
  title={Reinforce adversarial attacks on large language models: An adaptive, distributional, and semantic objective},
  author={Geisler, Simon and Wollschl{\"a}ger, Tom and Abdalla, MHI and Cohen-Addad, Vincent and Gasteiger, Johannes and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2502.17254},
  year={2025}
}
"""

import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from functools import partial

from judgezoo import Judge
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import trange
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..dataset import PromptDataset
from ..lm_utils import (
    TokenMergeError,
    filter_suffix,
    generate_ragged_batched,
    get_disallowed_ids,
    get_flops,
    get_losses_batched,
    prepare_conversation,
    with_max_batchsize,
)
from ..types import Conversation
from .attack import (
    Attack,
    AttackResult,
    AttackStepResult,
    GenerationConfig,
    SingleAttackRunResult,
)


@dataclass
class GCGReinforceConfig:
    name: str = "gcg"
    type: str = "discrete"
    version: str = ""
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    num_steps: int = 250
    seed: int = 0
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_constrained_gradient: bool = False
    mellowmax_alpha: float = 1.0
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    allow_special: bool = False
    filter_ids: bool = True
    verbosity: str = "WARNING"
    token_selection: str = "default"
    grow_target: bool = False
    grad_smoothing: int = 1  # 1 = no smoothing, 2 = smooth over 2 tokens, etc.
    grad_momentum: float = 0.0  # momentum over steps
    judge_model_id: str = "strong_reject"
    optim_num_return_sequences: int = 10
    optim_max_new_tokens: int = 256
    optim_temperature: float = 0.7
    optim_top_p: float = 1.0
    optim_top_k: int = 0
    reward_baseline: float = 0.5  # all generations below this are penalized no matter what


class GCGReinforceAttack(Attack):
    def __init__(self, config: GCGReinforceConfig):
        super().__init__(config)
        self.tokenizer = None  # Will be set in run()
        self.model = None
        self.logger = logging.getLogger("gcg_reinforce")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: PromptDataset,
    ) -> AttackResult:
        self.judge = Judge.from_name(self.config.judge_model_id)
        self.model = model
        self.tokenizer = tokenizer  # Store tokenizer as instance variable
        self.not_allowed_ids = get_disallowed_ids(
            tokenizer, self.config.allow_non_ascii, self.config.allow_special
        ).to(model.device)
        # need to have this filter here for models like gemma-3 which add extra tokens that do not have embeddings
        # we cannot filter the ids inside the get_disallowed_ids function because we need
        # the embedding layer weights to see the correct sizes
        self.not_allowed_ids = self.not_allowed_ids[
            self.not_allowed_ids < model.get_input_embeddings().weight.size(0)
        ]
        runs = []
        for conversation in dataset:
            runs.append(
                self._attack_single_conversation(model, tokenizer, conversation)
            )
        return AttackResult(runs=runs)

    def _attack_single_conversation(
        self, model, tokenizer, conversation
    ) -> SingleAttackRunResult:
        t0 = time.time()
        try:
            attack_conversation: Conversation = [
                {
                    "role": "user",
                    "content": conversation[0]["content"] + self.config.optim_str_init,
                },
                {"role": "assistant", "content": conversation[1]["content"]},
            ]
            (
                pre_ids,
                attack_prefix_ids,
                prompt_ids,
                attack_suffix_ids,
                post_ids,
                target_ids,
            ) = prepare_conversation(tokenizer, conversation, attack_conversation)[0]
        except TokenMergeError:
            attack_conversation: Conversation = [
                {
                    "role": "user",
                    "content": conversation[0]["content"]
                    + " "
                    + self.config.optim_str_init,
                },
                {"role": "assistant", "content": conversation[1]["content"]},
            ]
            (
                pre_ids,
                attack_prefix_ids,
                prompt_ids,
                attack_suffix_ids,
                post_ids,
                target_ids,
            ) = prepare_conversation(tokenizer, conversation, attack_conversation)[0]

        pre_ids = pre_ids.unsqueeze(0).to(model.device)
        # attack_prefix_ids = attack_prefix_ids.unsqueeze(0).to(model.device)
        prompt_ids = prompt_ids.unsqueeze(0).to(model.device)
        pre_prompt_ids = torch.cat([pre_ids, prompt_ids], dim=1)
        attack_ids = attack_suffix_ids.unsqueeze(0).to(model.device)
        post_ids = post_ids.unsqueeze(0).to(model.device)
        target_ids = target_ids.unsqueeze(0).to(model.device)

        # Embed everything that doesn't get optimized
        embedding_layer = model.get_input_embeddings()
        pre_prompt_embeds, post_embeds, target_embeds = [
            embedding_layer(ids) for ids in (pre_prompt_ids, post_ids, target_ids)
        ]

        self.target_ids = target_ids
        self.pre_prompt_embeds = pre_prompt_embeds
        self.post_embeds = post_embeds
        self.target_embeds = target_embeds

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = attack_ids

        rewards = []
        times = []
        flops = []
        optim_strings = []

        for i in (pbar := trange(self.config.num_steps, file=sys.stdout)):
            current_reward, time_for_step, optim_ids, optim_str, flops_for_step = (
                self._single_step(conversation, buffer, optim_ids)
            )
            rewards.append(current_reward)
            times.append(time_for_step)
            optim_strings.append(optim_str)
            if i == 0:
                flops.append(flops_for_step)
            else:
                flops.append(flops_for_step)
            pbar.set_postfix(
                {
                    "Reward": current_reward,
                    "Best Attack": optim_str[:80],
                }
            )
        token_list = []
        attack_conversations = []
        for attack in optim_strings:
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + attack},
                {"role": "assistant", "content": ""},
            ]
            tokens = prepare_conversation(tokenizer, conversation, attack_conversation)[0]
            token_list.append(torch.cat(tokens[:5]))
            attack_conversations.append(attack_conversation)

        batch_completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_list,
            initial_batch_size=len(token_list),
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )  # (N_steps, N_return_sequences, T)

        steps = []
        t1 = time.time()
        for i in range(len(optim_strings)):
            step = AttackStepResult(
                step=i,
                model_completions=batch_completions[i],
                time_taken=times[i],
                loss=rewards[i],
                flops=flops[i],
                model_input=attack_conversations[i],
                model_input_tokens=token_list[i].tolist(),
            )
            steps.append(step)

        run = SingleAttackRunResult(
            original_prompt=conversation,
            steps=steps,
            total_time=t1 - t0,
        )
        return run

    def _single_step(self, conversation, buffer, optim_ids):
        t0a = time.time()

        t_rewards_start = time.time()
        rewards, generations, flops_loss = self.compute_candidate_rewards(conversation, optim_ids)
        if buffer.size == 0 or rewards.mean().item() > buffer.get_highest_reward():
            buffer.add(rewards.mean().item(), optim_ids)
        t_rewards = time.time() - t_rewards_start
        print(f"  Compute rewards: {t_rewards:.2f}s")

        t_grad_start = time.time()
        advantages = self.compute_advantages(rewards)  # (num_gens,)
        optim_ids_one_hot = F.one_hot(optim_ids, num_classes=self.tokenizer.vocab_size).to(self.model.dtype).requires_grad_(True)
        loss = self.compute_loss(advantages, optim_ids_one_hot, generations)
        grad = torch.autograd.grad([loss], [optim_ids_one_hot])[0].squeeze(0)
        t_grad = time.time() - t_grad_start
        print(f"  Compute gradients: {t_grad:.2f}s")

        t_sample_start = time.time()
        candidate_ids, candidate_ids_pos = _sample_ids_from_grad(optim_ids.squeeze(0), grad, self.config.search_width, self.config.topk, self.config.n_replace, self.not_allowed_ids)
        t_sample = time.time() - t_sample_start
        print(f"  Sample candidates: {t_sample:.2f}s")

        with torch.no_grad():
            t_eval_start = time.time()
            # Sample candidate token sequences
            if self.config.filter_ids:
                # We're trying to be as strict as possible here, so we filter
                # the entire prompt, not just the attack sequence in an isolated
                # way. This is because the prompt and attack can affect each
                # other's tokenization in some cases.
                idx = filter_suffix(
                    self.tokenizer,
                    conversation,
                    [[None, candidate_ids.cpu()]],
                )
                candidate_ids = candidate_ids[idx] # (B, T)
                candidate_ids_pos = candidate_ids_pos[idx] # (B, T)

            compute_loss_fn = partial(self.compute_candidates_loss, generations, advantages)
            loss = with_max_batchsize(compute_loss_fn, candidate_ids)
            flops_loss = flops_loss.sum().item()
            optim_ids = candidate_ids[loss.argmin()].unsqueeze(0)

            # Update the buffer based on the loss
            flops_for_step = flops_loss

            t_eval = time.time() - t_eval_start
            print(f"  Evaluate candidates: {t_eval:.2f}s")

        optim_str = self.tokenizer.batch_decode(optim_ids)[0]
        return rewards.mean().item(), time.time() - t0a, optim_ids, optim_str, flops_for_step

    def init_buffer(self):
        config = self.config

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)
        return buffer

    @torch.no_grad()
    def compute_candidate_rewards(
        self,
        conversation: Conversation,
        attack_ids: Tensor,
    ) -> Tensor:
        """Computes the GCG reward on all candidate token id sequences.

        Args:
            attack_ids : Tensor, shape = (1, T)
                the attack token ids to evaluate
            conversation : Conversation
                the original dataset conversation to evaluate

        Returns:
            reward : Tensor, shape = (B,)
                the GCG reward on all candidate sequences
        """
        embeds_to_generate_with = torch.cat(
            [
                self.pre_prompt_embeds,
                self.model.get_input_embeddings()(attack_ids),
                self.post_embeds,
            ],
            dim=1,
        )[0] # (T, V)

        completions = generate_ragged_batched(
            model=self.model,
            tokenizer=self.tokenizer,
            embedding_list=[embeds_to_generate_with],
            max_new_tokens=self.config.optim_max_new_tokens,
            temperature=self.config.optim_temperature,
            top_p=self.config.optim_top_p,
            top_k=self.config.optim_top_k,
            num_return_sequences=self.config.optim_num_return_sequences,
            return_tokens=True
        )[0]

        conversation_with_completions: Conversation = [
            [
                {"role": "user", "content": conversation[0]["content"] },
                {"role": "assistant", "content": c},
            ] for c in self.tokenizer.batch_decode(completions)
        ]

        judgements = self.judge(conversation_with_completions)["p_harmful"]
        rewards = torch.tensor(judgements).to(self.model.device)

        gc.collect()
        torch.cuda.empty_cache()
        flops = 0

        return (
            rewards,
            completions,
            torch.tensor(flops).expand_as(rewards),
        )

    def compute_advantages(self, rewards):
        """Computes the reinforce advantages for this set of generations with their rewards.
        We use the leave-one-out estimator from Koop et al. 2019 to compute the advantages.

        Args:
            rewards: (B,)
            generations: (B, T)
        """
        total_sum = rewards.sum() + self.config.reward_baseline
        n = rewards.size(0) + 1
        advantages = (rewards * n - total_sum) / (n - 1)
        return advantages + 1e-8

    def compute_loss(self, advantages: Tensor, optim_ids_one_hot: Tensor, generations: list[Tensor]):
        """Computes the REINFORCE loss for a single set of optim_ids.

        Args:
            advantages: (B,) tensor
            optim_ids_one_hot: (B, V) tensor
            generations: list of (T,) tensors

        Returns:
            loss: (1,) tensor
        """
        losses = []
        optim_embeds = optim_ids_one_hot @ self.model.get_input_embeddings().weight  # (B, V)
        for gen in generations:
            non_gen_embeds = torch.cat([self.pre_prompt_embeds, optim_embeds, self.post_embeds], dim=1) # (1, T, V)
            gen_embeds = self.model.get_input_embeddings()(gen.to(self.model.device)).unsqueeze(0) # (1, T, V)
            full_embeds = torch.cat([non_gen_embeds, gen_embeds], dim=1)  # (1, T, V)
            out = self.model.forward(inputs_embeds=full_embeds).logits[:, -1-len(gen):-1, :] # (1, T, V)
            loss = F.cross_entropy(out.transpose(1, 2), gen.to(self.model.device).unsqueeze(0))
            losses.append(loss)
        losses = torch.stack(losses)
        return (losses * advantages).mean()

    def compute_candidates_loss(self, generations: list[Tensor], advantages: Tensor, candidate_ids: Tensor):
        """Computes the REINFORCE loss for a set of candidate token id sequences.
        Vectorized implementation using get_losses_batched.

        Args:
            generations: list of B (T,) tensors
            advantages: (B,) tensor in [0, 1]
            candidate_ids: (N, T) tensor

        Returns:
            loss: (N,) tensor
        """
        t0_candidates = time.time()
        N = candidate_ids.shape[0]
        B = len(generations)
        print(f"    Computing loss for {N} candidates x {B} generations")

        # Convert candidate_ids to one_hot and get embeddings
        t_embed_start = time.time()
        candidate_ids_one_hot = F.one_hot(candidate_ids, num_classes=self.tokenizer.vocab_size).to(self.model.dtype)
        optim_embeds = candidate_ids_one_hot @ self.model.get_input_embeddings().weight  # (N, T, V)
        t_embed = time.time() - t_embed_start
        print(f"    Embedding conversion: {t_embed:.3f}s")

        # Create input embeddings for each candidate and generation combination
        t_prep_start = time.time()
        embedding_list = []
        targets = []

        for i, gen in enumerate(generations):
            for j in range(N):
                # Construct full input embeddings: pre_prompt + attack + post + generation
                full_embeds = torch.cat([
                    self.pre_prompt_embeds.squeeze(0),  # Remove batch dim
                    optim_embeds[j],           # (T, V)
                    self.post_embeds.squeeze(0),       # Remove batch dim
                    self.model.get_input_embeddings()(gen.to(self.model.device))  # (T, V)
                ], dim=0)  # (T, V)
                target = torch.zeros(full_embeds.shape[0], device=self.model.device, dtype=torch.long) # (T,)
                target[-len(gen)-1:-1] = gen.to(self.model.device) # (T,)
                embedding_list.append(full_embeds) # (T, V)
                targets.append(target) # (T,)
        t_prep = time.time() - t_prep_start
        print(f"    Prepare {len(embedding_list)} embeddings: {t_prep:.3f}s")

        # Get losses using get_losses_batched
        t_loss_start = time.time()
        losses_list = get_losses_batched(
            model=self.model,
            targets=targets, # len(generations) * N * (T,)
            embedding_list=embedding_list, # len(generations) * N * (T, V)
            padding_side="right",
            initial_batch_size=512,
            verbose=True,
        )
        t_loss = time.time() - t_loss_start
        print(f"    Batched loss computation: {t_loss:.3f}s")

        # Reshape losses back to (B, N) and normalize by sequence length
        t_reshape_start = time.time()
        losses = torch.zeros((B, N), device=self.model.device)
        for i, gen in enumerate(generations):
            for j in range(N):
                idx = i * N + j
                losses[i, j] = losses_list[idx][-len(gen):].sum() / len(gen)
        # Apply advantages and take mean over generations
        losses = (losses * advantages.unsqueeze(1)).mean(dim=0)  # (N,)
        t_reshape = time.time() - t_reshape_start
        print(f"    Reshape and normalize: {t_reshape:.3f}s")

        t_total_candidates = time.time() - t0_candidates
        print(f"    Total candidates loss: {t_total_candidates:.3f}s")
        return losses


def _sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = None,
):
    """Returns `search_width` combinations of token ids based on the token gradient.
    Original GCG does this.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace: int
            the number of token positions to update per sequence
        not_allowed_ids: Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    # Initial gradient computation
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")
    # (n_optim_ids, topk)
    topk_ids = grad.topk(topk, dim=1, largest=False, sorted=False).indices

    sampled_ids_pos = torch.randint(
        0, n_optim_tokens, (search_width, n_replace), device=grad.device
    )  # (search_width, n_replace)
    sampled_topk_idx = torch.randint(
        0, topk, (search_width, n_replace, 1), device=grad.device
    )

    sampled_ids_val = (
        topk_ids[sampled_ids_pos].gather(2, sampled_topk_idx).squeeze(2)
    )  # (search_width, n_replace)

    new_ids = original_ids.scatter_(
        1, sampled_ids_pos, sampled_ids_val
    )  # (search_width, n_optim_ids)

    return new_ids, sampled_ids_pos


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[-1][1]

    def get_lowest_reward(self) -> float:
        return self.buffer[0][0]

    def get_highest_reward(self) -> float:
        return self.buffer[-1][0]
