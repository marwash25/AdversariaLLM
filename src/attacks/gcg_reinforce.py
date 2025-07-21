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
    name: str = "gcg_reinforce"
    type: str = "discrete"
    version: str = "0.0.1"
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    num_steps: int = 500
    seed: int = 0
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    mellowmax_alpha: float = 1.0
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    allow_special: bool = False
    filter_ids: bool = True
    verbosity: str = "WARNING"
    token_selection: str = "default"
    judge_model_id: str = "strong_reject"
    optim_num_return_sequences: int = 10
    optim_max_new_tokens: int = 256
    optim_temperature: float = 0.7
    optim_top_p: float = 1.0
    optim_top_k: int = 0
    reward_baseline: float = 0.1  # all generations below this are penalized no matter what


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
        optim_ids = attack_ids

        rewards = []
        times = []
        flops = []
        optim_strings = []
        losses = []
        for i in (pbar := trange(self.config.num_steps, file=sys.stdout)):
            current_reward, loss, time_for_step, optim_ids, flops_for_step = (
                self._single_step(conversation, optim_ids)
            )
            rewards.append(current_reward.mean().item())
            losses.append(loss)
            times.append(time_for_step)
            optim_strings.append(self.tokenizer.batch_decode(optim_ids)[0])
            if i == 0:
                flops.append(flops_for_step)
            else:
                flops.append(flops_for_step)
            pbar.set_postfix(
                {
                    "Mean Reward": current_reward.mean().item(),
                    "Max Reward": current_reward.max().item(),
                    "Loss": loss.item(),
                    "Best Attack": optim_strings[-1][:80],
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
                loss=losses[i],
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

    def _single_step(self, conversation, optim_ids):
        """Each GCG step consists of three basic steps:

        1) Compute gradients w.r.t a differentiable loss function
        2) Sample & Filter candidate token sequences (possibly with help of the gradients)
        3) Compute the loss for each candidate token sequence and select the best one

        In our case, the loss function is the REINFORCE loss.

        Parameters
        ----------
        conversation : Conversation
            The original conversation to attack
        optim_ids : Tensor, shape = (1, T)
            The current attack token ids
        """
        t0a = time.time()

        t_rewards_start = time.time()
        rewards, generations, flops_loss = self.compute_candidate_rewards(conversation, optim_ids)
        t_rewards = time.time() - t_rewards_start
        print(f"  Compute rewards: {t_rewards:.2f}s")

        t_grad_start = time.time()
        advantages = self.compute_advantages(rewards)  # (num_gens,)
        optim_ids_one_hot = F.one_hot(optim_ids, num_classes=self.tokenizer.vocab_size).to(self.model.dtype).requires_grad_(True)
        loss_for_grad = self.compute_reinforce_loss(generations, advantages, optim_ids_one_hot)[0]
        print(loss_for_grad.shape, optim_ids_one_hot.shape)
        grad = torch.autograd.grad([loss_for_grad], [optim_ids_one_hot])[0].squeeze(0)
        t_grad = time.time() - t_grad_start
        print(f"  Compute gradients: {t_grad:.2f}s")

        t_sample_start = time.time()
        if grad.isinf().all() or grad.isnan().all():
            candidate_ids, candidate_ids_pos = _random_overall(
                ids=optim_ids.squeeze(0),
                vocab_size=grad.size(-1),
                search_width=self.config.search_width,
                not_allowed_ids=self.not_allowed_ids,
            )
        else:
            candidate_ids, candidate_ids_pos = _sample_ids_from_grad(
                ids=optim_ids.squeeze(0),
                grad=grad,
                search_width=self.config.search_width,
                topk=self.config.topk,
                n_replace=self.config.n_replace,
                not_allowed_ids=self.not_allowed_ids,
            )
        t_sample = time.time() - t_sample_start
        print(f"  Sample candidates: {t_sample:.2f}s")

        with torch.no_grad():
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
            loss_for_candidates = with_max_batchsize(compute_loss_fn, candidate_ids)
            optim_ids = candidate_ids[loss_for_candidates.argmin()].unsqueeze(0)
            current_loss = loss_for_candidates[loss_for_candidates.argmin()]
            flops_for_step = flops_loss.sum().item()
            # Update the buffer based on the loss

        return rewards, current_loss, time.time() - t0a, optim_ids, flops_for_step

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
        n = rewards.size(0)
        if self.config.reward_baseline > 0:
            n += 1
        advantages = (rewards * n - total_sum) / (n - 1)
        print(advantages)
        return advantages + 1e-8

    def compute_affirmative_loss(self, candidate_ids: Tensor) -> Tensor:
        """Computes the affirmative loss for candidate token id sequences.

        Args:
            candidate_ids: (N, T) tensor

        Returns:
            loss: (N,) tensor
        """
        N = candidate_ids.shape[0]

        # Convert candidate_ids to one_hot and get embeddings
        candidate_ids_one_hot = F.one_hot(candidate_ids, num_classes=self.tokenizer.vocab_size).to(self.model.dtype)
        optim_embeds = candidate_ids_one_hot @ self.model.get_input_embeddings().weight  # (N, T, V)

        # Prepare input embeddings
        embedding_list = []
        targets = []
        pre: Tensor = self.pre_prompt_embeds           # (L_pre , V)
        post: Tensor = self.post_embeds                 # (L_post, V)
        pre = pre.expand(optim_embeds.size(0), -1, -1)   # (N, L_pre , V)
        post = post.expand(optim_embeds.size(0), -1, -1)    # (N, L_post, V)

        tgt_emb = self.target_embeds.expand(optim_embeds.size(0), -1, -1) # (N, L_gen , V)
        L_gen = self.target_embeds.size(1)
        full_embeds = torch.cat([pre, optim_embeds, post, tgt_emb], dim=1) # (N, L_pre + L_opt + L_post + L_gen, V)
        seq_len = full_embeds.size(1)
        tgt = full_embeds.new_zeros((optim_embeds.size(0), seq_len), dtype=torch.long)
        tgt[:, -L_gen - 1:-1] = self.target_ids[0]                # broadcast gen into every row

        embedding_list.extend([e for e in full_embeds])            # list of (N, …, V)
        targets.extend([t for t in tgt])                           # list of (N, …)

        # Get losses using get_losses_batched
        losses_list = get_losses_batched(
            model=self.model,
            targets=targets, # len(generations) * N * (T,)
            embedding_list=embedding_list, # len(generations) * N * (T, V)
            padding_side="right",
            initial_batch_size=512,
        )
        affirmatives_losses = torch.stack([losses_list[idx][-L_gen:].mean() for idx in range(N)])

        return affirmatives_losses

    def compute_entropy_loss(self, candidate_ids: Tensor) -> Tensor:
        """Computes the entropy maximization loss using kl_allowed_fwd for candidate token id sequences.

        This implements the entropy maximization loss that encourages the model to produce diverse
        tokens while avoiding disallowed tokens, similar to the kl_allowed_fwd implementation in gcg.py.

        Args:
            candidate_ids: (N, T) tensor

        Returns:
            loss: (N,) tensor
        """
        # Get embeddings directly from candidate_ids
        optim_embeds = self.model.get_input_embeddings()(candidate_ids)  # (N, T, D)

        N = optim_embeds.size(0)  # (N, T, V)
        # Prepare input embeddings
        pre_prompt: Tensor = self.pre_prompt_embeds  # (L_pre, D)
        post: Tensor = self.post_embeds  # (L_post, D)
        pre_prompt = pre_prompt.expand(N, -1, -1)  # (N, L_pre, D)
        post = post.expand(N, -1, -1)  # (N, L_post, D)

        # We only need the first token's logits for entropy computation, no generation required
        full_embeds = torch.cat([pre_prompt, optim_embeds, post], dim=1)  # (N, L_pre + L_opt + L_post, D)
        outputs = self.model(inputs_embeds=full_embeds)
        first_token_logits = outputs.logits[:, -1, :].clone()  # (N, T, V) -> (N, V)

        # Get the log probs for the last position
        # This corresponds to the first generated token
        log_probs = F.log_softmax(first_token_logits.float(), dim=-1)  # (N, V)

        # Create target distribution: uniform over allowed tokens, zero for disallowed
        V = first_token_logits.size(-1)
        N_valid = V - len(self.not_allowed_ids)
        tgt_dist = torch.full((V,), device=log_probs.device, fill_value=1 / N_valid)
        tgt_dist[self.not_allowed_ids] = 0

        model_probs = log_probs.exp()  # (N, V)
        log_tgt = torch.log(tgt_dist + 1e-30)  # (V,) - tiny ε avoids log(0) → -inf

        loss = F.kl_div(
            log_tgt.unsqueeze(0).expand(N, -1),  # (N, V)
            model_probs,  # (N, V)
            reduction="none"
        ).sum(dim=-1)  # (N, V) -> (N,)

        return loss

    def compute_reinforce_loss(self, generations: list[Tensor], advantages: Tensor, candidate_ids: Tensor) -> Tensor:
        """Computes the REINFORCE loss for candidate token id sequences.

        Args:
            generations: list of B (T,) tensors
            advantages: (B,) tensor in [0, 1]
            candidate_ids: (N, T) tensor or (N, T, V) tensor

        Returns:
            loss: (N,) tensor
        """
        N = candidate_ids.size(0)
        B = len(generations)

        # Convert candidate_ids to one_hot and get embeddings
        if candidate_ids.dim() == 2:
            # Can't compute gradients in this scenario
            candidate_ids_one_hot = F.one_hot(candidate_ids, num_classes=self.tokenizer.vocab_size).to(self.model.dtype)
        else:
            candidate_ids_one_hot = candidate_ids
        optim_embeds = candidate_ids_one_hot @ self.model.get_input_embeddings().weight  # (N, T, V)

        # Create input embeddings for each candidate and generation combination
        embedding_list = []
        targets = []
        pre: Tensor = self.pre_prompt_embeds           # (L_pre , V)
        post: Tensor = self.post_embeds                 # (L_post, V)
        pre = pre.expand(optim_embeds.size(0), -1, -1)   # (N, L_pre , V)
        post = post.expand(optim_embeds.size(0), -1, -1)    # (N, L_post, V)

        for gen in generations:                     # keep outer loop (ragged lengths)
            gen = gen.to(self.model.device)
            gen_emb: Tensor = self.model.get_input_embeddings()(gen)   # (L_gen , V)
            gen_emb = gen_emb.unsqueeze(0).expand(optim_embeds.size(0), *gen_emb.shape) # (N, L_gen , V)
            full_embeds = torch.cat([pre, optim_embeds, post, gen_emb], dim=1) # (N, L_pre + L_opt + L_post + L_gen, V)

            seq_len = full_embeds.size(1)
            tgt = full_embeds.new_zeros((optim_embeds.size(0), seq_len), dtype=torch.long)
            tgt[:, -len(gen) - 1:-1] = gen                # broadcast gen into every row

            embedding_list.extend([e for e in full_embeds])            # list of (N, …, V)
            targets.extend([t for t in tgt])                           # list of (N, …)

        # Get losses using get_losses_batched
        losses_list = get_losses_batched(
            model=self.model,
            targets=targets, # len(generations) * N * (T,)
            embedding_list=embedding_list, # len(generations) * N * (T, V)
            padding_side="right",
            initial_batch_size=512,
        )

        # Reshape losses back to (B, N) and normalize by sequence length
        reinforce_losses = torch.zeros((B, N), device=self.model.device)
        for i, gen in enumerate(generations):
            gen_len = len(gen)
            start_idx = i * N
            end_idx = start_idx + N
            batch_losses = torch.stack([losses_list[idx][-gen_len:].sum() / gen_len for idx in range(start_idx, end_idx)])
            reinforce_losses[i, :] = batch_losses
        # Apply advantages and take mean over generations
        reinforce_losses = (reinforce_losses * advantages.unsqueeze(1)).mean(dim=0)  # (N,)

        return reinforce_losses

    @torch.no_grad()
    def compute_candidates_loss(self, generations: list[Tensor], advantages: Tensor, candidate_ids: Tensor):
        """Computes the combined REINFORCE and affirmative loss for candidate token id sequences.

        Args:
            generations: list of B (T,) tensors
            advantages: (B,) tensor in [0, 1]
            candidate_ids: (N, T) tensor

        Returns:
            loss: (N,) tensor
        """
        t0_candidates = time.time()
        # N = candidate_ids.shape[0]
        # B = len(generations)
        # print(f"    Computing loss for {N} candidates x {B} generations")

        # Compute REINFORCE loss
        t_reinforce_start = time.time()
        reinforce_losses = self.compute_reinforce_loss(generations, advantages, candidate_ids)
        t_reinforce = time.time() - t_reinforce_start
        print(f"    REINFORCE loss computation: {t_reinforce:.3f}s")

        # # Compute affirmative loss
        # t_affirmative_start = time.time()
        # affirmative_losses = self.compute_affirmative_loss(candidate_ids)
        # t_affirmative = time.time() - t_affirmative_start
        # print(f"    Affirmative loss computation: {t_affirmative:.3f}s")

        # Compute entropy maximization loss
        t_entropy_start = time.time()
        entropy_losses = self.compute_entropy_loss(candidate_ids)
        t_entropy = time.time() - t_entropy_start
        # print(f"    Entropy loss computation: {t_entropy:.3f}s")

        # Combine losses
        # print(reinforce_losses, affirmative_losses, entropy_losses)
        losses = reinforce_losses + entropy_losses / 50
        # losses = entropy_losses

        t_total_candidates = time.time() - t0_candidates
        # print(f"    Total candidates loss: {t_total_candidates:.3f}s")
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
    n_optim_ids = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        # when we have a non-differentiable loss, we try fully random sampling
        if grad.isinf().all() or grad.isnan().all():
            raise ValueError("Gradient is all inf or nan")
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = grad.topk(topk, dim=1, largest=False, sorted=False).indices  # (n_optim_ids, topk)

    sampled_ids_pos = torch.randint(
        0, n_optim_ids, (search_width, n_replace), device=grad.device
    )  # (search_width, n_replace)
    sampled_topk_idx = torch.randint(
        0, topk, (search_width, n_replace, 1), device=grad.device
    )

    sampled_ids_val = (
        topk_ids[sampled_ids_pos].gather(2, sampled_topk_idx).squeeze(2)
    )  # (search_width, n_replace)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)  # (search_width, n_optim_ids)

    return new_ids, sampled_ids_pos


def _random_overall(
    ids: Tensor,
    vocab_size: int,
    search_width: int,
    not_allowed_ids: Tensor = None,
):
    """Returns `search_width` random token substitutions.

    Args:
        ids : Tensor, shape = (n_optim_ids,)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace: int
            the number of token positions to update per sequence

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = ids.shape[0]
    original_ids = ids.repeat(search_width, 1)

    # Create valid token mask
    valid_tokens = torch.ones(vocab_size, dtype=torch.bool, device=ids.device)
    if not_allowed_ids is not None:
        valid_tokens[not_allowed_ids.to(ids.device)] = False

    # Sample positions and token indices
    sampled_ids_pos = torch.randint(0, n_optim_tokens, (search_width, 1), device=ids.device)
    valid_token_indices = torch.nonzero(valid_tokens).squeeze()
    sampled_topk_idx = valid_token_indices[torch.randint(0, valid_token_indices.size(0), (search_width, 1), device=ids.device)]

    # Create new sequences with substitutions
    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_topk_idx)
    return new_ids, sampled_ids_pos
