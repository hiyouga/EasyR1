# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


domain_running_stats = defaultdict(lambda: {
    "mean": 0.0,
    "var": 0.0,    # We'll track population variance via Welford's algorithm, or something similar
    "count": 0
})
global_running_stats = {"mean": 0.0, "var": 0.0, "count": 0}
GLOBAL_TRAIN_STEP = 0        # you can increment this elsewhere
FADE_STEPS = 100             # after this many steps, group-based re-centering is 0


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None:
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


def update_global_stats(x: float):
    """Online update of overall mean/variance with Welford’s algorithm."""
    stats = global_running_stats
    stats["count"] += 1
    c = stats["count"]
    delta = x - stats["mean"]
    stats["mean"] += delta / c
    stats["var"] += delta * (x - stats["mean"])

def get_global_mean(eps: float = 1e-6) -> float:
    return global_running_stats["mean"] + eps    # avoid zero‑division later


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def update_domain_stats(domain: str, x: float):
    stats = domain_running_stats[domain]
    stats["count"] += 1
    c = stats["count"]
    delta = x - stats["mean"]
    stats["mean"] += delta / c
    stats["var"] += delta * (x - stats["mean"])


def get_domain_mean_std(domain: str, eps=1e-6):
    stats = domain_running_stats[domain]
    c = stats["count"]
    if c < 2:
        # not enough data, fallback
        return stats["mean"], 1.0
    mean = stats["mean"]
    var = stats["var"] / (c - 1)
    std = max(var, eps) ** 0.5
    return mean, std


def _quantile_safe(x: torch.Tensor, q: float, eps: float) -> torch.Tensor:
    """Robust percentile that never returns 0 (to avoid divide-by-zero)."""
    if torch.allclose(x, torch.zeros_like(x)):
        return torch.tensor(eps, dtype=x.dtype, device=x.device)
    return torch.quantile(x, q).detach().clamp_min(eps)


import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Tuple
import numpy as np


# ----  helper --------------------------------------------------------------
def _quantile_safe(x: torch.Tensor, q: float, eps: float) -> torch.Tensor:
    """Robust percentile that never returns 0 (to avoid divide-by-zero)."""
    if torch.allclose(x, torch.zeros_like(x)):
        return torch.tensor(eps, dtype=x.dtype, device=x.device)
    return torch.quantile(x, q).detach().clamp_min(eps)


# ----  main ----------------------------------------------------------------
@torch.no_grad()
def compute_drpo_outcome_advantage(
    token_level_rewards: torch.Tensor,       # (bs, seq_len)  —  reward **before** KL penalty
    response_mask:      torch.Tensor,        # (bs, seq_len)
    index:              np.ndarray,          # (bs,)          —  question UUIDs
    domain_info:        np.ndarray,          # (bs,)          —  domain strings
    log_probs:          torch.Tensor,        # (bs, seq_len)  —  log πθ
    ref_log_probs:      torch.Tensor,        # (bs, seq_len)  —  log πref
    eps: float = 1e-6,
    kl_q: float = 0.75,                      # percentile for the “soft” damper
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DRPO outcome-advantage with inverse-linear KL-aware damping.

    Steps
    -----
    1.   Sum token rewards → per-rollout raw score.
    2.   Update running mean/var per domain & global.
    3.   GRPO question-wise normalisation.
    4.   Domain temperature scaling.
    5.   **NEW**  KL-based inverse-linear damping
         m =  t / (z + t)  where  z = max(score * KL , 0),
         t = 75-th percentile of z in the current mini-batch.
    6.   Broadcast back to token level and return (advantages, returns).

    The function purposely *does not* subtract β·KL; that will be done
    later by `apply_kl_penalty`.
    """
    B, L = token_level_rewards.shape
    device = token_level_rewards.device

    # ------------------------------------------------------------------ #
    # 1) outcome score per rollout (raw reward, summed over tokens)
    # ------------------------------------------------------------------ #
    raw_scores = token_level_rewards.sum(dim=-1)                     # (B,)

    # ------------------------------------------------------------------ #
    # 2) update running stats (domain / global)            – unchanged –
    # ------------------------------------------------------------------ #
    for i in range(B):
        r = raw_scores[i].item()
        d = domain_info[i]
        update_domain_stats(d, r)
        update_global_stats(r)

    # ------------------------------------------------------------------ #
    # 3) GRPO question-wise normalisation                – unchanged –
    # ------------------------------------------------------------------ #
    scores = raw_scores.clone()                                      # (B,)
    id2score = defaultdict(list)

    for i in range(B):
        id2score[index[i]].append(scores[i])

    id2mean = {k: torch.mean(torch.stack(v)) for k, v in id2score.items()}
    id2std  = {k: torch.std (torch.stack(v)) for k, v in id2score.items()}

    for i in range(B):
        mu, sd = id2mean[index[i]], id2std[index[i]]
        scores[i] = (scores[i] - mu) / (sd + eps)

    # ------------------------------------------------------------------ #
    # 4) domain temperature scaling
    # ------------------------------------------------------------------ #
    N_total  = float(global_running_stats["count"])
    mu_total = float(global_running_stats["mean"]) if abs(global_running_stats["mean"]) > eps else eps

    for i in range(B):
        d_stats       = domain_running_stats[domain_info[i]]
        N_d, mu_d     = float(d_stats["count"]), float(d_stats["mean"])
        T_d           = max((N_d / N_total) * (mu_d / mu_total), eps)
        scores[i]     = scores[i] / T_d                               # (B,)

    # ------------------------------------------------------------------ #
    # 5)  KL-aware inverse-linear damping  (new)
    # ------------------------------------------------------------------ #
    # 5.1  rollout-level KL  (low-variance surrogate)
    kl_tok      = compute_kl(log_probs, ref_log_probs, "low_var_kl")  # (B,L)
    kl_tok      = kl_tok * response_mask
    kl_rollout  = kl_tok.sum(dim=-1)                                  # (B,)

    # 5.2  "push-away mass"
    z = torch.relu(scores * kl_rollout)                               # (B,)

    # 5.3  scale t = 75-th percentile of z
    t = _quantile_safe(z, kl_q, eps)                                  # scalar

    # 5.4  inverse-linear multiplier  m = t / (z + t)
    m = t / (z + t)                                                   # (B,)

    # 5.5  apply damping to the *reward* part only
    scores = m * scores                                               # (B,)

    # ------------------------------------------------------------------ #
    # 6) broadcast back to token level
    # ------------------------------------------------------------------ #
    returns = scores.unsqueeze(-1) * response_mask                    # (B,L)
    return returns, returns



@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the policy loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy

    """
    negative_approx_kl = log_probs - old_log_probs
    # clamp the ratio before exp to avoid nan
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(negative_approx_kl)
    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * clip_ratio_dual

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    pg_clipfrac_higher = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    pg_clipfrac_lower = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = VF.masked_mean(final_pg_loss, response_mask)
    pg_clipfrac_higher = VF.masked_mean(pg_clipfrac_higher, response_mask)
    pg_clipfrac_lower = VF.masked_mean(pg_clipfrac_lower, response_mask)
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)
    return final_pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    cliprange_value: float,
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_loss1, vf_loss2), action_mask)  # clip if vf_loss1 < vf_loss2
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), action_mask)
    return vf_loss, vf_clipfrac


def compute_kl(log_probs: torch.FloatTensor, ref_log_probs: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        kl = ref_log_probs - log_probs
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
