import torch
from collections import defaultdict


class DomainScaler(torch.nn.Module):
    """
    γ_d  =  γ0_d · δ_d
      γ0_d  = (N_d / N_global) / (mu_d / mu_global)
      δ_d   = learned (>0), initialised at 1, L2-regularised toward 1.
    All statistics and parameters are kept internally.
    """
    def __init__(self, lr: float = 3e-6, l2: float = 1e-3, eps: float = 1e-6):
        super().__init__()
        self.l2   = l2
        self.eps  = eps
        # learnable log δ per domain
        self._log_delta = torch.nn.ParameterDict()
        self.opt        = torch.optim.Adam([], lr=lr)

        # running statistics
        self._domain_stats = defaultdict(lambda: {"count": 0, "mean": 0.0})
        self._global_count = 0
        self._global_mean  = 0.0

    # ---------- stats update (called once per mini-batch) ----------
    @torch.no_grad()
    def update_stats(self, raw_scores: torch.Tensor, domains: List[str]):
        """
        raw_scores : 1-D tensor, length = batch size
        domains    : list[str]   same length
        """
        for s, d in zip(raw_scores.tolist(), domains):
            # per-domain running mean
            d_stat = self._domain_stats[d]
            d_stat["count"] += 1
            d_stat["mean"] += (s - d_stat["mean"]) / d_stat["count"]
            # global running mean
            self._global_count += 1
            self._global_mean  += (s - self._global_mean) / self._global_count

    def gamma(self, domain: str, device):
        """
        Returns γ_d  and its L2 reg-loss.
        Computes γ0_d on the fly from internal stats.
        """
        # 1) make sure δ_d param exists
        if domain not in self._log_delta:
            p = torch.nn.Parameter(torch.zeros(()))  # log δ=0 ⇒ δ=1
            self._log_delta[domain] = p
            self.opt.add_param_group({"params": p})

        # 2) compute heuristic γ0_d from current stats
        d_stat  = self._domain_stats[domain]
        Nd      = max(d_stat["count"], 1)
        mu_d    = d_stat["mean"]
        Ng      = max(self._global_count, 1)
        mu_g    = self._global_mean if abs(self._global_mean) > self.eps else self.eps
        gamma0  = (Nd / Ng) / (mu_d / mu_g + self.eps)

        # 3) learned residual δ_d > 0
        delta   = torch.nn.functional.softplus(self._log_delta[domain])
        gamma   = (gamma0 * delta).to(device)

        # 4) L2 regulariser pulling δ→1
        reg_loss = 0.5 * self.l2 * (delta - 1.0) ** 2
        return gamma, reg_loss.to(device)

    # ---------- optimiser step -------------------------------------
    def step(self):
        self.opt.step()
        self.opt.zero_grad()
