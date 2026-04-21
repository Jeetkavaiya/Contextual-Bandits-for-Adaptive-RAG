"""Linear Thompson Sampling — Bayesian contextual bandit.

Maintains a Gaussian posterior over the reward parameter θ_a for each arm:

    Prior:      θ_a ~ N(0, v²·I)
    Likelihood: r | x, θ_a ~ N(x^T θ_a, σ²)

After observing (x_t, r_t) for arm a:
    B_a  += x_t x_t^T      (B_a = λI + Σ x_t x_t^T)
    f_a  += r_t · x_t

Posterior mean:   μ_a = B_a^{-1} f_a
Posterior cov:    Σ_a = v² · B_a^{-1}

At each step: sample θ̃_a ~ N(μ_a, Σ_a), pick arm a* = argmax x^T θ̃_a.

Reference: Agrawal & Goyal, "Thompson Sampling for Contextual Bandits
           with Linear Payoffs", ICML 2013.
"""
from __future__ import annotations

import numpy as np

from .base import BaseBandit


class LinearThompsonSampling(BaseBandit):
    """
    Thompson Sampling with linear Gaussian reward model.

    Parameters
    ----------
    n_arms : int
    d : int
        Context dimensionality.
    v : float
        Prior / exploration variance scale.  Higher → more exploration.
        Typical range: [0.1, 5.0].  Default 1.0.
    reg_lambda : float
        Ridge regularisation (λ·I in B initialisation).  Default 1.0.
    """

    def __init__(
        self,
        n_arms: int,
        d: int,
        v: float = 1.0,
        reg_lambda: float = 1.0,
    ) -> None:
        super().__init__(n_arms)
        self.d = d
        self.v = v
        self.reg_lambda = reg_lambda

        # Sufficient statistics per arm
        self.B = reg_lambda * np.tile(np.eye(d), (n_arms, 1, 1))  # (K, d, d)
        self.f = np.zeros((n_arms, d))                             # (K, d)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _posterior(self, arm: int):
        """Returns (mu, cov) of posterior for given arm."""
        B_inv = np.linalg.solve(self.B[arm], np.eye(self.d))
        mu = B_inv @ self.f[arm]
        cov = (self.v ** 2) * B_inv
        return mu, cov

    # ------------------------------------------------------------------
    # BaseBandit API
    # ------------------------------------------------------------------

    def select(self, context: np.ndarray) -> int:  # type: ignore[override]
        x = context.astype(float)
        best_arm = 0
        best_score = -np.inf

        for a in range(self.n_arms):
            mu, cov = self._posterior(a)
            # Sample θ̃_a from posterior
            try:
                theta_sample = np.random.multivariate_normal(mu, cov)
            except np.linalg.LinAlgError:
                # Fallback: use mean if covariance is degenerate
                theta_sample = mu
            score = x @ theta_sample
            if score > best_score:
                best_score = score
                best_arm = a

        return best_arm

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:  # type: ignore[override]
        x = context.astype(float)
        self.B[arm] += np.outer(x, x)
        self.f[arm] += reward * x
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.t += 1

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------

    def posterior_means(self) -> np.ndarray:
        """Return (n_arms, d) matrix of posterior means."""
        return np.stack([self._posterior(a)[0] for a in range(self.n_arms)])
