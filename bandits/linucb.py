"""LinUCB Disjoint — the primary contextual bandit algorithm.

Reference: Li et al., "A Contextual-Bandit Approach to Personalized
           News Article Recommendation", WWW 2010.

Each arm keeps its own ridge-regression model:
    A_a  ∈ R^{d×d}   initialized to λ·I
    b_a  ∈ R^d        initialized to 0

UCB score for arm a given context x:
    θ_a  = A_a^{-1} b_a
    p_a  = x^T θ_a  +  α · sqrt(x^T A_a^{-1} x)

Update on observed reward r for chosen arm a*:
    A_{a*} += x x^T
    b_{a*} += r · x
"""
from __future__ import annotations

import numpy as np

from .base import BaseBandit


class LinUCBDisjoint(BaseBandit):
    """
    LinUCB with disjoint linear models.

    Parameters
    ----------
    n_arms : int
        Number of actions.
    d : int
        Dimension of the context vector.
    alpha : float
        Exploration bonus coefficient.  Higher → more exploration.
        Typical range: [0.1, 2.0].  Default 1.0.
    reg_lambda : float
        Ridge regularisation for A initialisation (λ·I).
        Acts as prior strength.  Default 1.0.
    """

    def __init__(
        self,
        n_arms: int,
        d: int,
        alpha: float = 1.0,
        reg_lambda: float = 1.0,
    ) -> None:
        super().__init__(n_arms)
        self.d = d
        self.alpha = alpha
        self.reg_lambda = reg_lambda

        # Per-arm parameters
        # A[a] = λI + Σ x_t x_t^T   (d × d)
        # b[a] = Σ r_t x_t           (d,)
        self.A = reg_lambda * np.tile(np.eye(d), (n_arms, 1, 1))  # (K, d, d)
        self.b = np.zeros((n_arms, d))                             # (K, d)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _theta(self, arm: int) -> np.ndarray:
        """Posterior mean for arm (uses solve for numerical stability)."""
        return np.linalg.solve(self.A[arm], self.b[arm])

    def _ucb_scores(self, x: np.ndarray) -> np.ndarray:
        """Compute UCB score for every arm given normalised context x."""
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta_a = np.linalg.solve(self.A[a], self.b[a])
            # Solve A_a v = x  →  v = A_a^{-1} x
            v = np.linalg.solve(self.A[a], x)
            exploit = x @ theta_a
            explore = self.alpha * np.sqrt(x @ v)
            scores[a] = exploit + explore
        return scores

    # ------------------------------------------------------------------
    # BaseBandit API
    # ------------------------------------------------------------------

    def select(self, context: np.ndarray) -> int:  # type: ignore[override]
        scores = self._ucb_scores(context.astype(float))
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:  # type: ignore[override]
        x = context.astype(float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.t += 1

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def theta_matrix(self) -> np.ndarray:
        """Return (n_arms, d) matrix of posterior means."""
        return np.stack([self._theta(a) for a in range(self.n_arms)])

    def confidence_widths(self, x: np.ndarray) -> np.ndarray:
        """Return per-arm exploration bonus widths for context x."""
        widths = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            v = np.linalg.solve(self.A[a], x.astype(float))
            widths[a] = self.alpha * np.sqrt(x @ v)
        return widths
