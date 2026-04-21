"""UCB1 bandit (non-contextual baseline)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseBandit


class UCB1(BaseBandit):
    """
    UCB1 (Auer et al., 2002) — non-contextual upper confidence bound.

    score_a = Q_a + c * sqrt(ln(t) / N_a)

    Tries each arm once before applying the UCB formula.
    """

    def __init__(self, n_arms: int, c: float = 1.0) -> None:
        super().__init__(n_arms)
        self.c = c
        self.q = np.zeros(n_arms)

    def select(self, context: Optional[np.ndarray] = None) -> int:
        # Force one pull per arm before computing confidence bounds
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return int(untried[0])

        log_t = np.log(self.t + 1)
        ucb_scores = self.q + self.c * np.sqrt(log_t / self.counts)
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        n = self.counts[arm]
        self.q[arm] += (reward - self.q[arm]) / n
        self.t += 1
