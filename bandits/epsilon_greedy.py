"""Epsilon-Greedy bandit (non-contextual baseline)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseBandit


class EpsilonGreedy(BaseBandit):
    """
    ε-greedy: explore randomly with probability ε, exploit best arm otherwise.

    Two schedules supported:
      - fixed:   ε stays constant
      - decay:   ε_t = ε_0 / (1 + decay_rate * t)  (annealing)
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        decay_rate: float = 0.0,
    ) -> None:
        super().__init__(n_arms)
        self.epsilon0 = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.q = np.zeros(n_arms)  # incremental mean estimates

    def _current_epsilon(self) -> float:
        if self.decay_rate > 0.0:
            return self.epsilon0 / (1.0 + self.decay_rate * self.t)
        return self.epsilon0

    def select(self, context: Optional[np.ndarray] = None) -> int:
        eps = self._current_epsilon()
        if np.random.random() < eps:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.q))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        # Incremental mean update
        n = self.counts[arm]
        self.q[arm] += (reward - self.q[arm]) / n
        self.t += 1
        self.epsilon = self._current_epsilon()
