"""Abstract base class for all bandit algorithms."""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class BaseBandit(ABC):
    """
    Common interface for all bandit algorithms.

    Contextual bandits receive a context vector at each step.
    Non-contextual bandits ignore context (pass None).
    """

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms
        self.t: int = 0                              # total steps
        self.counts = np.zeros(n_arms, dtype=int)    # pulls per arm
        self.sum_rewards = np.zeros(n_arms)          # cumulative reward per arm

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @abstractmethod
    def select(self, context: Optional[np.ndarray] = None) -> int:
        """Return the index of the chosen arm given optional context."""

    @abstractmethod
    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        """Update internal state after observing reward for chosen arm."""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def mean_rewards(self) -> np.ndarray:
        """Empirical mean reward per arm (0 for unpulled arms)."""
        with np.errstate(invalid="ignore"):
            return np.where(self.counts > 0, self.sum_rewards / self.counts, 0.0)

    def best_arm(self) -> int:
        """Arm with highest empirical mean (greedy)."""
        return int(np.argmax(self.mean_rewards()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))
