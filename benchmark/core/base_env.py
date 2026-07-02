from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class TaskMeta:
    tier: int
    env_name: str
    horizon: str           # "short" | "medium" | "long"
    reward_density: str    # "sparse" | "dense"
    stochastic: bool
    goal_conditioned: bool
    branching_factor: int


class BenchmarkEnv(ABC):

    @abstractmethod
    def reset(self, seed: int = None) -> np.ndarray: ...

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]: ...

    @abstractmethod
    def clone_state(self) -> Any:
        """Return a deep copy of internal env state (for PerfectModel)."""

    @abstractmethod
    def restore_state(self, state: Any) -> None: ...

    def all_states(self) -> List[Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def task_meta(self) -> TaskMeta: ...

    @property
    @abstractmethod
    def n_actions(self) -> int: ...

    @property
    @abstractmethod
    def max_steps(self) -> int: ...
