import time
from dataclasses import dataclass


@dataclass
class BudgetConfig:
    max_model_calls: int = 1000
    max_wall_time_s: float = 30.0


class BudgetExhausted(Exception):
    pass


class PlanningBudget:
    def __init__(self, config: BudgetConfig):
        self._config = config
        self._calls = 0
        self._t0 = time.time()

    def consume(self, n: int = 1) -> None:
        self._calls += n
        if self._calls >= self._config.max_model_calls:
            raise BudgetExhausted(f"model_calls limit ({self._config.max_model_calls})")
        if time.time() - self._t0 >= self._config.max_wall_time_s:
            raise BudgetExhausted("wall_time limit")

    @property
    def model_calls_used(self) -> int:
        return self._calls

    @property
    def wall_time_s(self) -> float:
        return time.time() - self._t0

    @property
    def exhausted(self) -> bool:
        return (
            self._calls >= self._config.max_model_calls
            or self.wall_time_s >= self._config.max_wall_time_s
        )
