from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseGrader(ABC):
    def __init__(self, trajectory: List[Tuple]):
        self.trajectory = trajectory
        self._score: Optional[float] = None

    def score(self) -> float:
        if self._score is None:
            self._score = float(self._compute())
            self._score = max(1e-6, min(1 - 1e-6, self._score))    # ← strict open interval
            assert 1e-6 <= self._score <= 1 - 1e-6, f"{self.__class__.__name__} returned {self._score}, must be in (0,1)"
        return self._score

    @abstractmethod
    def _compute(self) -> float:
        raise NotImplementedError
