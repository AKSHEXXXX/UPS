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
            self._score = max(0.0, min(1.0, self._score))
            assert 0.0 <= self._score <= 1.0, f"{self.__class__.__name__} returned {self._score}, must be in [0,1]"
        return self._score

    @abstractmethod
    def _compute(self) -> float:
        raise NotImplementedError
