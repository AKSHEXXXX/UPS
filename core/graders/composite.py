from __future__ import annotations

from .base import BaseGrader
from .delivery_success import DeliverySuccessGrader
from .efficiency import EfficiencyGrader
from .thermal_integrity import ThermalIntegrityGrader


class CompositeGrader(BaseGrader):
    WEIGHTS = {
        "delivery": 0.50,
        "thermal": 0.30,
        "efficiency": 0.20,
    }

    def _compute(self) -> float:
        delivery = DeliverySuccessGrader(self.trajectory).score()
        thermal = ThermalIntegrityGrader(self.trajectory).score()
        efficiency = EfficiencyGrader(self.trajectory).score()
        return (
            self.WEIGHTS["delivery"] * delivery
            + self.WEIGHTS["thermal"] * thermal
            + self.WEIGHTS["efficiency"] * efficiency
        )
