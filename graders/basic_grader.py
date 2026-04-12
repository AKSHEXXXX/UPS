from __future__ import annotations

from .base import BaseGrader
from .delivery_success import DeliverySuccessGrader
from .thermal_integrity import ThermalIntegrityGrader


class EasyGrader(BaseGrader):
    """
    Intro-level score focused on mission success + temperature safety.

    Includes thermal integrity so even basic scoring rewards safe transport.
    """

    WEIGHTS = {
        "delivery": 0.70,
        "thermal": 0.30,
    }

    def _compute(self) -> float:
        delivery = DeliverySuccessGrader(self.trajectory).score()
        thermal = ThermalIntegrityGrader(self.trajectory).score()
        return self.WEIGHTS["delivery"] * delivery + self.WEIGHTS["thermal"] * thermal
