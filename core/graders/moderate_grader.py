from __future__ import annotations

from .base import BaseGrader
from .carrier_thermal_load import CarrierThermalLoadGrader
from .delivery_success import DeliverySuccessGrader
from .thermal_integrity import ThermalIntegrityGrader


class ModerateGrader(BaseGrader):
    """
    Mid-level score adds carrier-level thermal execution quality.
    """

    WEIGHTS = {
        "delivery": 0.50,
        "thermal": 0.30,
        "carrier_thermal_load": 0.20,
    }

    def _compute(self) -> float:
        delivery = DeliverySuccessGrader(self.trajectory).score()
        thermal = ThermalIntegrityGrader(self.trajectory).score()
        carrier_thermal_load = CarrierThermalLoadGrader(self.trajectory).score()
        return (
            self.WEIGHTS["delivery"] * delivery
            + self.WEIGHTS["thermal"] * thermal
            + self.WEIGHTS["carrier_thermal_load"] * carrier_thermal_load
        )
