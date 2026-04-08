from __future__ import annotations

from .base import BaseGrader
from .carrier_thermal_load import CarrierThermalLoadGrader
from .delivery_success import DeliverySuccessGrader
from .efficiency import EfficiencyGrader
from .thermal_integrity import ThermalIntegrityGrader


class HardGrader(BaseGrader):
    """
    Advanced score balances delivery, thermal safety, and operational discipline.
    """

    WEIGHTS = {
        "delivery": 0.40,
        "thermal": 0.25,
        "carrier_thermal_load": 0.15,
        "efficiency": 0.20,
    }

    def _compute(self) -> float:
        delivery = DeliverySuccessGrader(self.trajectory).score()
        thermal = ThermalIntegrityGrader(self.trajectory).score()
        carrier_thermal_load = CarrierThermalLoadGrader(self.trajectory).score()
        efficiency = EfficiencyGrader(self.trajectory).score()
        return (
            self.WEIGHTS["delivery"] * delivery
            + self.WEIGHTS["thermal"] * thermal
            + self.WEIGHTS["carrier_thermal_load"] * carrier_thermal_load
            + self.WEIGHTS["efficiency"] * efficiency
        )
