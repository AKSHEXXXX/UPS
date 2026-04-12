from __future__ import annotations

from statistics import mean
from typing import Any, Dict

from .base import BaseGrader


def _transition_info(transition: Any) -> Dict:
    if isinstance(transition, dict):
        return dict(transition.get("info", {}))
    if isinstance(transition, (tuple, list)) and len(transition) > 3 and isinstance(transition[3], dict):
        return dict(transition[3])
    return {}


class ThermalIntegrityGrader(BaseGrader):
    def _compute(self) -> float:
        if not self.trajectory:
            return 0.0

        final_info = _transition_info(self.trajectory[-1])
        shipment_states = final_info.get("per_shipment_status", {})
        if not shipment_states:
            return 0.0

        scores = []
        total_steps = max(1, len(self.trajectory))
        for shipment in shipment_states.values():
            if int(shipment.get("current_vehicle_id", -1)) < 0 and not bool(shipment.get("is_delivered", False)):
                continue

            excursion_duration = float(shipment.get("excursion_duration", 0.0))
            active_since_step = int(shipment.get("active_since_step", 0))
            total_active_duration = max(1, total_steps - active_since_step)
            per_shipment_score = 1.0 - (excursion_duration / total_active_duration)
            scores.append(max(0.0, min(1.0, per_shipment_score)))

        if not scores:
            return 0.0
        return float(mean(scores))
