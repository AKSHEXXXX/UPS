from __future__ import annotations

from typing import Dict, Iterable

from .base import BaseGrader


def _priority_weight(priority: int) -> float:
    if priority >= 2:
        return 2.5
    if priority == 1:
        return 1.5
    return 1.0


class DeliverySuccessGrader(BaseGrader):
    def _compute(self) -> float:
        if not self.trajectory:
            return 0.0

        final_info = self.trajectory[-1][3] if len(self.trajectory[-1]) > 3 else {}
        shipment_states: Dict[str, Dict] = final_info.get("per_shipment_status", {})
        if not shipment_states:
            return 0.0

        delivered_steps = self._delivery_steps()
        weighted_score = 0.0
        max_weight = 0.0

        for raw_id, shipment in shipment_states.items():
            shipment_id = int(raw_id)
            priority = int(shipment.get("priority", 0))
            weight = _priority_weight(priority)
            max_weight += weight

            delivered = bool(shipment.get("is_delivered", False))
            destroyed = bool(shipment.get("is_destroyed", False))
            excursion_duration = float(shipment.get("excursion_duration", 0.0))
            deadline_step = int(shipment.get("deadline_step", 0))
            delivered_step = delivered_steps.get(shipment_id)

            if destroyed or delivered_step is None:
                shipment_score = 0.0
            else:
                on_time = delivered_step <= deadline_step
                had_excursion = excursion_duration > 0.0
                if on_time and not had_excursion:
                    shipment_score = 1.0
                elif on_time and had_excursion:
                    shipment_score = 0.6
                elif not on_time and not had_excursion:
                    shipment_score = 0.3
                else:
                    shipment_score = 0.1

            weighted_score += shipment_score * weight

        if max_weight <= 0.0:
            return 0.0
        return weighted_score / max_weight

    def _delivery_steps(self) -> dict[int, int]:
        delivered_steps: dict[int, int] = {}
        for step_index, transition in enumerate(self.trajectory):
            info = transition[3] if len(transition) > 3 else {}
            states = info.get("per_shipment_status", {})
            for raw_id, shipment in states.items():
                shipment_id = int(raw_id)
                if bool(shipment.get("is_delivered", False)) and shipment_id not in delivered_steps:
                    delivered_steps[shipment_id] = step_index
        return delivered_steps
