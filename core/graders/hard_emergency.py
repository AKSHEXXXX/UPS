from __future__ import annotations

from .base import BaseGrader


class HardEmergencyCaseGrader(BaseGrader):
    """
    Lightweight emergency-case evaluator for hard runs.

    Scores 1.0 when a run avoids shipment destruction and includes at least one
    explicit emergency action after cargo gets near the hard threshold.
    """

    def _compute(self) -> float:
        if not self.trajectory:
            return 0.0

        emergency_action_seen = False
        destroyed_seen = False
        borderline_seen = False

        for transition in self.trajectory:
            action = transition[1] if len(transition) > 1 else None
            info = transition[3] if len(transition) > 3 else {}
            if action is not None and len(action) > 1 and int(action[1]) in {2, 5}:
                emergency_action_seen = True

            per_shipment = info.get("per_shipment_status", {})
            for shipment in per_shipment.values():
                destroyed_seen = destroyed_seen or bool(shipment.get("is_destroyed", False))
                cargo_temp = float(shipment.get("cargo_temp", 0.0))
                lower = float(shipment.get("temp_lower_bound", cargo_temp))
                upper = float(shipment.get("temp_upper_bound", cargo_temp))
                if cargo_temp <= lower or cargo_temp >= upper:
                    borderline_seen = True

        if destroyed_seen:
            return 0.0
        if borderline_seen and emergency_action_seen:
            return 1.0
        if borderline_seen:
            return 0.5
        return 0.75
