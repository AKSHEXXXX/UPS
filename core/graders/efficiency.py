from __future__ import annotations

from .base import BaseGrader


def _action_type(action) -> int:
    if action is None:
        return 0
    try:
        return int(action[1])
    except Exception:
        return 0


class EfficiencyGrader(BaseGrader):
    def _compute(self) -> float:
        if not self.trajectory:
            return 0.0

        final_info = self.trajectory[-1][3] if len(self.trajectory[-1]) > 3 else {}
        shipment_states = final_info.get("per_shipment_status", {})
        vehicle_states = final_info.get("per_vehicle_status", {})

        fuel_used = 0.0
        direct_delivery_costs = 0.0
        detour_costs = 0.0
        abort_count = 0
        unnecessary_aborts = 0

        for obs, action, reward, info in self.trajectory:
            action_type = _action_type(action)
            if action_type == 4:
                fuel_used += 0.1
            elif action_type == 1:
                fuel_used += 0.01
            elif action_type == 2:
                fuel_used += 0.03
                vehicle_index = int(action[0]) if action is not None else 0
                vehicle_info = info.get("per_vehicle_status", {}).get(vehicle_index, {})
                detour_costs += float(vehicle_info.get("detour_cost", 0.0))
            elif action_type == 3:
                fuel_used += 0.05
            elif action_type == 5:
                fuel_used += 0.2
                abort_count += 1
                vehicle_index = int(action[0]) if action is not None else 0
                vehicle_info = info.get("per_vehicle_status", {}).get(vehicle_index, {})
                vehicle_shipments = vehicle_info.get("shipments_onboard", [])
                if vehicle_shipments:
                    shipment_id = int(vehicle_shipments[0])
                    shipment_info = info.get("per_shipment_status", {}).get(shipment_id, {})
                    tolerance_steps = 1.0 if shipment_info.get("cargo_type") == "organ" else 10.0
                    if float(shipment_info.get("excursion_duration", 0.0)) < tolerance_steps * 0.5:
                        unnecessary_aborts += 1

        for shipment in shipment_states.values():
            direct_delivery_costs += float(shipment.get("deadline_step", 1))

        max_possible_fuel = max(1.0, len(self.trajectory) * 0.2)
        fuel_used_ratio = min(1.0, fuel_used / max_possible_fuel)
        unnecessary_aborts_rate = unnecessary_aborts / max(1, abort_count)
        detour_overhead = detour_costs / max(1.0, direct_delivery_costs)

        penalty = 0.4 * fuel_used_ratio + 0.3 * unnecessary_aborts_rate + 0.3 * detour_overhead
        return 1.0 - max(0.0, min(1.0, penalty))
