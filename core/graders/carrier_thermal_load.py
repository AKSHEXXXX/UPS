from __future__ import annotations

from statistics import mean

from .base import BaseGrader


class CarrierThermalLoadGrader(BaseGrader):
    """
    Scores how often carrier vehicles (vehicles with onboard shipments) keep
    those shipments within their temperature bounds.

    Per-step score:
        (# carrier vehicles whose onboard cargo temps are all in-range)
        / max(1, # carrier vehicles)

    Final score is the average over steps where at least one carrier vehicle exists.
    """

    def _compute(self) -> float:
        if not self.trajectory:
            return 0.0

        step_scores: list[float] = []

        for transition in self.trajectory:
            info = transition[3] if len(transition) > 3 else {}
            per_vehicle = info.get("per_vehicle_status", {})
            per_shipment = info.get("per_shipment_status", {})

            if not per_vehicle or not per_shipment:
                continue

            carrier_vehicle_count = 0
            carrier_temp_ok_count = 0

            for vehicle in per_vehicle.values():
                onboard = vehicle.get("shipments_onboard", [])
                if not onboard:
                    continue

                carrier_vehicle_count += 1
                vehicle_ok = True

                for shipment_id in onboard:
                    shipment = per_shipment.get(int(shipment_id), per_shipment.get(str(int(shipment_id)), {}))
                    if not shipment:
                        vehicle_ok = False
                        break

                    cargo_temp = float(shipment.get("cargo_temp", 0.0))
                    lower = float(shipment.get("temp_lower_bound", cargo_temp))
                    upper = float(shipment.get("temp_upper_bound", cargo_temp))

                    if cargo_temp < lower or cargo_temp > upper:
                        vehicle_ok = False
                        break

                if vehicle_ok:
                    carrier_temp_ok_count += 1

            if carrier_vehicle_count > 0:
                step_scores.append(carrier_temp_ok_count / float(carrier_vehicle_count))

        if not step_scores:
            return 0.0
        return float(mean(step_scores))
