from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CARGO_SPECS = {
    "vaccine": {"low": 2.0, "high": 8.0, "tolerance_steps": 30, "k": 0.08},
    "insulin": {"low": 2.0, "high": 8.0, "tolerance_steps": 20, "k": 0.10},
    "blood": {"low": 1.0, "high": 6.0, "tolerance_steps": 10, "k": 0.12},
    "organ": {"low": 0.0, "high": 4.0, "tolerance_steps": 0, "k": 0.15},
}


@dataclass
class Shipment:
    id: int
    cargo_type: str
    cargo_temp: float
    temp_lower_bound: float
    temp_upper_bound: float
    destination_node: int
    deadline_step: int
    priority: int
    current_vehicle_id: int = -1
    excursion_count: int = 0
    excursion_duration: int = 0
    in_excursion: bool = False
    is_destroyed: bool = False
    is_delivered: bool = False
    steps_since_last_reading: int = 0
    active_since_step: int = 0


def update_temperature(shipment: Shipment, ambient_hold_temp: float, step_duration_hours: float) -> None:
    if shipment.is_destroyed or shipment.is_delivered:
        return

    k = CARGO_SPECS[shipment.cargo_type]["k"]
    delta_t = -k * (shipment.cargo_temp - ambient_hold_temp) * step_duration_hours
    shipment.cargo_temp = float(np.clip(shipment.cargo_temp + delta_t, -80.0, 80.0))

    out_of_range = shipment.cargo_temp < shipment.temp_lower_bound or shipment.cargo_temp > shipment.temp_upper_bound
    if out_of_range:
        if not shipment.in_excursion:
            shipment.excursion_count += 1
            shipment.in_excursion = True
        shipment.excursion_duration += 1
        if shipment.cargo_type == "organ":
            shipment.is_destroyed = True
    else:
        shipment.in_excursion = False

    shipment.steps_since_last_reading += 1
    if shipment.steps_since_last_reading >= 3:
        shipment.steps_since_last_reading = 0

    tolerance = CARGO_SPECS[shipment.cargo_type]["tolerance_steps"]
    if shipment.excursion_duration > tolerance:
        shipment.is_destroyed = True
