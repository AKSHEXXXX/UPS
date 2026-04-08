from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Set

import numpy as np

from .config import ColdChainConfig


class VehicleStatus(IntEnum):
    IDLE = 0
    IN_TRANSIT = 1
    AT_STOP = 2
    BROKEN = 3


class RefrigStatus(IntEnum):
    WORKING = 0
    DEGRADED = 1
    FAILED = 2


@dataclass
class Vehicle:
    id: int
    location: int
    status: VehicleStatus = VehicleStatus.IDLE
    refrig_status: RefrigStatus = RefrigStatus.WORKING
    fuel_level: float = 1.0
    steps_until_next_waypoint: int = 0
    route: List[int] = field(default_factory=list)
    shipments_onboard: List[int] = field(default_factory=list)
    nearest_cold_depot: int = -1
    steps_to_cold_depot: int = 9999
    steps_to_destination: int = 9999
    detour_cost: int = 9999
    visited_nodes_this_route: Set[int] = field(default_factory=set)
    hold_temp: float = 4.0
    dock_steps_remaining: int = 0


def get_hold_temperature(vehicle: Vehicle, outdoor_temp: float, refrig_setpoint: float = 4.0) -> float:
    if vehicle.refrig_status == RefrigStatus.WORKING:
        return refrig_setpoint
    if vehicle.refrig_status == RefrigStatus.DEGRADED:
        return vehicle.hold_temp
    return outdoor_temp


def maybe_degrade_refrigeration(vehicle: Vehicle, rng: np.random.Generator, config: ColdChainConfig) -> None:
    if vehicle.refrig_status == RefrigStatus.WORKING:
        if rng.random() < config.refrigeration_degradation_prob:
            vehicle.refrig_status = RefrigStatus.DEGRADED
    elif vehicle.refrig_status == RefrigStatus.DEGRADED:
        if rng.random() < config.refrigeration_degradation_prob * 2:
            vehicle.refrig_status = RefrigStatus.FAILED


def maybe_breakdown(vehicle: Vehicle, rng: np.random.Generator, config: ColdChainConfig) -> None:
    if vehicle.status != VehicleStatus.BROKEN and rng.random() < config.breakdown_probability:
        vehicle.status = VehicleStatus.BROKEN


def step_dock_timer(vehicle: Vehicle, refrig_setpoint: float = 4.0) -> None:
    if vehicle.dock_steps_remaining <= 0:
        return
    vehicle.dock_steps_remaining -= 1
    if vehicle.dock_steps_remaining == 0:
        vehicle.refrig_status = RefrigStatus.WORKING
        vehicle.hold_temp = refrig_setpoint
        vehicle.status = VehicleStatus.IDLE
