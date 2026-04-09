# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ColdChainAction(Action):
    vehicle_index: int = Field(..., description="Vehicle slot to command; use n_vehicles for global no-op")
    action_type: int = Field(..., description="0=WAIT, 1=REROUTE, 2=DIVERT_COLD_DEPOT, 3=SWAP_VEHICLE, 4=EXPEDITE, 5=ABORT")
    target_index: int = Field(0, description="Target node / vehicle index / ignored depending on action type")


class GlobalTelemetry(BaseModel):
    ambient_temperature: float = Field(..., description="Outdoor temperature in Celsius")
    time_of_day: float = Field(..., description="Normalized time-of-day in [0,1]")
    traffic_multiplier: float = Field(..., description="Traffic multiplier")
    weather_event: int = Field(..., description="0=none,1=heatwave,2=storm,3=power_outage")
    hub_cold_storage_temp: float = Field(..., description="Hub cold storage temperature in Celsius")
    steps_elapsed: int = Field(..., description="Episode steps elapsed")
    steps_remaining: int = Field(..., description="Episode steps remaining")


class VehicleTelemetry(BaseModel):
    id: int
    location: int
    status: int
    refrigeration_status: int
    fuel_level: float
    steps_until_next_waypoint: int
    shipments_onboard: List[int] = Field(default_factory=list)
    nearest_cold_depot_node: int = -1
    steps_to_cold_depot: int = 9999
    steps_to_destination: int = 9999
    detour_cost_to_depot: int = 9999
    hold_temp: float = 4.0
    dock_steps_remaining: int = 0


class ShipmentTelemetry(BaseModel):
    id: int
    cargo_temp: float
    temp_lower_bound: float
    temp_upper_bound: float
    time_to_deadline: int
    current_vehicle: int
    destination_node: int
    priority: int
    cargo_type: int
    excursion_count: int
    excursion_duration: int
    is_destroyed: int
    is_delivered: int
    steps_since_last_reading: int


class ColdChainObservation(Observation):
    global_state: GlobalTelemetry
    vehicles: List[VehicleTelemetry] = Field(default_factory=list)
    shipments: List[ShipmentTelemetry] = Field(default_factory=list)
    action_mask: List[int] = Field(default_factory=list)
    action_was_masked: bool = False
    illegal_action_count: int = 0
    last_action_error: str | None = None
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    episode_summary: Dict[str, Any] = Field(default_factory=dict)
    grader_scores: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


class ColdChainState(State):
    illegal_action_count: int = 0
    terminated: bool = False
    truncated: bool = False
    steps_remaining: int = 0
