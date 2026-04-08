from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np

from .config import ColdChainConfig
from .shipment import Shipment
from .vehicle import Vehicle, VehicleStatus


def compute_action_mask(vehicles: List[Vehicle], shipments: List[Shipment], graph: nx.Graph, config: ColdChainConfig, difficulty: int = 3) -> np.ndarray:
    total = (config.n_vehicles + 1) * 6 * config.n_nodes
    mask = np.zeros(total, dtype=np.int8)

    cold_depots = list(graph.graph.get("cold_depot_nodes", []))

    for vehicle_index, vehicle in enumerate(vehicles):
        base = vehicle_index * 6 * config.n_nodes
        if vehicle.status == VehicleStatus.BROKEN:
            continue

        # WAIT ignores target_index, only allow target 0 to prevent bias.
        mask[base + 0 * config.n_nodes] = 1
        if vehicle_index == 0:
             # print(f"DEBUG: Vehicle 0 shipments: {vehicle.shipments_onboard}")
             pass

        # Prevent route-reset thrashing: while already moving to a waypoint, disallow
        # actions that reset the route and prevent location updates.
        is_in_transit = vehicle.status == VehicleStatus.IN_TRANSIT and vehicle.steps_until_next_waypoint > 0

        if not is_in_transit:
            for node in range(config.n_nodes):
                if node != vehicle.location:
                    mask[base + 1 * config.n_nodes + node] = 1

        if difficulty >= 2:
            # DIVERT: only legal if NOT already standing at a cold depot.
            # Prevents the "depot-hop" exploit where the agent repeatedly
            # re-issues DIVERT to reset the stall counter.
            if vehicle.location not in cold_depots:
                for depot_node in cold_depots:
                    mask[base + 2 * config.n_nodes + depot_node] = 1

        if difficulty >= 3:
            for other_vehicle in vehicles:
                if (
                    other_vehicle.id != vehicle.id
                    and other_vehicle.location == vehicle.location
                    and other_vehicle.status != VehicleStatus.BROKEN
                    and len(other_vehicle.shipments_onboard) < config.max_cargo_per_vehicle
                ):
                    mask[base + 3 * config.n_nodes + other_vehicle.id] = 1

        if difficulty >= 1 and vehicle.fuel_level >= 0.1 and vehicle.status == VehicleStatus.IN_TRANSIT:
            # EXPEDITE only at Difficulty 3? No, let's keep it for Level 1 too?
            # Actually, the plan says Level 3 adds all other actions.
            if difficulty >= 3:
                mask[base + 4 * config.n_nodes] = 1

        if len(vehicle.shipments_onboard) > 0 and difficulty >= 3:
            # ABORT only at Difficulty 3.
            mask[base + 5 * config.n_nodes] = 1

        row_start = base
        row_end = base + 6 * config.n_nodes
        if not np.any(mask[row_start:row_end]):
            mask[base + 0 * config.n_nodes] = 1

    global_base = config.n_vehicles * 6 * config.n_nodes
    # Global no-op also ignores target_index, only allow target 0.
    mask[global_base + 0 * config.n_nodes] = 1
    active_shipments = any(not shipment.is_delivered and not shipment.is_destroyed for shipment in shipments)
    # Keep no-op only when no active work remains.
    if not active_shipments:
        for vehicle_index, vehicle in enumerate(vehicles):
            if vehicle.status != VehicleStatus.BROKEN:
                mask[vehicle_index * 6 * config.n_nodes + 0 * config.n_nodes : vehicle_index * 6 * config.n_nodes + 1 * config.n_nodes] = 1
    return mask


def flatten_action(action, config: ColdChainConfig) -> int:
    if isinstance(action, dict):
        vehicle_index = int(action["vehicle_index"])
        action_type   = int(action["action_type"])
        target_index  = int(action.get("target_index", 0))
    else:
        vehicle_index = int(action[0])
        action_type   = int(action[1])
        target_index  = int(action[2])
    return vehicle_index * 6 * config.n_nodes + action_type * config.n_nodes + target_index
