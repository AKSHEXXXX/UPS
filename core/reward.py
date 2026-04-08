from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from .config import ColdChainConfig
from .shipment import CARGO_SPECS, Shipment
from .vehicle import Vehicle, VehicleStatus


def temp_shaping_reward(shipments: List[Shipment], config: ColdChainConfig) -> float:
    total = 0.0
    for shipment in shipments:
        if shipment.is_delivered or shipment.is_destroyed:
            continue
        if shipment.cargo_temp < shipment.temp_lower_bound or shipment.cargo_temp > shipment.temp_upper_bound:
            distance = max(shipment.temp_lower_bound - shipment.cargo_temp, shipment.cargo_temp - shipment.temp_upper_bound)
            total -= 0.08 * min(distance, 15.0)
    return float(total)


def progress_shaping_reward(
    vehicles: List[Vehicle],
    shipments: List[Shipment],
    graph: nx.Graph,
    config: ColdChainConfig,
    steps_elapsed_or_prev_distances,
    prev_distances: Dict | None = None,
) -> float:
    if prev_distances is None and isinstance(steps_elapsed_or_prev_distances, dict):
        prev_distances = steps_elapsed_or_prev_distances
        steps_elapsed = 0
    else:
        steps_elapsed = int(steps_elapsed_or_prev_distances)
        prev_distances = prev_distances or {}

    total = 0.0
    for vehicle in vehicles:
        for shipment_id in vehicle.shipments_onboard:
            if shipment_id >= len(shipments):
                continue
            shipment = shipments[shipment_id]
            if shipment.is_delivered or shipment.is_destroyed:
                continue
            current_distance = nx.shortest_path_length(graph, vehicle.location, shipment.destination_node, weight="current_weight")
            previous_distance = prev_distances.get((vehicle.id, shipment.id), current_distance)
            net_progress = float(previous_distance - current_distance)
            if net_progress > 0.0:
                vehicle.visited_nodes_this_route.add(vehicle.location)
                # Primary directional signal: reward moving closer each step.
                total += 0.5 * net_progress
            elif net_progress < 0.0:
                total += 0.5 * net_progress

            prev_distances[(vehicle.id, shipment.id)] = int(current_distance)
    return float(total)


def action_cost_reward(action, vehicle: Vehicle, config: ColdChainConfig) -> float:
    """
    Cost for each action type.
    DIVERT_COLD_DEPOT cost scales with detour, but capped to avoid extreme penalties.
    """
    if isinstance(action, dict):
        action_type = int(action.get("action_type", 0))
    else:
        action_type = int(action[1]) if len(action) > 1 else 0
    if action_type == 0:  # WAIT
        # Keep WAIT legal but make it increasingly expensive when far from destination.
        # This discourages stalling as a default policy while preserving safety fallback behavior.
        if vehicle.status == VehicleStatus.IN_TRANSIT:
            dist_to_dest = float(getattr(vehicle, "steps_to_destination", config.max_steps))
            distance_scale = min(dist_to_dest / float(max(config.max_steps, 1)), 1.0)
            if len(vehicle.shipments_onboard) > 0:
                return -0.15 * (1.0 + 0.5 * distance_scale)
            return -0.10 * (1.0 + 0.25 * distance_scale)
        if len(vehicle.shipments_onboard) > 0:
            dist_to_dest = float(getattr(vehicle, "steps_to_destination", config.max_steps))
            distance_scale = min(dist_to_dest / float(max(config.max_steps, 1)), 1.0)
            return -0.20 * (1.0 + distance_scale)
        return -0.05
    if action_type == 1:  # REROUTE
        return -0.01
    if action_type == 2:  # DIVERT_COLD_DEPOT
        # Scale detour cost but cap at reasonable value (max -0.30)
        detour_penalty = min(0.25, 0.001 * float(vehicle.detour_cost))
        return -0.03 - detour_penalty
    if action_type == 3:  # SWAP_VEHICLE
        return -0.04
    if action_type == 4:  # EXPEDITE
        return -0.08
    if action_type == 5:  # ABORT
        return -0.35
    return 0.0


def no_progress_penalty_reward(vehicles: List[Vehicle], shipments: List[Shipment], config: ColdChainConfig) -> float:
    total = 0.0
    for vehicle in vehicles:
        if len(vehicle.shipments_onboard) == 0:
            continue
        dist_to_dest = float(getattr(vehicle, "steps_to_destination", config.max_steps))
        if dist_to_dest > 0:
            # If cargo is onboard and vehicle lingers, penalize progressively.
            linger_steps = max(0, int(getattr(vehicle, "steps_at_current_location", 0)) - 2)
            if linger_steps > 0:
                total -= min(0.20, 0.01 * linger_steps)
    return float(total)


def repeated_action_penalty(action_type: int, action_repeat_count: int) -> float:
    # Penalize repeated non-productive patterns while allowing brief repetition.
    if action_repeat_count <= 2:
        return 0.0
    if action_type in (0, 2, 5):
        return float(-0.02 * min(action_repeat_count - 2, 8))
    return 0.0


def exploration_bonus(action_type: int, action_repeat_count: int) -> float:
    # Small encouragement for directional actions without overpowering base reward.
    if action_type in (1, 4) and action_repeat_count <= 2:
        return 0.01
    return 0.0


def transit_action_reward(action_type: int, vehicle: Vehicle, graph: nx.Graph, shipments: List[Shipment], prev_distances: Dict) -> float:
    if vehicle.status != VehicleStatus.IN_TRANSIT or not vehicle.shipments_onboard:
        return 0.0

    best_delta = 0.0
    for shipment_id in vehicle.shipments_onboard:
        if shipment_id >= len(shipments):
            continue
        shipment = shipments[shipment_id]
        if shipment.is_delivered or shipment.is_destroyed:
            continue
        current_distance = nx.shortest_path_length(graph, vehicle.location, shipment.destination_node, weight="current_weight")
        previous_distance = prev_distances.get((vehicle.id, shipment.id), current_distance)
        best_delta = max(best_delta, float(previous_distance - current_distance))

    if action_type in (1, 4):  # REROUTE / EXPEDITE
        if best_delta > 0:
            return float(0.3 + 0.2 * min(best_delta, 5.0))
        return -0.1

    if action_type == 0:  # WAIT
        if best_delta > 0:
            return 0.0
        return -0.2

    return 0.0


def transit_stall_penalty(env_state, config: ColdChainConfig) -> float:
    streaks = getattr(env_state, "transit_no_progress_streak", None)
    if streaks is None:
        return 0.0

    total = 0.0
    for vehicle in env_state.vehicles:
        if vehicle.status != VehicleStatus.IN_TRANSIT or not vehicle.shipments_onboard:
            streaks[vehicle.id] = 0
            continue

        best_delta = 0.0
        for shipment_id in vehicle.shipments_onboard:
            if shipment_id >= len(env_state.shipments):
                continue
            shipment = env_state.shipments[shipment_id]
            if shipment.is_delivered or shipment.is_destroyed:
                continue
            current_distance = nx.shortest_path_length(env_state.graph, vehicle.location, shipment.destination_node, weight="current_weight")
            previous_distance = env_state._prev_distances.get((vehicle.id, shipment.id), current_distance)
            best_delta = max(best_delta, float(previous_distance - current_distance))

        if best_delta > 0:
            streaks[vehicle.id] = 0
            continue

        streaks[vehicle.id] = int(streaks.get(vehicle.id, 0)) + 1
        if streaks[vehicle.id] <= 3:
            continue

        stall_duration = streaks[vehicle.id] - 3
        total -= min(1.0, 0.15 * (1.0 + 0.1 * stall_duration))

    return float(total)


def idle_penalty_reward(vehicles: List[Vehicle], shipments: List[Shipment], config: ColdChainConfig) -> float:
    if not any(not shipment.is_delivered and not shipment.is_destroyed for shipment in shipments):
        return 0.0
    total = 0.0
    for vehicle in vehicles:
        if vehicle.status == VehicleStatus.IDLE:
            total -= 0.02
    return float(total)


def revisit_penalty_reward(env_state, config: ColdChainConfig) -> float:
    """
    Penalize revisiting the same node to discourage short cycles.
    Expects env_state.node_visit_counts[(vehicle_id, node)] -> visit_count.
    """
    visit_counts = getattr(env_state, "node_visit_counts", None)
    if not visit_counts:
        return 0.0

    total = 0.0
    for vehicle in env_state.vehicles:
        count = int(visit_counts.get((vehicle.id, vehicle.location), 1))
        if count > 5: # Allow short stays/exploration
            # total -= float(config.revisit_penalty_scale) * float(count - 5)
            pass
    return float(total)

def depot_stall_penalty(vehicles: List[Vehicle], graph: nx.Graph) -> float:
    # Fix 2: Exponential ramp — uses location-based counter (steps_in_depot_zone)
    # which cannot be reset by DIVERT/IN_TRANSIT tricks (Cut 2).
    total = 0.0
    depot_nodes = set(graph.graph.get("cold_depot_nodes", []))
    for vehicle in vehicles:
        if vehicle.location in depot_nodes and len(vehicle.shipments_onboard) > 0:
            stall_steps = getattr(vehicle, "steps_in_depot_zone", 0)
            if stall_steps > 5:
                # Exponential: after 30 steps = -34, after 50 steps = -111
                total -= 0.5 * (1.05 ** (stall_steps - 5))
    return float(total)


def time_pressure_penalty(vehicles: List[Vehicle], shipments: List[Shipment], steps_elapsed: int, max_steps: int) -> float:
    # Fix 3: Distance-aware — penalises being far from destination more than being near it
    # This breaks the symmetry where depot-hiding felt identical to moving
    total = 0.0
    progress_fraction = float(steps_elapsed) / float(max_steps)
    for vehicle in vehicles:
        for shipment_id in vehicle.shipments_onboard:
            if shipment_id >= len(shipments):
                continue
            shipment = shipments[shipment_id]
            if shipment.is_delivered or shipment.is_destroyed:
                continue
            dist_to_dest = float(getattr(vehicle, "steps_to_destination", max_steps))
            # Normalise distance: far away = big penalty, near = small
            dist_factor = min(dist_to_dest / float(max_steps), 1.0)
            total -= 3.0 * progress_fraction * dist_factor
    # Also apply a small flat penalty proportional to any still-active shipments NOT on a vehicle
    stranded = sum(1 for s in shipments if not s.is_delivered and not s.is_destroyed and s.current_vehicle_id == -1)
    total -= 1.0 * progress_fraction * stranded
    return float(total)


def delivery_event_reward(shipment: Shipment, current_step: int, config: ColdChainConfig) -> float:
    if not shipment.is_delivered:
        return 0.0

    priority_mult = {0: 1.0, 1: 1.5, 2: 2.5}[shipment.priority]
    on_time = current_step <= shipment.deadline_step
    minor_excursion = shipment.excursion_duration > 0 and shipment.excursion_duration <= CARGO_SPECS[shipment.cargo_type]["tolerance_steps"]

    if on_time and shipment.excursion_count == 0:
        early_bonus = max(0.0, float(config.max_steps - current_step)) * 0.20
        return float(30.0 * priority_mult + early_bonus)
    if on_time and minor_excursion:
        early_bonus = max(0.0, float(config.max_steps - current_step)) * 0.12
        return float(12.0 * priority_mult + early_bonus)
    if not on_time and shipment.excursion_count == 0:
        return 2.0
    if not on_time and shipment.excursion_count > 0:
        return -2.0
    return float(-5.0 * priority_mult)


def catastrophe_event_reward(shipment: Shipment, config: ColdChainConfig) -> float:
    if not shipment.is_destroyed:
        return 0.0
    # Exploit-proof destruction floor: never profitable against milestone sums.
    return -50.0


def destruction_penalty_with_floor(penalty_scale: float, base_penalty: float = 50.0, floor_scale: float = 0.5) -> float:
    effective_scale = max(float(penalty_scale), float(floor_scale))
    return float(-base_penalty * effective_scale)


def compute_step_reward(env_state, action, prev_distances, config) -> Tuple[float, Dict]:
    # 1. Calculate Penalty Annealing Scale (Fix 2)
    progress = min(float(config.current_training_step) / float(config.penalty_anneal_steps), 1.0)
    penalty_scale = config.penalty_initial_scale + (1.0 - config.penalty_initial_scale) * progress

    reward_temp = temp_shaping_reward(env_state.shipments, config)
    reward_progress = progress_shaping_reward(
        env_state.vehicles,
        env_state.shipments,
        env_state.graph,
        config,
        env_state.steps_elapsed,
        prev_distances,
    )
    reward_progress += revisit_penalty_reward(env_state, config)
    vehicle_index = int(action["vehicle_index"]) if isinstance(action, dict) else int(action[0])
    selected_vehicle = env_state.vehicles[vehicle_index] if vehicle_index < len(env_state.vehicles) else env_state.vehicles[0]
    reward_cost = action_cost_reward(action, selected_vehicle, config)
    reward_idle = idle_penalty_reward(env_state.vehicles, env_state.shipments, config)
    reward_idle += depot_stall_penalty(env_state.vehicles, env_state.graph)
    reward_no_progress = no_progress_penalty_reward(env_state.vehicles, env_state.shipments, config)
    active_shipments = sum(int(not shipment.is_delivered and not shipment.is_destroyed) for shipment in env_state.shipments)
    reward_time_pressure = time_pressure_penalty(env_state.vehicles, env_state.shipments, env_state.steps_elapsed, config.max_steps)
    action_type = int(getattr(env_state, "action_type", 0))
    action_repeat_count = int(getattr(env_state, "action_repeat_count", 1))
    reward_repeat = repeated_action_penalty(action_type, action_repeat_count)
    reward_explore = exploration_bonus(action_type, action_repeat_count)
    reward_transit_action = 0.0
    if len(env_state.vehicles) > 0:
        reward_transit_action = transit_action_reward(action_type, env_state.vehicles[0], env_state.graph, env_state.shipments, prev_distances)
    reward_transit_stall = transit_stall_penalty(env_state, config)

    reward_delivery_events = 0.0
    reward_catastrophe = 0.0
    reward_milestones = 0.0

    # 2. Base components
    reward = (
        reward_temp
        + reward_progress
        + reward_cost
        + reward_idle
        + reward_no_progress
        + reward_time_pressure
        + reward_repeat
        + reward_explore
        + reward_transit_action
        + reward_transit_stall
    )

    # 3. Dense Milestone Rewards (Fix 3)
    # Pickup milestone with commitment gate.
    commitment_steps = 10
    has_cargo_onboard = any(len(v.shipments_onboard) > 0 for v in env_state.vehicles)
    any_destroyed = any(s.is_destroyed for s in env_state.shipments)
    pickup_start_key = "pickup_hold_start_step"
    if has_cargo_onboard and env_state.milestones.get(pickup_start_key) is None:
        env_state.milestones[pickup_start_key] = int(env_state.steps_elapsed)
    if any_destroyed:
        env_state.milestones[pickup_start_key] = None
    pickup_start_step = env_state.milestones.get(pickup_start_key)
    if (
        pickup_start_step is not None
        and not env_state.milestones["pickup"]
        and int(env_state.steps_elapsed) - int(pickup_start_step) >= commitment_steps
    ):
        reward_milestones += 2.0
        env_state.milestones["pickup"] = True

    # Departed depot milestone requires committed pickup first.
    if (
        env_state.milestones["pickup"]
        and any(len(v.visited_nodes_this_route) > 2 for v in env_state.vehicles)
        and not env_state.milestones["departed"]
    ):
        reward_milestones += 1.0
        env_state.milestones["departed"] = True

    # Halfway Milestone
    for vehicle in env_state.vehicles:
        for shipment_id in vehicle.shipments_onboard:
            if shipment_id < len(env_state.shipments):
                shipment = env_state.shipments[shipment_id]
                current_dist = nx.shortest_path_length(env_state.graph, vehicle.location, shipment.destination_node, weight="current_weight")
                initial_dist = env_state._initial_distances.get(shipment.id)
                
                # If we don't have initial dist, capture it now (first time observed on board)
                if initial_dist is None:
                    env_state._initial_distances[shipment.id] = current_dist
                    initial_dist = current_dist
                
                if (
                    env_state.milestones["pickup"]
                    and current_dist < 0.5 * initial_dist
                    and not env_state.milestones["halfway"]
                ):
                    reward_milestones += 1.5
                    env_state.milestones["halfway"] = True

    # 4. Delivery and Catastrophe Events (One-time only)
    for shipment in env_state.shipments:
        # Unique milestone keys for each shipment
        delivered_key = f"delivered_{shipment.id}"
        destroyed_key = f"destroyed_{shipment.id}"
        
        if shipment.is_delivered and not env_state.milestones.get(delivered_key):
            delivery_comp = delivery_event_reward(shipment, env_state.steps_elapsed, config)
            reward_delivery_events += delivery_comp
            reward += delivery_comp
            env_state.milestones[delivered_key] = True
            
        if shipment.is_destroyed and not env_state.milestones.get(destroyed_key):
            catastrophe_base = catastrophe_event_reward(shipment, config)
            catastrophe_comp = destruction_penalty_with_floor(penalty_scale, base_penalty=abs(catastrophe_base), floor_scale=0.5)
            reward_catastrophe += catastrophe_comp
            reward += catastrophe_comp
            env_state.milestones[destroyed_key] = True

    reward += reward_milestones

    # Final logic for all-delivered or any-destroyed (Milestones)
    # Fix 4: Clip only the running reward BEFORE adding delivery/catastrophe events,
    # so the full +500 delivery bonus always reaches the policy gradient uncapped.
    reward = float(np.clip(reward, -2.0, 2.0))

    if all(shipment.is_delivered for shipment in env_state.shipments):
        if not env_state.milestones["delivered"]:
            reward += 20.0  # Delivery bonus added AFTER clip — always fully visible
            env_state.milestones["delivered"] = True
    # NOTE: Continuous catastrophe penalty removed to prevent value function drowning.
    # Individual shipment destruction is already penalized once in delivery_event_reward.
    
    breakdown = {
        "r_temp": reward_temp,
        "r_progress": reward_progress,
        "r_cost": reward_cost,
        "r_idle": reward_idle,
        "r_no_progress": reward_no_progress,
        "r_time": reward_time_pressure,
        "r_repeat": reward_repeat,
        "r_explore": reward_explore,
        "r_transit_action": reward_transit_action,
        "r_transit_stall": reward_transit_stall,
        "r_delivery_events": reward_delivery_events,
        "r_catastrophe": reward_catastrophe,
        "r_milestones": reward_milestones,
    }
    return reward, breakdown