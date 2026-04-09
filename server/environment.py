from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from uuid import uuid4

import networkx as nx
import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from core.action_mask import compute_action_mask, flatten_action
from core.city_graph import build_city_graph, detour_cost, nearest_cold_depot, refresh_edge_weights
from core.config import ColdChainConfig
from core.graders import CompositeGrader, DeliverySuccessGrader, EfficiencyGrader, ThermalIntegrityGrader
from core.models import ColdChainAction, ColdChainObservation, ColdChainState, GlobalTelemetry, ShipmentTelemetry, VehicleTelemetry
from core.reward import compute_step_reward, partial_delivery_terminal_reward
from core.shipment import CARGO_SPECS, Shipment, update_temperature
from core.vehicle import RefrigStatus, Vehicle, VehicleStatus, get_hold_temperature, maybe_breakdown, maybe_degrade_refrigeration, step_dock_timer
from core.weather import WeatherEvent, WeatherSystem


class ColdChainEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: Optional[ColdChainConfig] = None):
        self.config = config or ColdChainConfig()
        self._state = ColdChainState(episode_id=str(uuid4()), step_count=0, steps_remaining=self.config.max_steps)
        self.graph: nx.Graph | None = None
        self.vehicles: list[Vehicle] = []
        self.shipments: list[Shipment] = []
        self.weather = WeatherSystem()
        self.steps_elapsed = 0
        self._episode_done = False
        self._illegal_action_count = 0
        self._last_action_was_masked = False
        self._last_action_error: str | None = None
        self._last_info: Dict[str, Any] = {}
        self._prev_distances: Dict[tuple[int, int], int] = {}
        self._node_visit_counts: Dict[tuple[int, int], int] = {}
        self._last_known_temps: Dict[int, float] = {}
        self._transit_no_progress_streak: Dict[int, int] = {}
        self._trajectory: list[tuple[Any, Any, float, Dict[str, Any]]] = []
        self.milestones = {"pickup": False, "departed": False, "halfway": False, "delivered": False}
        self._initial_distances: Dict[int, float] = {}
        self._last_action_type: Optional[int] = None
        self._same_action_streak: int = 0
        self.difficulty = 1
        self.active_vehicle_count = self.config.n_vehicles
        self.active_shipment_count = self.config.n_shipments
        self._forced_weather_schedule: Dict[int, tuple[WeatherEvent, int]] = {}
        self._forced_weather_only: bool = False
        self._forced_breakdown_schedule: Dict[int, List[int]] = {}

    def reset(self, seed: Optional[int] = None, episode_id=None, **kwargs) -> ColdChainObservation:
        self._rng = np.random.default_rng(seed if seed is not None else self.config.graph_seed)
        self.graph = build_city_graph(self.config, self._rng)
        self.weather = WeatherSystem()
        self.steps_elapsed = 0
        self._episode_done = False
        self._illegal_action_count = 0
        self._last_action_was_masked = False
        self._last_action_error = None
        self._last_info = {}
        self._prev_distances = {}
        self._node_visit_counts = {}
        self._last_known_temps = {}
        self._transit_no_progress_streak = {}
        self._trajectory = []
        self.milestones = {"pickup": False, "departed": False, "halfway": False, "delivered": False}
        self._initial_distances = {}
        self._last_action_type = None
        self._same_action_streak = 0
        self.active_vehicle_count = self.config.n_vehicles
        self.active_shipment_count = self.config.n_shipments
        self._forced_weather_schedule = {}
        self._forced_weather_only = bool(kwargs.get("forced_weather_only", False))
        self._forced_breakdown_schedule = {}

        self.vehicles = [Vehicle(id=index, location=0) for index in range(self.config.n_vehicles)]
        self.shipments = self._build_shipments()

        self._initialize_vehicle_state()

        vehicle_start_nodes = kwargs.get("vehicle_start_nodes") or kwargs.get("vehicle_start_locations")
        shipment_destinations = kwargs.get("shipment_destinations")
        shipment_cargo_temps = kwargs.get("shipment_cargo_temps")
        shipment_cargo_types = kwargs.get("shipment_cargo_types")
        shipment_priorities = kwargs.get("shipment_priorities")
        shipment_deadlines = kwargs.get("shipment_deadlines")
        shipment_assignments = kwargs.get("shipment_assignments")
        active_vehicle_count = kwargs.get("active_vehicle_count")
        active_shipment_count = kwargs.get("active_shipment_count")
        forced_weather_events = kwargs.get("forced_weather_events")
        forced_breakdowns = kwargs.get("forced_breakdowns")
        load_shipments_on_start = bool(kwargs.get("load_shipments_on_start", False))
        custom_layout = any(
            value is not None
            for value in (
                vehicle_start_nodes,
                shipment_destinations,
                shipment_cargo_temps,
                shipment_cargo_types,
                shipment_priorities,
                shipment_deadlines,
                shipment_assignments,
            )
        )

        if forced_weather_events:
            for event in list(forced_weather_events):
                step = int(event.get("step", 0))
                duration = int(event.get("duration", 1))
                event_name = str(event.get("event", "NONE")).upper()
                if event_name in WeatherEvent.__members__:
                    self._forced_weather_schedule[step] = (WeatherEvent[event_name], duration)

        if forced_breakdowns:
            for item in list(forced_breakdowns):
                step = int(item.get("step", 0))
                vehicle_ids = [int(vehicle_id) for vehicle_id in item.get("vehicle_ids", [])]
                self._forced_breakdown_schedule.setdefault(step, []).extend(vehicle_ids)

        if vehicle_start_nodes is not None:
            for index, node in enumerate(list(vehicle_start_nodes)[: len(self.vehicles)]):
                self.vehicles[index].location = int(node)
                self.vehicles[index].visited_nodes_this_route = {int(node)}

        if shipment_destinations is not None:
            for index, node in enumerate(list(shipment_destinations)[: len(self.shipments)]):
                self.shipments[index].destination_node = int(node)

        if shipment_cargo_temps is not None:
            for index, temp in enumerate(list(shipment_cargo_temps)[: len(self.shipments)]):
                self.shipments[index].cargo_temp = float(temp)

        if shipment_cargo_types is not None:
            for index, cargo_type in enumerate(list(shipment_cargo_types)[: len(self.shipments)]):
                if str(cargo_type) in CARGO_SPECS:
                    bounds = CARGO_SPECS[str(cargo_type)]
                    shipment = self.shipments[index]
                    shipment.cargo_type = str(cargo_type)
                    shipment.temp_lower_bound = float(bounds["low"])
                    shipment.temp_upper_bound = float(bounds["high"])

        if shipment_priorities is not None:
            for index, priority in enumerate(list(shipment_priorities)[: len(self.shipments)]):
                self.shipments[index].priority = int(priority)

        if shipment_deadlines is not None:
            for index, deadline in enumerate(list(shipment_deadlines)[: len(self.shipments)]):
                self.shipments[index].deadline_step = int(deadline)

        for shipment in self.shipments:
            shipment.current_vehicle_id = -1

        for vehicle in self.vehicles:
            vehicle.shipments_onboard = []

        if shipment_assignments is not None:
            for shipment_index, vehicle_index in enumerate(list(shipment_assignments)[: len(self.shipments)]):
                if 0 <= int(vehicle_index) < len(self.vehicles):
                    vehicle = self.vehicles[int(vehicle_index)]
                    shipment = self.shipments[shipment_index]
                    if len(vehicle.shipments_onboard) < self.config.max_cargo_per_vehicle:
                        vehicle.shipments_onboard.append(shipment.id)
                        shipment.current_vehicle_id = vehicle.id
        elif load_shipments_on_start:
            for shipment_index, shipment in enumerate(self.shipments):
                if shipment_index < len(self.vehicles):
                    vehicle = self.vehicles[shipment_index]
                    if len(vehicle.shipments_onboard) < self.config.max_cargo_per_vehicle:
                        vehicle.shipments_onboard.append(shipment.id)
                        shipment.current_vehicle_id = vehicle.id

        self._apply_activity_profile(active_vehicle_count=active_vehicle_count, active_shipment_count=active_shipment_count)

        # Curriculum Learning: Teleport vehicle near destination (Fix 1 in Debug.md)
        self.difficulty = kwargs.get("curriculum_difficulty", 3)
        difficulty = self.difficulty
        if difficulty < 3 and not custom_layout:
            active_shipments = self._active_shipments()
            if active_shipments and self.vehicles and self.active_vehicle_count > 0:
                shipment = active_shipments[0]
                vehicle = self.vehicles[0]
                
                # Pre-load shipment onto vehicle
                shipment.current_vehicle_id = vehicle.id
                vehicle.shipments_onboard = [shipment.id]
                
                dest = shipment.destination_node
                if difficulty == 1:
                    # 1 hop away
                    neighbors = [n for n in self.graph.neighbors(dest) if n != 0]
                    if not neighbors: # Fallback if only 0 is neighbor
                        neighbors = list(self.graph.neighbors(dest))
                    if neighbors:
                        vehicle.location = int(neighbors[0])
                        # Initialize distances for reward function (Fix 3)
                        dist = int(nx.shortest_path_length(self.graph, vehicle.location, dest, weight="current_weight"))
                        self._initial_distances[shipment.id] = dist
                        self._prev_distances[(vehicle.id, shipment.id)] = dist
                elif difficulty == 2:
                    # 2 hops away
                    paths = nx.single_source_shortest_path_length(self.graph, dest, cutoff=2)
                    level_2_nodes = [node for node, dist in paths.items() if dist == 2]
                    if level_2_nodes:
                        vehicle.location = int(level_2_nodes[0])
                        
        self._assert_node_types()
        self._refresh_vehicle_metrics()
        self._cache_visible_temperatures()
        self._update_visit_counts()
        self._state = ColdChainState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            steps_remaining=self.config.max_steps,
            illegal_action_count=0,
            terminated=False,
            truncated=False,
        )

        observation = self._get_obs(reward=0.0, done=False, message="ColdChain episode reset", info=self._get_info())
        self._last_info = self._get_info()
        return observation

    def _apply_activity_profile(self, active_vehicle_count: Optional[int], active_shipment_count: Optional[int]) -> None:
        requested_vehicle_count = self.config.n_vehicles if active_vehicle_count is None else int(active_vehicle_count)
        requested_shipment_count = self.config.n_shipments if active_shipment_count is None else int(active_shipment_count)
        self.active_vehicle_count = max(1, min(self.config.n_vehicles, requested_vehicle_count))
        self.active_shipment_count = max(1, min(self.config.n_shipments, requested_shipment_count))

        for shipment in self.shipments:
            shipment.is_active = shipment.id < self.active_shipment_count
            if not shipment.is_active:
                shipment.current_vehicle_id = -1
                shipment.is_destroyed = False
                shipment.is_delivered = True
                shipment.excursion_count = 0
                shipment.excursion_duration = 0
                shipment.steps_since_last_reading = 0

        for vehicle in self.vehicles:
            if vehicle.id >= self.active_vehicle_count:
                vehicle.status = VehicleStatus.BROKEN
                vehicle.route = []
                vehicle.shipments_onboard = []
                vehicle.steps_until_next_waypoint = 0

        for vehicle in self.vehicles:
            vehicle.shipments_onboard = [
                shipment_id
                for shipment_id in vehicle.shipments_onboard
                if shipment_id < len(self.shipments) and self.shipments[shipment_id].is_active
            ]

    def _active_shipments(self) -> list[Shipment]:
        return [shipment for shipment in self.shipments if shipment.is_active]

    def step(self, action: ColdChainAction, timeout_s=None, **kwargs) -> ColdChainObservation:  # type: ignore[override]
        if self.graph is None:
            raise RuntimeError("Environment must be reset before stepping")

        action_payload = action.model_dump() if hasattr(action, "model_dump") else dict(action)
        prev_locations = tuple(vehicle.location for vehicle in self.vehicles)
        mask = self.action_masks()
        flat_index = flatten_action(action_payload, self.config)
        info: Dict[str, Any] = {"action_was_masked": False, "last_action_error": None}

        if flat_index >= len(mask) or mask[flat_index] == 0:
            error_message = "Illegal action received; forcing WAIT"
            warnings.warn(error_message, RuntimeWarning)
            self._illegal_action_count += 1
            self._last_action_was_masked = True
            self._last_action_error = error_message
            info["action_was_masked"] = True
            info["last_action_error"] = error_message
            action_payload = {"vehicle_index": int(action_payload["vehicle_index"]), "action_type": 0, "target_index": 0}
        else:
            self._last_action_was_masked = False
            self._last_action_error = None

        current_action_type = int(action_payload.get("action_type", 0))
        if self._last_action_type is None or current_action_type != self._last_action_type:
            self._same_action_streak = 1
        else:
            self._same_action_streak += 1
        self._last_action_type = current_action_type

        if self._forced_weather_only:
            if self.weather.event_duration_remaining > 0:
                self.weather.event_duration_remaining -= 1
                if self.weather.event_duration_remaining == 0:
                    self.weather.current_event = WeatherEvent.NONE
        else:
            self.weather.step(self._rng, self.config)

        forced_weather = self._forced_weather_schedule.get(self.steps_elapsed)
        if forced_weather is not None:
            event_type, duration = forced_weather
            self.weather.current_event = event_type
            self.weather.event_duration_remaining = max(0, int(duration))
        refresh_edge_weights(self.graph, self.weather.traffic_multiplier)
        self._execute_action(action_payload)
        self._advance_vehicles()
        self._assert_node_types()
        self._update_visit_counts()
        self._handle_automatic_pickups()
        self._update_shipments()
        self._refresh_vehicle_metrics()

        env_state = SimpleNamespace(
            shipments=self._active_shipments(),
            vehicles=self.vehicles,
            graph=self.graph,
            steps_elapsed=self.steps_elapsed,
            difficulty=self.difficulty,
            active_shipment_count=self.active_shipment_count,
            active_vehicle_count=self.active_vehicle_count,
            _prev_distances=self._prev_distances,
            node_visit_counts=self._node_visit_counts,
            milestones=self.milestones,
            _initial_distances=self._initial_distances,
            transit_no_progress_streak=self._transit_no_progress_streak,
            action_type=current_action_type,
            action_repeat_count=self._same_action_streak,
        )
        reward, breakdown = compute_step_reward(env_state, action_payload, self._prev_distances, self.config)
        info["reward_breakdown"] = breakdown
        info["action_mask"] = mask.tolist()

        self.steps_elapsed += 1
        self._state.step_count = self.steps_elapsed
        self._state.steps_remaining = max(0, self.config.max_steps - self.steps_elapsed)
        self._state.illegal_action_count = self._illegal_action_count

        all_delivered = self._all_shipments_delivered()
        all_destroyed = self._all_shipments_destroyed()
        terminated = all_delivered
        truncated = (self.steps_elapsed >= self.config.max_steps) or all_destroyed
        self._episode_done = terminated or truncated
        self._state.terminated = terminated
        self._state.truncated = truncated

        if all_destroyed and not all_delivered:
            # Hard-stop failure mode: do not allow "destroy fast" to be a viable shortcut.
            reward += -30.0 * float(len(self._active_shipments()))
            info["all_destroyed_terminal_penalty"] = -30.0 * float(len(self._active_shipments()))

        # Terminal non-delivery penalty — scaled to avoid drowning dense learning signal.
        if truncated and not terminated:
            undelivered = [s for s in self._active_shipments() if not s.is_delivered and not s.is_destroyed]
            if undelivered:
                # Terminal penalties never anneal: timeout must remain a hard failure.
                terminal_penalty = -5.0 * len(undelivered)
                reward += terminal_penalty
                info["terminal_penalty"] = terminal_penalty
                if self.active_shipment_count > 1:
                    partial_bonus = partial_delivery_terminal_reward(self.shipments)
                    reward += partial_bonus
                    info["partial_delivery_terminal_bonus"] = float(partial_bonus)

        self._update_visible_temperature_cache()
        curr_locations = tuple(vehicle.location for vehicle in self.vehicles)
        info.update(self._get_info())
        observation = self._get_obs(reward=reward, done=self._episode_done, message="step complete", info=info)
        if self.config.debug_step_trace:
            print(
                f"step={self.steps_elapsed} action={action_payload} "
                f"loc={prev_locations}->{curr_locations} reward={float(reward):.4f} done={self._episode_done}"
            )
        self._trajectory.append((observation.model_dump(), action_payload, float(reward), dict(info)))
        self._last_info = info
        return observation

    @property
    def state(self) -> ColdChainState:
        return self._state

    def action_masks(self) -> np.ndarray:
        if self.graph is None:
            total = (self.config.n_vehicles + 1) * 6 * self.config.n_nodes
            return np.zeros(total, dtype=np.int8)
        return compute_action_mask(self.vehicles, self.shipments, self.graph, self.config, difficulty=self.difficulty)

    def _build_shipments(self) -> list[Shipment]:
        shipments: list[Shipment] = []
        cargo_types = list(CARGO_SPECS.keys())
        for shipment_id in range(self.config.n_shipments):
            cargo_type = str(self._rng.choice(cargo_types))
            bounds = CARGO_SPECS[cargo_type]
            destination_node = int(self._rng.integers(1, self.config.n_nodes))
            priority = int(self._rng.integers(0, 3))
            deadline_step = int(self._rng.integers(self.config.max_steps // 3, self.config.max_steps))
            shipments.append(
                Shipment(
                    id=shipment_id,
                    cargo_type=cargo_type,
                    cargo_temp=(bounds["low"] + bounds["high"]) / 2.0,
                    temp_lower_bound=bounds["low"],
                    temp_upper_bound=bounds["high"],
                    destination_node=destination_node,
                    deadline_step=deadline_step,
                    priority=priority,
                    active_since_step=0,
                )
            )
        return shipments

    def _initialize_vehicle_state(self) -> None:
        for vehicle in self.vehicles:
            vehicle.location = 0
            vehicle.status = VehicleStatus.IDLE
            vehicle.refrig_status = RefrigStatus.WORKING
            vehicle.fuel_level = 1.0
            vehicle.steps_until_next_waypoint = 0
            vehicle.route = []
            vehicle.shipments_onboard = []
            vehicle.visited_nodes_this_route = {0}
            vehicle.hold_temp = 4.0
            vehicle.dock_steps_remaining = 0
            vehicle.steps_at_current_location = 0
            vehicle.steps_in_depot_zone = 0  # Cut 2: location-based depot stall counter

    def _refresh_vehicle_metrics(self) -> None:
        assert self.graph is not None
        depot_nodes = list(self.graph.graph.get("cold_depot_nodes", []))
        for vehicle in self.vehicles:
            nearest_depot, steps_to_depot = nearest_cold_depot(self.graph, vehicle.location, depot_nodes)
            vehicle.nearest_cold_depot = nearest_depot
            vehicle.steps_to_cold_depot = steps_to_depot
            vehicle.steps_to_destination = 9999
            vehicle.detour_cost = 9999
            if vehicle.shipments_onboard:
                shipment = self.shipments[vehicle.shipments_onboard[0]]
                vehicle.steps_to_destination = int(nx.shortest_path_length(self.graph, vehicle.location, shipment.destination_node, weight="current_weight"))
                vehicle.detour_cost = detour_cost(self.graph, vehicle.location, nearest_depot, shipment.destination_node)

        for vehicle in self.vehicles:
            for shipment_id in vehicle.shipments_onboard:
                self._prev_distances[(vehicle.id, shipment_id)] = vehicle.steps_to_destination

    def _advance_vehicles(self) -> None:
        assert self.graph is not None
        depot_nodes = list(self.graph.graph.get("cold_depot_nodes", []))
        forced_breakdowns = set(self._forced_breakdown_schedule.get(self.steps_elapsed, []))
        for vehicle in self.vehicles:
            maybe_degrade_refrigeration(vehicle, self._rng, self.config)
            if vehicle.id in forced_breakdowns:
                vehicle.status = VehicleStatus.BROKEN
            maybe_breakdown(vehicle, self._rng, self.config)

            if vehicle.status == VehicleStatus.BROKEN:
                continue

            if vehicle.dock_steps_remaining > 0:
                step_dock_timer(vehicle)
                continue

            if vehicle.status == VehicleStatus.IN_TRANSIT:
                if vehicle.steps_until_next_waypoint > 0:
                    vehicle.steps_until_next_waypoint -= 1
                    vehicle.steps_at_current_location = 0
                    if vehicle.steps_until_next_waypoint > 0:
                        continue

                if vehicle.steps_until_next_waypoint <= 0 and vehicle.route:
                    vehicle.location = int(vehicle.route.pop(0))
                    vehicle.visited_nodes_this_route.add(vehicle.location)
                    vehicle.steps_at_current_location = 0
                    if vehicle.route:
                        next_node = int(vehicle.route[0])
                        vehicle.steps_until_next_waypoint = int(
                            nx.shortest_path_length(self.graph, vehicle.location, next_node, weight="current_weight")
                        )
                        vehicle.status = VehicleStatus.IN_TRANSIT
                    elif vehicle.location in depot_nodes:
                        vehicle.status = VehicleStatus.AT_STOP
                        vehicle.dock_steps_remaining = 2
                        vehicle.hold_temp = 4.0
                    else:
                        vehicle.status = VehicleStatus.IDLE
            else:
                # IDLE or AT_STOP or BROKEN
                vehicle.steps_at_current_location += 1

            # Cut 2: location-based depot zone counter — unaffected by IN_TRANSIT status.
            # Resets ONLY when vehicle physically arrives at a non-depot node.
            if vehicle.location in depot_nodes:
                vehicle.steps_in_depot_zone = getattr(vehicle, "steps_in_depot_zone", 0) + 1
            else:
                vehicle.steps_in_depot_zone = 0

    def _execute_action(self, action_payload: Dict[str, Any]) -> None:
        vehicle_index = int(action_payload["vehicle_index"])
        action_type = int(action_payload["action_type"])
        target_index = int(action_payload.get("target_index", 0))

        if vehicle_index == self.config.n_vehicles or vehicle_index >= len(self.vehicles):
            return

        vehicle = self.vehicles[vehicle_index]
        if vehicle.status == VehicleStatus.BROKEN:
            return

        if action_type == 0:
            return
        if action_type == 1:
            if target_index != vehicle.location:
                path = nx.shortest_path(self.graph, vehicle.location, target_index, weight="current_weight")
                vehicle.route = [int(node) for node in path[1:]]
                if vehicle.route:
                    vehicle.steps_until_next_waypoint = int(
                        nx.shortest_path_length(self.graph, vehicle.location, vehicle.route[0], weight="current_weight")
                    )
                vehicle.status = VehicleStatus.IN_TRANSIT
                vehicle.visited_nodes_this_route = {vehicle.location}
                vehicle.fuel_level = max(0.0, vehicle.fuel_level - 0.01)
        elif action_type == 2:
            if target_index in self.graph.graph.get("cold_depot_nodes", []):
                path = nx.shortest_path(self.graph, vehicle.location, target_index, weight="current_weight")
                vehicle.route = [int(node) for node in path[1:]]
                if vehicle.route:
                    vehicle.steps_until_next_waypoint = int(
                        nx.shortest_path_length(self.graph, vehicle.location, vehicle.route[0], weight="current_weight")
                    )
                vehicle.status = VehicleStatus.IN_TRANSIT
                vehicle.visited_nodes_this_route = {vehicle.location}
                vehicle.fuel_level = max(0.0, vehicle.fuel_level - 0.03)
        elif action_type == 3:
            if target_index < len(self.vehicles):
                other_vehicle = self.vehicles[target_index]
                if other_vehicle.id != vehicle.id and other_vehicle.location == vehicle.location:
                    transferable = list(vehicle.shipments_onboard)
                    available = self.config.max_cargo_per_vehicle - len(other_vehicle.shipments_onboard)
                    if available > 0:
                        moved = transferable[:available]
                        for shipment_id in moved:
                            vehicle.shipments_onboard.remove(shipment_id)
                            other_vehicle.shipments_onboard.append(shipment_id)
                            self.shipments[shipment_id].current_vehicle_id = other_vehicle.id
                    vehicle.fuel_level = max(0.0, vehicle.fuel_level - 0.05)
        elif action_type == 4:
            if vehicle.status == VehicleStatus.IN_TRANSIT and vehicle.steps_until_next_waypoint > 0 and vehicle.fuel_level >= 0.1:
                vehicle.steps_until_next_waypoint = max(0, vehicle.steps_until_next_waypoint - 1)
                vehicle.fuel_level = max(0.0, vehicle.fuel_level - 0.1)
        elif action_type == 5:
            if vehicle.shipments_onboard:
                for shipment_id in list(vehicle.shipments_onboard):
                    self.shipments[shipment_id].current_vehicle_id = -1
                    vehicle.shipments_onboard.remove(shipment_id)
                path = nx.shortest_path(self.graph, vehicle.location, 0, weight="current_weight")
                vehicle.route = [int(node) for node in path[1:]]
                if vehicle.route:
                    vehicle.steps_until_next_waypoint = int(
                        nx.shortest_path_length(self.graph, vehicle.location, vehicle.route[0], weight="current_weight")
                    )
                vehicle.status = VehicleStatus.IN_TRANSIT
                vehicle.visited_nodes_this_route = {vehicle.location}
                vehicle.fuel_level = max(0.0, vehicle.fuel_level - 0.2)
    def _handle_automatic_pickups(self) -> None:
        """Automatically pick up shipments at the Hub (Node 0) if vehicle has space."""
        for vehicle in self.vehicles:
            if vehicle.location == 0 and len(vehicle.shipments_onboard) < self.config.max_cargo_per_vehicle:
                for shipment in self.shipments:
                    if shipment.is_active and shipment.current_vehicle_id == -1 and not shipment.is_delivered and not shipment.is_destroyed:
                        # Pickup!
                        shipment.current_vehicle_id = vehicle.id
                        vehicle.shipments_onboard.append(shipment.id)
                        if len(vehicle.shipments_onboard) >= self.config.max_cargo_per_vehicle:
                            break

    def _update_shipments(self) -> None:
        hub_temp = 4.0 if self.weather.current_event.name != "POWER_OUTAGE" else self.weather.outdoor_temperature
        for shipment in self.shipments:
            if not shipment.is_active:
                continue
            if shipment.is_destroyed or shipment.is_delivered:
                continue

            vehicle = self.vehicles[shipment.current_vehicle_id] if 0 <= shipment.current_vehicle_id < len(self.vehicles) else None
            if vehicle is None:
                update_temperature(shipment, hub_temp, self.config.step_duration_hours)
            else:
                ambient_hold_temp = get_hold_temperature(vehicle, self.weather.outdoor_temperature)
                update_temperature(shipment, ambient_hold_temp, self.config.step_duration_hours)

            if shipment.is_destroyed:
                continue

            if self.steps_elapsed >= shipment.deadline_step and not shipment.is_delivered:
                shipment.is_destroyed = True

            if vehicle is not None and vehicle.location == shipment.destination_node and shipment.current_vehicle_id == vehicle.id:
                shipment.is_delivered = True
                if shipment.id in vehicle.shipments_onboard:
                    vehicle.shipments_onboard.remove(shipment.id)
                shipment.current_vehicle_id = vehicle.id

            if shipment.steps_since_last_reading == 0:
                self._last_known_temps[shipment.id] = shipment.cargo_temp

    def _cache_visible_temperatures(self) -> None:
        for shipment in self.shipments:
            if shipment.is_active:
                self._last_known_temps[shipment.id] = shipment.cargo_temp

    def _update_visit_counts(self) -> None:
        for vehicle in self.vehicles:
            key = (vehicle.id, int(vehicle.location))
            self._node_visit_counts[key] = self._node_visit_counts.get(key, 0) + 1

    def _assert_node_types(self) -> None:
        for vehicle in self.vehicles:
            if not isinstance(vehicle.location, (int, np.integer)):
                raise TypeError(f"Vehicle {vehicle.id} location must be int, got {type(vehicle.location).__name__}")
        for shipment in self.shipments:
            if not isinstance(shipment.destination_node, (int, np.integer)):
                raise TypeError(
                    f"Shipment {shipment.id} destination_node must be int, got {type(shipment.destination_node).__name__}"
                )

    def _update_visible_temperature_cache(self) -> None:
        for shipment in self.shipments:
            if shipment.is_active and shipment.steps_since_last_reading == 0:
                self._last_known_temps[shipment.id] = shipment.cargo_temp

    def _get_obs(self, reward: float, done: bool, message: str, info: Optional[Dict[str, Any]] = None) -> ColdChainObservation:
        global_obs = GlobalTelemetry(
            ambient_temperature=float(self.weather.outdoor_temperature),
            time_of_day=float((self.steps_elapsed % 96) / 96.0),
            traffic_multiplier=float(self.weather.traffic_multiplier),
            weather_event=int(self.weather.current_event),
            hub_cold_storage_temp=float(4.0 if self.weather.current_event.name != "POWER_OUTAGE" else self.weather.outdoor_temperature),
            steps_elapsed=self.steps_elapsed,
            steps_remaining=max(0, self.config.max_steps - self.steps_elapsed),
            difficulty_level_norm=float(self.difficulty) / 5.0,
            active_shipments_norm=float(self.active_shipment_count) / float(max(1, self.config.max_shipments)),
            active_vehicles_norm=float(self.active_vehicle_count) / float(max(1, self.config.n_vehicles)),
        )

        vehicle_rows: list[VehicleTelemetry] = []
        for vehicle in sorted(self.vehicles, key=lambda item: item.id):
            cargo_slots = [-1] * self.config.max_cargo_per_vehicle
            for slot_index, shipment_id in enumerate(vehicle.shipments_onboard[: self.config.max_cargo_per_vehicle]):
                cargo_slots[slot_index] = shipment_id
            vehicle_rows.append(
                VehicleTelemetry(
                    id=vehicle.id,
                    location=vehicle.location,
                    status=int(vehicle.status),
                    refrigeration_status=int(vehicle.refrig_status),
                    fuel_level=float(vehicle.fuel_level),
                    steps_until_next_waypoint=vehicle.steps_until_next_waypoint,
                    shipments_onboard=cargo_slots,
                    nearest_cold_depot_node=vehicle.nearest_cold_depot,
                    steps_to_cold_depot=vehicle.steps_to_cold_depot,
                    steps_to_destination=vehicle.steps_to_destination,
                    detour_cost_to_depot=vehicle.detour_cost,
                    hold_temp=float(vehicle.hold_temp),
                    dock_steps_remaining=vehicle.dock_steps_remaining,
                )
            )

        shipment_rows: list[ShipmentTelemetry] = []
        for shipment in sorted(self.shipments, key=lambda item: item.id):
            shipment_rows.append(
                ShipmentTelemetry(
                    id=shipment.id,
                    cargo_temp=float(self._last_known_temps.get(shipment.id, shipment.cargo_temp) if shipment.steps_since_last_reading != 0 else shipment.cargo_temp),
                    temp_lower_bound=float(shipment.temp_lower_bound),
                    temp_upper_bound=float(shipment.temp_upper_bound),
                    time_to_deadline=max(0, shipment.deadline_step - self.steps_elapsed),
                    current_vehicle=shipment.current_vehicle_id,
                    destination_node=shipment.destination_node,
                    priority=shipment.priority,
                    cargo_type={"vaccine": 0, "insulin": 1, "blood": 2, "organ": 3}[shipment.cargo_type],
                    excursion_count=shipment.excursion_count,
                    excursion_duration=shipment.excursion_duration,
                    is_destroyed=int(shipment.is_destroyed),
                    is_delivered=int(shipment.is_delivered),
                    steps_since_last_reading=shipment.steps_since_last_reading,
                )
            )

        action_mask = self.action_masks().tolist()
        info = info or self._get_info()
        return ColdChainObservation(
            done=done,
            reward=reward,
            global_state=global_obs,
            vehicles=vehicle_rows,
            shipments=shipment_rows,
            action_mask=action_mask,
            action_was_masked=self._last_action_was_masked,
            illegal_action_count=self._illegal_action_count,
            last_action_error=info.get("last_action_error"),
            reward_breakdown=info.get("reward_breakdown", {}),
            episode_summary=info.get("episode_summary", {}),
            grader_scores=info.get("grader_scores", {}),
            message=message,
        )

    def _get_info(self):
        active_shipments = self._active_shipments()
        all_delivered = all(shipment.is_delivered for shipment in active_shipments)
        all_destroyed = all(shipment.is_destroyed for shipment in active_shipments)
        any_delivered = any(shipment.is_delivered for shipment in active_shipments)
        termination_reason = "in_progress"
        if self._state.terminated:
            if all_delivered:
                termination_reason = "all_delivered"
            elif self._all_shipments_complete():
                termination_reason = "mixed_complete"
            else:
                termination_reason = "terminated_unknown"
        elif self._state.truncated:
            if all_destroyed:
                termination_reason = "all_destroyed"
            else:
                termination_reason = "max_steps"

        info = {
            "action_was_masked": self._last_action_was_masked,
            "illegal_action_count": self._illegal_action_count,
            "last_action_error": self._last_action_error,
            "last_action_type": self._last_action_type,
            "same_action_streak": self._same_action_streak,
            "transit_no_progress_streak": dict(self._transit_no_progress_streak),
            "delivery_success": bool(all_delivered),
            "delivery_any": bool(any_delivered),
            "active_vehicle_count": int(self.active_vehicle_count),
            "active_shipment_count": int(self.active_shipment_count),
            "termination_reason": termination_reason,
            "per_shipment_status": {shipment.id: dataclasses.asdict(shipment) for shipment in self.shipments},
            "per_vehicle_status": {vehicle.id: dataclasses.asdict(vehicle) for vehicle in self.vehicles},
        }
        if self._episode_done:
            info["grader_scores"] = self._compute_grader_scores()
            info["episode_summary"] = self._build_episode_summary()
        return info

    def _compute_grader_scores(self) -> Dict[str, float]:
        trajectory = self._trajectory
        if not trajectory:
            return {"delivery": 0.0, "thermal": 0.0, "efficiency": 0.0, "composite": 0.0}
        delivery = DeliverySuccessGrader(trajectory).score()
        thermal = ThermalIntegrityGrader(trajectory).score()
        efficiency = EfficiencyGrader(trajectory).score()
        composite = CompositeGrader(trajectory).score()
        return {"delivery": delivery, "thermal": thermal, "efficiency": efficiency, "composite": composite}

    def _all_shipments_complete(self) -> bool:
        return all(shipment.is_delivered or shipment.is_destroyed for shipment in self._active_shipments())

    def _all_shipments_delivered(self) -> bool:
        return all(shipment.is_delivered for shipment in self._active_shipments())

    def _all_shipments_destroyed(self) -> bool:
        return all(shipment.is_destroyed for shipment in self._active_shipments())

    def _build_episode_summary(self) -> Dict[str, Any]:
        active_shipments = self._active_shipments()
        return {
            "steps_elapsed": self.steps_elapsed,
            "shipments_delivered": sum(int(shipment.is_delivered) for shipment in active_shipments),
            "shipments_destroyed": sum(int(shipment.is_destroyed) for shipment in active_shipments),
            "illegal_actions": self._illegal_action_count,
        }
