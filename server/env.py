from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from core.config import ColdChainConfig
from core.models import ColdChainAction
from server.environment import ColdChainEnvironment as OpenEnvColdChainEnvironment

class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.difficulty = 1
        self.success_count = 0
        self.successes_to_advance = 2
        self.max_difficulty = 3
        self.replay_prob = 0.0
        self.replay_min_difficulty = 1
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = dict(options or {})
        effective_difficulty = self.difficulty
        if self.difficulty > self.replay_min_difficulty and self.replay_prob > 0.0:
            if float(self._rng.random()) < self.replay_prob:
                low = max(1, int(self.replay_min_difficulty))
                high = max(low + 1, int(self.difficulty))
                effective_difficulty = int(self._rng.integers(low, high))

        options["curriculum_difficulty"] = effective_difficulty
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["curriculum_difficulty"] = int(effective_difficulty)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Progress curriculum only on successful terminal delivery outcomes.
        if terminated and bool(info.get("delivery_success", False)):
            self.success_count += 1
            print(f"DEBUG: Success count = {self.success_count}/{self.successes_to_advance} for Difficulty {self.difficulty}")
            if self.success_count >= self.successes_to_advance:
                if self.difficulty < self.max_difficulty:
                    old_diff = self.difficulty
                    self.difficulty += 1
                    self.success_count = 0
                    print(f"🎓 Curriculum graduated: difficulty {old_diff} -> {self.difficulty}")
                
        return obs, reward, terminated, truncated, info


class ColdChainEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, config: Optional[ColdChainConfig] = None, render_mode=None):
        super().__init__()
        self.config = config or ColdChainConfig()
        self.render_mode = render_mode
        self._fig = None
        self._ax = None
        self._core = OpenEnvColdChainEnvironment(self.config)
        self._define_spaces()

    def _define_spaces(self) -> None:
        cfg = self.config
        max_vehicle_features = 9 + cfg.max_cargo_per_vehicle

        self.observation_space = gym.spaces.Dict(
            {
                "global": gym.spaces.Box(
                    low=np.array([-40.0, 0.0, 1.0, 0.0, -10.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([60.0, 1.0, 4.0, 3.0, 30.0, float(cfg.max_steps), float(cfg.max_steps)], dtype=np.float32),
                    dtype=np.float32
                ),
                "vehicles": gym.spaces.Box(
                    low=-1.0,
                    high=float(max(cfg.n_nodes, cfg.max_steps, 9999)),
                    shape=(cfg.n_vehicles, max_vehicle_features),
                    dtype=np.float32,
                ),
                "shipments": gym.spaces.Box(
                    low=-1.0,
                    high=float(max(cfg.n_nodes, cfg.max_steps, 9999)),
                    shape=(cfg.max_shipments, 13),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Discrete((cfg.n_vehicles + 1) * 6 * cfg.n_nodes)

    @property
    def graph(self):
        return self._core.graph

    @property
    def vehicles(self):
        return self._core.vehicles

    @property
    def shipments(self):
        return self._core.shipments

    @property
    def weather(self):
        return self._core.weather

    @property
    def steps_elapsed(self):
        return self._core.steps_elapsed

    @property
    def state(self):
        return self._core.state

    @property
    def _illegal_action_count(self):
        return self._core._illegal_action_count

    @property
    def _last_action_was_masked(self):
        return self._core._last_action_was_masked

    @property
    def _last_info(self):
        return self._core._last_info

    @_last_info.setter
    def _last_info(self, value):
        self._core._last_info = value

    @property
    def _trajectory(self):
        return self._core._trajectory

    @_trajectory.setter
    def _trajectory(self, value):
        self._core._trajectory = value

    def __getattr__(self, name: str):
        return getattr(self._core, name)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self._core.reset(seed=seed, **(options or {}))
        info = dict(self._core._get_info())
        return self._convert_observation(observation), info

    def step(self, action):
        a = int(action)
        
        # Decode action
        n_nodes = self.config.n_nodes
        vehicle_index = a // (6 * n_nodes)
        action_type = (a % (6 * n_nodes)) // n_nodes
        target_index = a % n_nodes

        action_model = ColdChainAction(
            vehicle_index=vehicle_index,
            action_type=action_type,
            target_index=target_index,
        )
        # Increment global training step for annealing (Fix 2 in Debug.md)
        self.config.current_training_step += 1
        
        observation = self._core.step(action_model)
        info = dict(self._core._last_info)
        converted_obs = self._convert_observation(observation)
        reward = float(observation.reward or 0.0)
        terminated = bool(self._core.state.terminated)
        truncated = bool(self._core.state.truncated)

        action_array = np.array([vehicle_index, action_type, target_index], dtype=np.int64)
        self._core._trajectory[-1] = (converted_obs, action_array, reward, dict(info))
        return converted_obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            self._render_human()
        return None

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def action_masks(self) -> np.ndarray:
        return self._core.action_masks()

    def _convert_observation(self, observation) -> Dict[str, Any]:
        payload = observation.model_dump() if hasattr(observation, "model_dump") else dict(observation)
        global_state = payload.get("global_state", {})
        vehicles = payload.get("vehicles", [])
        shipments = payload.get("shipments", [])

        converted_global = np.array([
            global_state.get("ambient_temperature", 0.0),
            global_state.get("time_of_day", 0.0),
            global_state.get("traffic_multiplier", 1.0),
            float(global_state.get("weather_event", 0)),
            global_state.get("hub_cold_storage_temp", 4.0),
            float(global_state.get("steps_elapsed", 0)),
            float(global_state.get("steps_remaining", 0))
        ], dtype=np.float32)

        vehicle_rows = []
        for vehicle in vehicles[: self.config.n_vehicles]:
            cargo_slots = list(vehicle.get("shipments_onboard", []))[: self.config.max_cargo_per_vehicle]
            cargo_slots.extend([-1] * (self.config.max_cargo_per_vehicle - len(cargo_slots)))
            vehicle_rows.append(
                [
                    float(vehicle.get("location", 0)),
                    float(vehicle.get("status", 0)),
                    float(vehicle.get("refrigeration_status", 0)),
                    float(vehicle.get("fuel_level", 0.0)),
                    float(vehicle.get("steps_until_next_waypoint", 0)),
                    *[float(slot) for slot in cargo_slots],
                    float(vehicle.get("nearest_cold_depot_node", -1)),
                    float(vehicle.get("steps_to_cold_depot", 9999)),
                    float(vehicle.get("steps_to_destination", 9999)),
                    float(vehicle.get("detour_cost_to_depot", 9999)),
                ]
            )
        while len(vehicle_rows) < self.config.n_vehicles:
            vehicle_rows.append([0.0] * (9 + self.config.max_cargo_per_vehicle))

        shipment_rows = []
        for shipment in shipments[: self.config.max_shipments]:
            shipment_rows.append(
                [
                    float(shipment.get("cargo_temp", 0.0)),
                    float(shipment.get("temp_lower_bound", 0.0)),
                    float(shipment.get("temp_upper_bound", 0.0)),
                    float(shipment.get("time_to_deadline", 0)),
                    float(shipment.get("current_vehicle", -1)),
                    float(shipment.get("destination_node", 0)),
                    float(shipment.get("priority", 0)),
                    float(shipment.get("cargo_type", 0)),
                    float(shipment.get("excursion_count", 0)),
                    float(shipment.get("excursion_duration", 0)),
                    float(shipment.get("is_destroyed", 0)),
                    float(shipment.get("is_delivered", 0)),
                    float(shipment.get("steps_since_last_reading", 0)),
                ]
            )
        while len(shipment_rows) < self.config.max_shipments:
            shipment_rows.append([0.0] * 13)

        return {
            "global": converted_global,
            "vehicles": np.asarray(vehicle_rows, dtype=np.float32),
            "shipments": np.asarray(shipment_rows, dtype=np.float32),
        }

    def _render_ansi(self) -> str:
        lines = [f"ColdChainEnv step={self.steps_elapsed} weather={int(self.weather.current_event)}"]
        for vehicle in self.vehicles:
            lines.append(
                f"Vehicle {vehicle.id}: loc={vehicle.location} status={int(vehicle.status)} refrig={int(vehicle.refrig_status)} cargo={vehicle.shipments_onboard}"
            )
        for shipment in self.shipments:
            lines.append(
                f"Shipment {shipment.id}: temp={shipment.cargo_temp:.2f} dest={shipment.destination_node} delivered={shipment.is_delivered} destroyed={shipment.is_destroyed}"
            )
        return "\n".join(lines)

    def _render_human(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(10, 8))

        assert self.graph is not None
        self._ax.clear()
        positions = nx.spring_layout(self.graph, seed=self.config.graph_seed)
        cold_depots = set(self.graph.graph.get("cold_depot_nodes", []))

        node_colors = []
        for node in self.graph.nodes:
            if node == 0:
                node_colors.append("#d97706")
            elif node in cold_depots:
                node_colors.append("#2563eb")
            else:
                node_colors.append("#64748b")

        nx.draw(self.graph, positions, ax=self._ax, with_labels=True, node_color=node_colors, node_size=700, font_size=8)

        for vehicle in self.vehicles:
            x, y = positions[vehicle.location]
            self._ax.scatter([x], [y], s=180, c="#16a34a", marker="o", edgecolors="black", zorder=5)
            self._ax.text(x, y + 0.06, f"V{vehicle.id}", ha="center", fontsize=8, color="black")

        self._ax.set_title(f"ColdChain-Gym | step {self.steps_elapsed}")
        self._fig.tight_layout()
        plt.pause(0.001)
