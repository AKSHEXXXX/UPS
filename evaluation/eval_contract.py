from __future__ import annotations

from typing import Any, Dict, List
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.city_graph import nearest_cold_depot
from core.config import ColdChainConfig
from server.env import ColdChainEnv


def get_mask(env):
    return env.unwrapped.action_masks()


def decode_flat_action(flat_action: int, n_nodes: int) -> Dict[str, int]:
    action_value = int(flat_action)
    vehicle_index = action_value // (6 * n_nodes)
    action_type = (action_value % (6 * n_nodes)) // n_nodes
    target_index = action_value % n_nodes
    return {
        "vehicle_index": vehicle_index,
        "action_type": action_type,
        "target_index": target_index,
    }


def build_eval_env(config: ColdChainConfig):
    env = ColdChainEnv(config=config)
    env = ActionMasker(env, get_mask)
    env = gym.wrappers.FlattenObservation(env)
    return env


def _heuristic_action(raw_env) -> int:
    graph = raw_env.graph
    if graph is None:
        return 0

    n_nodes = raw_env.config.n_nodes
    depot_nodes = list(graph.graph.get("cold_depot_nodes", []))

    for vehicle in raw_env.vehicles:
        if not vehicle.shipments_onboard:
            continue

        shipment = raw_env.shipments[vehicle.shipments_onboard[0]]
        if shipment.is_delivered or shipment.is_destroyed:
            continue

        if vehicle.location == shipment.destination_node:
            return vehicle.id * (6 * n_nodes)

        target_node = shipment.destination_node
        action_type = 1
        if shipment.cargo_temp > shipment.temp_upper_bound and depot_nodes:
            target_node, _ = nearest_cold_depot(graph, vehicle.location, depot_nodes)
            action_type = 2

        return vehicle.id * (6 * n_nodes) + action_type * n_nodes + int(target_node)

    return 0


def run_eval_episode(
    model,
    env,
    *,
    seed: int,
    deterministic: bool,
    curriculum_difficulty: int,
    evaluation_training_step: int,
    reset_options: Dict[str, Any] | None = None,
    trace_every: int = 0,
    collect_trajectory: bool = False,
) -> Dict[str, Any]:
    raw_env = env.unwrapped
    raw_env.config.current_training_step = int(evaluation_training_step)
    options = {"curriculum_difficulty": int(curriculum_difficulty)}
    if reset_options:
        options.update(reset_options)
    obs, _ = env.reset(seed=seed, options=options)
    raw_env.config.current_training_step = int(evaluation_training_step)
    graph_snapshot = raw_env.graph.copy() if raw_env.graph is not None else None

    total_reward = 0.0
    done = False
    action_type_counts = {k: 0 for k in range(6)}
    terminal = "unknown"
    delivered_any = False
    delivery_success = False
    destroyed_any = False
    trajectory: List[Any] = []
    first_legal_actions = int(np.count_nonzero(raw_env.action_masks()))

    while not done:
        mask = raw_env.action_masks()
        use_model = model is not None and getattr(model, "action_space", None) == getattr(env, "action_space", None)
        if use_model:
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            action_int = int(action)
        else:
            action_int = int(_heuristic_action(raw_env))
            if action_int >= len(mask) or mask[action_int] == 0:
                legal_actions = np.flatnonzero(mask)
                action_int = int(legal_actions[0]) if len(legal_actions) else 0
        action_parts = decode_flat_action(action_int, raw_env.config.n_nodes)
        action_type_counts[action_parts["action_type"]] += 1

        next_obs, reward, terminated, truncated, info = env.step(action_int)
        if collect_trajectory:
            trajectory.append((next_obs, action_int, float(reward), dict(info)))

        if trace_every > 0 and (raw_env.steps_elapsed % trace_every == 0 or terminated or truncated):
            print(
                f"Step {raw_env.steps_elapsed:3}: reward={float(reward):7.2f}, "
                f"action=[V{action_parts['vehicle_index']} A{action_parts['action_type']} T{action_parts['target_index']}], "
                f"legal={int(np.count_nonzero(mask))}"
            )

        total_reward += float(reward)
        delivered_any = delivered_any or any(sh.is_delivered for sh in raw_env.shipments)
        delivery_success = bool(info.get("delivery_success", False))
        destroyed_any = destroyed_any or any(sh.is_destroyed for sh in raw_env.shipments)
        obs = next_obs
        done = bool(terminated or truncated)
        if terminated:
            terminal = "terminated"
        elif truncated:
            terminal = "truncated"

    return {
        "seed": int(seed),
        "difficulty": int(raw_env.difficulty),
        "evaluation_training_step": int(evaluation_training_step),
        "deterministic": bool(deterministic),
        "steps": int(raw_env.steps_elapsed),
        "total_reward": float(total_reward),
        "delivered_any": bool(delivered_any),
        "delivery_success": bool(delivery_success),
        "destroyed_any": bool(destroyed_any),
        "terminal": terminal,
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "initial_legal_actions": int(first_legal_actions),
        "action_type_counts": action_type_counts,
        "trajectory": trajectory,
        "graph_snapshot": graph_snapshot,
    }