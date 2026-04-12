from __future__ import annotations

import os
from dataclasses import replace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from core.config import ColdChainConfig
from server.env import ColdChainEnv
from graders import BasicGrader
from core.vehicle import VehicleStatus
from examples.phase2_stabilization import CFG as BASE_CFG


VEHICLE_SPECS = [
    {"vehicle_index": 0, "start_node": 7, "target_node": 16},
    {"vehicle_index": 1, "start_node": 19, "target_node": 14},
]


def draw_multi_vehicle_path_png(env: ColdChainEnv, paths: list[list[int]], specs: list[dict], output_path: str, title: str) -> None:
    if env.graph is None:
        return

    graph = env.graph
    positions = nx.spring_layout(graph, seed=7)
    path_colors = ["#f58518", "#4c78a8", "#54a24b", "#b279a2"]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(graph, positions, edge_color="#d0d7de", width=1.0, alpha=0.7)
    nx.draw_networkx_nodes(graph, positions, node_color="#e8eef3", node_size=420, edgecolors="#7b8794", linewidths=0.8)

    cold_depots = list(graph.graph.get("cold_depot_nodes", []))
    if cold_depots:
        nx.draw_networkx_nodes(graph, positions, nodelist=cold_depots, node_color="#4c78a8", node_size=520, edgecolors="#1f3d5a", linewidths=1.0)

    for idx, spec in enumerate(specs):
        start_node = int(spec["start_node"])
        target_node = int(spec["target_node"])
        color = path_colors[idx % len(path_colors)]
        nx.draw_networkx_nodes(graph, positions, nodelist=[start_node], node_color="#e45756", node_size=620, edgecolors="#7f1d1d", linewidths=1.0)
        nx.draw_networkx_nodes(graph, positions, nodelist=[target_node], node_color="#f3a712", node_size=620, edgecolors="#7a4e00", linewidths=1.0)

        traversed_edges = []
        for left, right in zip(paths[idx], paths[idx][1:]):
            if left != right and graph.has_edge(int(left), int(right)):
                traversed_edges.append((int(left), int(right)))
        if traversed_edges:
            nx.draw_networkx_edges(graph, positions, edgelist=traversed_edges, edge_color=color, width=3.0, alpha=0.95)

    traversed_nodes = sorted(set(node for path in paths for node in path))
    nx.draw_networkx_nodes(graph, positions, nodelist=traversed_nodes, node_color="#54a24b", node_size=500, edgecolors="#1f5f24", linewidths=1.0)
    nx.draw_networkx_labels(graph, positions, font_size=8, font_color="#1f2933")

    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def prepare_two_vehicle_episode(
    env: ColdChainEnv,
    seed: int,
    vehicle_specs: list[dict],
    cargo_temp: float = 3.0,
) -> None:
    env.reset(seed=seed)

    for vehicle in env.vehicles:
        vehicle.status = VehicleStatus.IDLE
        vehicle.steps_until_next_waypoint = 0
        vehicle.route = []
        vehicle.shipments_onboard = []
        vehicle.visited_nodes_this_route = {vehicle.location}
        vehicle.fuel_level = 1.0
        vehicle.dock_steps_remaining = 0

    for idx, spec in enumerate(vehicle_specs):
        vehicle = env.vehicles[spec["vehicle_index"]]
        shipment = env.shipments[idx]

        shipment.destination_node = int(spec["target_node"])
        shipment.cargo_temp = float(cargo_temp)
        shipment.current_vehicle_id = vehicle.id
        shipment.is_destroyed = False
        shipment.is_delivered = False
        shipment.deadline_step = env.config.max_steps - 1
        shipment.in_excursion = False
        shipment.excursion_count = 0
        shipment.excursion_duration = 0
        shipment.steps_since_last_reading = 0

        vehicle.location = int(spec["start_node"])
        vehicle.shipments_onboard = [shipment.id]
        vehicle.status = VehicleStatus.IDLE
        vehicle.steps_until_next_waypoint = 0
        vehicle.route = []
        vehicle.visited_nodes_this_route = {int(spec["start_node"])}

    env._refresh_vehicle_metrics()
    env._cache_visible_temperatures()


def choose_vehicle_action(env: ColdChainEnv, vehicle_index: int) -> np.ndarray:
    vehicle = env.vehicles[vehicle_index]
    if not vehicle.shipments_onboard:
        return np.array([vehicle_index, 0, 0], dtype=np.int64)

    shipment = env.shipments[vehicle.shipments_onboard[0]]
    hard_temp = shipment.temp_lower_bound + env.config.temp_integrity_hard_margin

    if shipment.cargo_temp <= hard_temp and not shipment.is_delivered and not shipment.is_destroyed:
        nearest_depot = int(vehicle.nearest_cold_depot)
        if nearest_depot >= 0:
            return np.array([vehicle_index, 2, nearest_depot], dtype=np.int64)
        return np.array([vehicle_index, 5, 0], dtype=np.int64)

    if vehicle.location == shipment.destination_node:
        return np.array([vehicle_index, 0, 0], dtype=np.int64)

    if vehicle.status == VehicleStatus.IN_TRANSIT and vehicle.steps_until_next_waypoint > 0:
        return np.array([vehicle_index, 4, 0], dtype=np.int64)

    shortest_path = nx.dijkstra_path(env.graph, vehicle.location, shipment.destination_node, weight="current_weight")
    next_hop = int(shortest_path[1]) if len(shortest_path) > 1 else int(shipment.destination_node)
    return np.array([vehicle_index, 1, next_hop], dtype=np.int64)


def select_controlled_vehicle(env: ColdChainEnv, vehicle_specs: list[dict], step: int) -> int | None:
    active_indices: list[int] = []
    for spec in vehicle_specs:
        vehicle_index = int(spec["vehicle_index"])
        vehicle = env.vehicles[vehicle_index]
        if not vehicle.shipments_onboard:
            continue
        shipment = env.shipments[vehicle.shipments_onboard[0]]
        if shipment.is_delivered or shipment.is_destroyed:
            continue
        active_indices.append(vehicle_index)

    if not active_indices:
        return None
    active_indices.sort()
    return active_indices[step % len(active_indices)]


def run_two_vehicle_episode(cfg: ColdChainConfig, seed: int, vehicle_specs: list[dict], max_steps: int = 40):
    env = ColdChainEnv(config=cfg)
    prepare_two_vehicle_episode(env, seed=seed, vehicle_specs=vehicle_specs)

    paths = {spec["vehicle_index"]: [int(spec["start_node"])] for spec in vehicle_specs}
    trajectory = []
    actions = []
    done = False
    step = 0

    while not done and step < max_steps:
        controlled_index = select_controlled_vehicle(env, vehicle_specs, step)
        if controlled_index is None:
            break

        action = choose_vehicle_action(env, controlled_index)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((obs, action, float(reward), dict(info)))
        actions.append((int(action[0]), int(action[1]), int(action[2])))

        for spec in vehicle_specs:
            vehicle_index = int(spec["vehicle_index"])
            paths[vehicle_index].append(int(env.vehicles[vehicle_index].location))

        done = bool(terminated or truncated)
        step += 1

    compact_paths = {}
    for spec in vehicle_specs:
        vehicle_index = int(spec["vehicle_index"])
        path = paths[vehicle_index]
        compact = [path[0]]
        for node in path[1:]:
            if node != compact[-1]:
                compact.append(node)
        compact_paths[vehicle_index] = compact

    score = BasicGrader(trajectory).score()
    result = {
        "seed": seed,
        "paths": paths,
        "compact_paths": compact_paths,
        "actions": actions,
        "basic_score": score,
        "delivered": [bool(env.shipments[idx].is_delivered) for idx in range(len(vehicle_specs))],
        "destroyed": [bool(env.shipments[idx].is_destroyed) for idx in range(len(vehicle_specs))],
        "cargo_temps": [float(env.shipments[idx].cargo_temp) for idx in range(len(vehicle_specs))],
        "targets": [int(spec["target_node"]) for spec in vehicle_specs],
    }
    env.close()
    return result


def main() -> int:
    cfg = replace(BASE_CFG, n_vehicles=2, n_shipments=2, max_cargo_per_vehicle=1, temp_integrity_hard_margin=1.0, temp_integrity_borderline_margin=3.0)
    seed = 0

    result = run_two_vehicle_episode(cfg, seed=seed, vehicle_specs=VEHICLE_SPECS)

    graph_env = ColdChainEnv(config=cfg)
    graph_env.reset(seed=seed)
    output_root = "outputs/grader_results/basic"
    draw_multi_vehicle_path_png(
        graph_env,
        [result["compact_paths"][0], result["compact_paths"][1]],
        VEHICLE_SPECS,
        f"{output_root}/two_vehicle_model_paths.png",
        "Two-vehicle shortest paths",
    )
    graph_env.close()

    os.makedirs(output_root, exist_ok=True)
    report_path = f"{output_root}/two_vehicle_basic_grader.txt"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("Two-vehicle BasicGrader Demo\n")
        handle.write("===========================\n\n")
        handle.write(f"Seed used: {seed}\n")
        for spec in VEHICLE_SPECS:
            idx = int(spec["vehicle_index"])
            handle.write(f"Vehicle {idx}: start {spec['start_node']} -> target {spec['target_node']}\n")
        handle.write(f"BasicGrader: {result['basic_score']:.4f}\n\n")
        for spec in VEHICLE_SPECS:
            idx = int(spec["vehicle_index"])
            handle.write(f"Vehicle {idx} path: {' -> '.join(map(str, result['paths'][idx]))}\n")
            handle.write(f"Vehicle {idx} compact path: {' -> '.join(map(str, result['compact_paths'][idx]))}\n")
            handle.write(f"Delivered: {result['delivered'][idx]} | Destroyed: {result['destroyed'][idx]} | Final temp: {result['cargo_temps'][idx]:.3f}\n\n")

    print("=== TWO-VEHICLE BASIC GRADER DEMO ===")
    print(f"Seed used: {seed}")
    for spec in VEHICLE_SPECS:
        idx = int(spec["vehicle_index"])
        print(f"Vehicle {idx}: start {spec['start_node']} -> target {spec['target_node']}")
        print(f"Vehicle {idx} path: {' -> '.join(map(str, result['paths'][idx]))}")
        print(f"Vehicle {idx} compact path: {' -> '.join(map(str, result['compact_paths'][idx]))}")
        print(f"Vehicle {idx} delivered: {result['delivered'][idx]} | destroyed: {result['destroyed'][idx]} | final temp: {result['cargo_temps'][idx]:.3f}")
    print(f"BasicGrader: {result['basic_score']:.4f}")
    print(f"Saved graph PNG: {output_root}/two_vehicle_model_paths.png")
    print(f"Saved text report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
