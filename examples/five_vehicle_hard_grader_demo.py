from __future__ import annotations

import os
from dataclasses import replace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from core.config import ColdChainConfig
from server.env import ColdChainEnv
from core.graders import HardGrader
from core.graders import HardEmergencyCaseGrader
from core.shipment import CARGO_SPECS
from core.vehicle import RefrigStatus, VehicleStatus
from examples.phase2_stabilization import CFG as BASE_CFG


ACTION_NAMES = {
    0: "WAIT",
    1: "REROUTE",
    2: "DIVERT_DEPOT",
    3: "SWAP",
    4: "EXPEDITE",
    5: "ABORT",
}


VEHICLE_SPECS = [
    {
        "vehicle_index": 0,
        "start_node": 4,
        "target_node": 18,
        "cargo_temp": 7.30,
        "hard_margin": 0.60,
        "borderline_margin": 0.15,
    },
    {
        "vehicle_index": 1,
        "start_node": 7,
        "target_node": 16,
        "cargo_temp": 6.60,
        "hard_margin": 0.70,
        "borderline_margin": 0.30,
    },
    {
        "vehicle_index": 2,
        "start_node": 13,
        "target_node": 24,
        "cargo_temp": 5.80,
        "hard_margin": 0.70,
        "borderline_margin": 0.25,
    },
    {
        "vehicle_index": 3,
        "start_node": 19,
        "target_node": 28,
        "cargo_temp": 5.00,
        "hard_margin": 0.80,
        "borderline_margin": 0.35,
    },
    {
        "vehicle_index": 4,
        "start_node": 25,
        "target_node": 2,
        "cargo_temp": 4.40,
        "hard_margin": 0.90,
        "borderline_margin": 2.70,
    },
]


def draw_multi_vehicle_path_png(env: ColdChainEnv, paths: list[list[int]], specs: list[dict], output_path: str, title: str) -> None:
    if env.graph is None:
        return

    graph = env.graph
    positions = nx.spring_layout(graph, seed=7)
    path_colors = ["#f58518", "#4c78a8", "#54a24b", "#b279a2", "#e45756"]

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_edges(graph, positions, edge_color="#d0d7de", width=1.0, alpha=0.7)
    nx.draw_networkx_nodes(graph, positions, node_color="#e8eef3", node_size=360, edgecolors="#7b8794", linewidths=0.8)

    cold_depots = list(graph.graph.get("cold_depot_nodes", []))
    if cold_depots:
        nx.draw_networkx_nodes(graph, positions, nodelist=cold_depots, node_color="#4c78a8", node_size=500, edgecolors="#1f3d5a", linewidths=1.0)

    for idx, spec in enumerate(specs):
        start_node = int(spec["start_node"])
        target_node = int(spec["target_node"])
        color = path_colors[idx % len(path_colors)]
        nx.draw_networkx_nodes(graph, positions, nodelist=[start_node], node_color="#e45756", node_size=600, edgecolors="#7f1d1d", linewidths=1.0)
        nx.draw_networkx_nodes(graph, positions, nodelist=[target_node], node_color="#f3a712", node_size=600, edgecolors="#7a4e00", linewidths=1.0)

        traversed_edges = []
        for left, right in zip(paths[idx], paths[idx][1:]):
            if left != right and graph.has_edge(int(left), int(right)):
                traversed_edges.append((int(left), int(right)))
        if traversed_edges:
            nx.draw_networkx_edges(graph, positions, edgelist=traversed_edges, edge_color=color, width=3.0, alpha=0.95)

    traversed_nodes = sorted(set(node for path in paths for node in path))
    nx.draw_networkx_nodes(graph, positions, nodelist=traversed_nodes, node_color="#54a24b", node_size=470, edgecolors="#1f5f24", linewidths=1.0)
    nx.draw_networkx_labels(graph, positions, font_size=7, font_color="#1f2933")

    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def prepare_five_vehicle_episode(env: ColdChainEnv, seed: int, vehicle_specs: list[dict]) -> None:
    env.reset(seed=seed)

    for vehicle in env.vehicles:
        vehicle.status = VehicleStatus.IDLE
        vehicle.refrig_status = RefrigStatus.WORKING
        vehicle.steps_until_next_waypoint = 0
        vehicle.route = []
        vehicle.shipments_onboard = []
        vehicle.visited_nodes_this_route = {vehicle.location}
        vehicle.fuel_level = 1.0
        vehicle.dock_steps_remaining = 0

    for idx, spec in enumerate(vehicle_specs):
        vehicle = env.vehicles[spec["vehicle_index"]]
        shipment = env.shipments[idx]

        shipment.cargo_type = "vaccine"
        bounds = CARGO_SPECS[shipment.cargo_type]
        shipment.temp_lower_bound = float(bounds["low"])
        shipment.temp_upper_bound = float(bounds["high"])
        shipment.destination_node = int(spec["target_node"])
        shipment.cargo_temp = float(spec["cargo_temp"])
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


def choose_vehicle_action(
    env: ColdChainEnv,
    vehicle_index: int,
    spec_by_vehicle: dict[int, dict],
    detour_completed: dict[int, bool],
) -> np.ndarray:
    vehicle = env.vehicles[vehicle_index]
    spec = spec_by_vehicle[vehicle_index]

    if not vehicle.shipments_onboard:
        return np.array([vehicle_index, 0, 0], dtype=np.int64)

    shipment = env.shipments[vehicle.shipments_onboard[0]]
    hard_temp = shipment.temp_lower_bound + float(spec["hard_margin"])
    borderline_temp = shipment.temp_lower_bound + float(spec["borderline_margin"])
    cold_depots = set(env.graph.graph.get("cold_depot_nodes", [])) if env.graph is not None else set()

    if vehicle.location in cold_depots and detour_completed.get(vehicle_index, False):
        if vehicle.location == shipment.destination_node:
            return np.array([vehicle_index, 0, 0], dtype=np.int64)

        if vehicle.status == VehicleStatus.IN_TRANSIT and vehicle.steps_until_next_waypoint > 0:
            return np.array([vehicle_index, 4, 0], dtype=np.int64)

        shortest_path = nx.dijkstra_path(env.graph, vehicle.location, shipment.destination_node, weight="current_weight")
        next_hop = int(shortest_path[1]) if len(shortest_path) > 1 else int(shipment.destination_node)
        return np.array([vehicle_index, 1, next_hop], dtype=np.int64)

    if shipment.cargo_temp <= hard_temp and not shipment.is_delivered and not shipment.is_destroyed:
        nearest_depot = int(vehicle.nearest_cold_depot)
        if nearest_depot >= 0:
            return np.array([vehicle_index, 2, nearest_depot], dtype=np.int64)
        return np.array([vehicle_index, 5, 0], dtype=np.int64)

    if shipment.cargo_temp <= borderline_temp and not shipment.is_delivered and not shipment.is_destroyed:
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


def run_five_vehicle_episode(cfg: ColdChainConfig, seed: int, vehicle_specs: list[dict], max_steps: int = 80):
    env = ColdChainEnv(config=cfg)
    prepare_five_vehicle_episode(env, seed=seed, vehicle_specs=vehicle_specs)

    spec_by_vehicle = {int(spec["vehicle_index"]): spec for spec in vehicle_specs}
    detour_completed = {spec["vehicle_index"]: False for spec in vehicle_specs}
    paths = {spec["vehicle_index"]: [int(spec["start_node"])] for spec in vehicle_specs}
    trajectory = []
    actions = []
    delivery_events = {idx: None for idx in range(len(vehicle_specs))}
    borderline_events = {idx: None for idx in range(len(vehicle_specs))}
    done = False
    step = 0

    while not done and step < max_steps:
        controlled_index = select_controlled_vehicle(env, vehicle_specs, step)
        if controlled_index is None:
            break

        before_temps = {idx: float(env.shipments[idx].cargo_temp) for idx in range(len(vehicle_specs))}
        action = choose_vehicle_action(env, controlled_index, spec_by_vehicle, detour_completed)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((obs, action, float(reward), dict(info)))
        actions.append((int(action[0]), int(action[1]), int(action[2])))

        for idx, spec in enumerate(vehicle_specs):
            shipment = env.shipments[idx]
            vehicle_index = int(spec["vehicle_index"])
            border = shipment.temp_lower_bound + float(spec["borderline_margin"])
            if borderline_events[idx] is None and before_temps[idx] <= border and not shipment.is_delivered and not shipment.is_destroyed:
                borderline_events[idx] = {
                    "step": step,
                    "temp_before": before_temps[idx],
                    "action": ACTION_NAMES.get(int(action[1]), str(int(action[1]))) if int(action[0]) == idx else "NONE",
                    "location": int(env.vehicles[vehicle_index].location),
                }

            if delivery_events[idx] is None and shipment.is_delivered:
                delivery_events[idx] = {
                    "step": step,
                    "temp_before_step": before_temps[idx],
                    "temp_after_delivery": float(shipment.cargo_temp),
                    "location": int(env.vehicles[vehicle_index].location),
                }

            if vehicle_index == 4 and env.vehicles[vehicle_index].location in env.graph.graph.get("cold_depot_nodes", []):
                detour_completed[vehicle_index] = True

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

    score = HardGrader(trajectory).score()
    result = {
        "seed": seed,
        "paths": paths,
        "compact_paths": compact_paths,
        "actions": actions,
        "hard_score": score,
        "trajectory": trajectory,
        "delivery_events": delivery_events,
        "borderline_events": borderline_events,
        "delivered": [bool(env.shipments[idx].is_delivered) for idx in range(len(vehicle_specs))],
        "destroyed": [bool(env.shipments[idx].is_destroyed) for idx in range(len(vehicle_specs))],
        "cargo_temps": [float(env.shipments[idx].cargo_temp) for idx in range(len(vehicle_specs))],
        "targets": [int(spec["target_node"]) for spec in vehicle_specs],
    }
    env.close()
    return result


def main() -> int:
    cfg = replace(BASE_CFG, n_nodes=30, n_vehicles=5, n_shipments=5, max_cargo_per_vehicle=1, max_steps=120)
    seed = 0

    result = run_five_vehicle_episode(cfg, seed=seed, vehicle_specs=VEHICLE_SPECS)

    graph_env = ColdChainEnv(config=cfg)
    graph_env.reset(seed=seed)
    output_root = "outputs/grader_results/hard"
    draw_multi_vehicle_path_png(
        graph_env,
        [result["compact_paths"][spec["vehicle_index"]] for spec in VEHICLE_SPECS],
        VEHICLE_SPECS,
        f"{output_root}/five_vehicle_hard_model_paths.png",
        "Five-vehicle shortest paths (HardGrader)",
    )
    graph_env.close()

    os.makedirs(output_root, exist_ok=True)
    emergency_output_root = f"{output_root}/emergency"
    os.makedirs(emergency_output_root, exist_ok=True)
    report_path = f"{output_root}/five_vehicle_hard_grader.txt"
    emergency_report_path = f"{emergency_output_root}/five_vehicle_hard_emergency.txt"
    emergency_score = HardEmergencyCaseGrader(result["trajectory"]).score() if "trajectory" in result else 0.0
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("Five-vehicle HardGrader Demo\n")
        handle.write("===========================\n\n")
        handle.write(f"Seed used: {seed}\n")
        handle.write("Graph size: 30 nodes\n\n")
        for spec in VEHICLE_SPECS:
            idx = int(spec["vehicle_index"])
            hard_margin = float(spec["hard_margin"])
            borderline_margin = float(spec["borderline_margin"])
            handle.write(
                f"Vehicle {idx}: start {spec['start_node']} -> target {spec['target_node']} | "
                f"cargo_temp={float(spec['cargo_temp']):.2f} | hard_margin={hard_margin:.2f} | borderline_margin={borderline_margin:.2f}\n"
            )
        handle.write(f"HardGrader: {result['hard_score']:.4f}\n\n")

        for spec in VEHICLE_SPECS:
            idx = int(spec["vehicle_index"])
            handle.write(f"Vehicle {idx} path: {' -> '.join(map(str, result['paths'][idx]))}\n")
            handle.write(f"Vehicle {idx} compact path: {' -> '.join(map(str, result['compact_paths'][idx]))}\n")
            handle.write(f"Vehicle {idx} actions: {[ (a,b,c) for a,b,c in result['actions'] if a == idx ]}\n")
            borderline_event = result["borderline_events"][idx]
            if borderline_event is not None:
                handle.write(
                    f"Vehicle {idx} borderline trigger: step {borderline_event['step']} | temp before step {borderline_event['temp_before']:.3f} | action {borderline_event['action']} | location {borderline_event['location']}\n"
                )
            delivery_event = result["delivery_events"][idx]
            if delivery_event is not None:
                handle.write(
                    f"Vehicle {idx} delivery step: {delivery_event['step']} | temp before step {delivery_event['temp_before_step']:.3f} | temp after delivery {delivery_event['temp_after_delivery']:.3f}\n"
                )
            handle.write(
                f"Delivered: {result['delivered'][idx]} | Destroyed: {result['destroyed'][idx]} | Final temp: {result['cargo_temps'][idx]:.3f}\n\n"
            )

    with open(emergency_report_path, "w", encoding="utf-8") as handle:
        handle.write("Five-vehicle Hard Emergency Case\n")
        handle.write("===============================\n\n")
        handle.write(f"EmergencyCaseGrader: {emergency_score:.4f}\n")
        borderline_event = result["borderline_events"][4]
        handle.write(f"Vehicle 4 borderline trigger: {borderline_event}\n")
        handle.write(f"Vehicle 4 path: {' -> '.join(map(str, result['paths'][4]))}\n")
        handle.write(f"Vehicle 4 compact path: {' -> '.join(map(str, result['compact_paths'][4]))}\n")
        handle.write(f"Vehicle 4 delivered: {result['delivered'][4]} | Destroyed: {result['destroyed'][4]} | Final temp: {result['cargo_temps'][4]:.3f}\n")

    print("=== FIVE-VEHICLE HARD GRADER DEMO ===")
    print(f"Seed used: {seed}")
    print("Graph size: 30 nodes")
    for spec in VEHICLE_SPECS:
        idx = int(spec["vehicle_index"])
        print(
            f"Vehicle {idx}: start {spec['start_node']} -> target {spec['target_node']} | "
            f"cargo_temp={float(spec['cargo_temp']):.2f} | hard_margin={float(spec['hard_margin']):.2f} | borderline_margin={float(spec['borderline_margin']):.2f}"
        )
        print(f"Vehicle {idx} path: {' -> '.join(map(str, result['paths'][idx]))}")
        print(f"Vehicle {idx} compact path: {' -> '.join(map(str, result['compact_paths'][idx]))}")
        print(f"Vehicle {idx} actions: {[ (a,b,c) for a,b,c in result['actions'] if a == idx ]}")
        borderline_event = result["borderline_events"][idx]
        if borderline_event is not None:
            print(
                f"Vehicle {idx} borderline trigger: step {borderline_event['step']} | temp before step {borderline_event['temp_before']:.3f} | action {borderline_event['action']} | location {borderline_event['location']}"
            )
        delivery_event = result["delivery_events"][idx]
        if delivery_event is not None:
            print(
                f"Vehicle {idx} delivery step: {delivery_event['step']} | temp before step {delivery_event['temp_before_step']:.3f} | temp after delivery {delivery_event['temp_after_delivery']:.3f}"
            )
        print(f"Vehicle {idx} delivered: {result['delivered'][idx]} | destroyed: {result['destroyed'][idx]} | final temp: {result['cargo_temps'][idx]:.3f}")
    print(f"HardGrader: {result['hard_score']:.4f}")
    print(f"Saved graph PNG: {output_root}/five_vehicle_hard_model_paths.png")
    print(f"Saved text report: {report_path}")
    print(f"Saved emergency report: {emergency_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
