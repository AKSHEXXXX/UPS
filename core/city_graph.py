from __future__ import annotations

from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np

from .config import ColdChainConfig


def build_city_graph(config: ColdChainConfig, rng: np.random.Generator) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(config.n_nodes))

    graph.graph["cold_depot_nodes"] = list(range(1, config.n_cold_depots + 1))

    for node in range(1, config.n_nodes):
        parent = int(rng.integers(0, node))
        base_weight = int(rng.integers(1, 8))
        graph.add_edge(
            parent,
            node,
            base_weight=base_weight,
            current_weight=float(base_weight),
        )

    for left in range(config.n_nodes):
        for right in range(left + 1, config.n_nodes):
            if graph.has_edge(left, right):
                continue
            if rng.random() < 0.3:
                base_weight = int(rng.integers(1, 8))
                graph.add_edge(
                    left,
                    right,
                    base_weight=base_weight,
                    current_weight=float(base_weight),
                )

    return graph


def refresh_edge_weights(graph: nx.Graph, traffic_multiplier: float) -> None:
    for left, right, data in graph.edges(data=True):
        data["current_weight"] = float(data["base_weight"] * traffic_multiplier)


def nearest_cold_depot(graph: nx.Graph, from_node: int, depot_nodes: List[int]) -> Tuple[int, int]:
    lengths = nx.single_source_dijkstra_path_length(graph, from_node, weight="current_weight")
    best_depot = min(depot_nodes, key=lambda depot: lengths.get(depot, float("inf")))
    return best_depot, int(lengths.get(best_depot, 9999))


def detour_cost(graph: nx.Graph, from_node: int, depot_node: int, dest_node: int) -> int:
    direct = nx.shortest_path_length(graph, from_node, dest_node, weight="current_weight")
    via_depot = nx.shortest_path_length(graph, from_node, depot_node, weight="current_weight")
    via_depot += nx.shortest_path_length(graph, depot_node, dest_node, weight="current_weight")
    return max(0, int(via_depot - direct))
