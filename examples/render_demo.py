from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from server.env import ColdChainEnv


def _get_first_legal_action(env: ColdChainEnv) -> np.ndarray:
    """Get the first legal action from the action mask."""
    mask = env.action_masks()
    flat_index = int(np.argmax(mask))
    vehicle = flat_index // (6 * env.config.n_nodes)
    remainder = flat_index % (6 * env.config.n_nodes)
    action_type = remainder // env.config.n_nodes
    target = remainder % env.config.n_nodes
    return np.array([vehicle, action_type, target])



def _node_positions(graph: nx.Graph) -> dict[int, tuple[float, float]]:
    return nx.spring_layout(graph, seed=7)


def _render_env(env: ColdChainEnv) -> None:
    """
    Render the cold chain environment showing:
    - City graph with nodes (hub in red, cold depots in blue)
    - Vehicle positions as colored dots
    - Shipment status table
    """
    assert env.graph is not None

    positions = _node_positions(env.graph)
    cold_depots = set(env.graph.graph.get("cold_depot_nodes", []))
    hub_node = 0

    figure = plt.figure(figsize=(15, 8))
    grid = figure.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
    ax_graph = figure.add_subplot(grid[0, 0])
    ax_table = figure.add_subplot(grid[0, 1])

    # Draw edges with widths based on traffic
    edge_widths = [0.8 + float(data.get("current_weight", 1.0)) * 0.05 for _, _, data in env.graph.edges(data=True)]
    nx.draw_networkx_edges(env.graph, positions, ax=ax_graph, width=edge_widths, edge_color="#a7b3c5", alpha=0.75)

    # Draw nodes
    regular_nodes = [node for node in env.graph.nodes if node not in cold_depots and node != hub_node]
    nx.draw_networkx_nodes(env.graph, positions, nodelist=regular_nodes, node_color="#d7dbe4", node_size=520, ax=ax_graph)
    nx.draw_networkx_nodes(env.graph, positions, nodelist=[hub_node], node_color="#ff6b6b", node_size=700, ax=ax_graph, label="Hub")
    if cold_depots:
        nx.draw_networkx_nodes(env.graph, positions, nodelist=list(cold_depots), node_color="#4dabf7", node_size=700, ax=ax_graph, label="Cold Depot")

    labels = {node: str(node) for node in env.graph.nodes}
    nx.draw_networkx_labels(env.graph, positions, labels=labels, font_size=9, font_color="#1f2937", ax=ax_graph)

    # Draw vehicles
    vehicle_colors = ["#e03131", "#2b8a3e", "#f59f00", "#6741d9", "#0b7285", "#c2255c"]
    for index, vehicle in enumerate(sorted(env.vehicles, key=lambda item: item.id)):
        x_coord, y_coord = positions[vehicle.location]
        color = vehicle_colors[index % len(vehicle_colors)]
        ax_graph.scatter([x_coord], [y_coord], s=180, c=color, edgecolors="white", linewidths=1.5, zorder=5)
        ax_graph.text(x_coord, y_coord + 0.04, f"V{vehicle.id}", fontsize=8, color=color, ha="center", va="bottom", weight="bold")

    ax_graph.set_title("ColdChain City Graph (Phase 12.3)", fontsize=14, weight="bold")
    ax_graph.legend(loc="upper left", fontsize=9)
    ax_graph.axis("off")

    # Build shipment table
    shipment_rows = []
    status_counts = Counter()
    for shipment in sorted(env.shipments, key=lambda item: item.id):
        status = "delivered" if shipment.is_delivered else "destroyed" if shipment.is_destroyed else "active"
        status_counts[status] += 1
        shipment_rows.append(
            [
                str(shipment.id),
                shipment.cargo_type[:3],  # abbreviate for space
                str(shipment.current_vehicle_id) if shipment.current_vehicle_id >= 0 else "hub",
                str(shipment.destination_node),
                f"{shipment.cargo_temp:.1f}°C",
                status,
            ]
        )

    ax_table.axis("off")
    ax_table.set_title("Shipment Status Snapshot", fontsize=14, weight="bold", pad=12)
    if shipment_rows:
        table = ax_table.table(
            cellText=shipment_rows,
            colLabels=["ID", "Cargo", "Veh", "Dest", "Temp", "Status"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)

    # Summary stats
    summary_text = (
        f"Vehicles: {len(env.vehicles)}\n"
        f"Shipments: {len(env.shipments)}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Active: {status_counts.get('active', 0)}\n"
        f"Delivered: {status_counts.get('delivered', 0)}\n"
        f"Destroyed: {status_counts.get('destroyed', 0)}"
    )
    ax_table.text(0.05, 0.05, summary_text, transform=ax_table.transAxes, fontsize=10, 
                  va="bottom", ha="left", family="monospace",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    plt.savefig("/tmp/coldchain_render.png", dpi=100, bbox_inches="tight")
    print("✓ Render saved to /tmp/coldchain_render.png")
    plt.show()


def main():
    """
    Render Demo for ColdChain-Gym (Phase 12.3)
    
    Demonstrates:
    - Environment instantiation
    - Running a few steps with legal actions
    - Visualizing the city graph with matplotlib
    - Displaying vehicle positions and shipment status
    """
    print("=" * 60)
    print("ColdChain-Gym Render Demo")
    print("=" * 60)
    
    env = ColdChainEnv()
    obs, info = env.reset(seed=0)
    
    print(f"Environment created with {env.config.n_vehicles} vehicles, {env.config.n_nodes} nodes")
    print(f"Running 5 steps with legal actions...\n")
    
    for step_num in range(5):
        action = _get_first_legal_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_num + 1}: action={action}, reward={reward:.4f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step_num + 1}")
            break
    
    print("\nRendering environment visualization...")
    _render_env(env)
    print("=" * 60)


if __name__ == "__main__":
    main()

