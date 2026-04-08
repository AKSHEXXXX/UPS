from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import networkx as nx

from core.config import ColdChainConfig
from core.shipment import CARGO_SPECS


@dataclass(frozen=True)
class ScenarioCase:
    name: str
    config: ColdChainConfig
    reset_options: Dict[str, Any]
    risky_shipment_ids: Sequence[int] = ()


def _cargo_type_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    reverse_map = {0: "vaccine", 1: "insulin", 2: "blood", 3: "organ"}
    return reverse_map.get(int(value), "vaccine")


class ScenarioGraderBase:
    level_name = "scenario"
    delivery_weight = 0.5
    path_weight = 0.3
    thermal_weight = 0.2
    safety_weight = 0.0
    cases: Sequence[ScenarioCase] = ()

    def evaluate_model(
        self,
        model,
        *,
        seed: int = 42,
        deterministic: bool = True,
        evaluation_training_step: int = 100000,
        trace_every: int = 0,
    ) -> Dict[str, Any]:
        case_summaries: List[Dict[str, Any]] = []
        delivery_scores: List[float] = []
        path_scores: List[float] = []
        thermal_scores: List[float] = []
        safety_scores: List[float] = []

        from evaluation.eval_contract import build_eval_env, run_eval_episode

        for index, case in enumerate(self.cases):
            env = build_eval_env(case.config)
            try:
                result = run_eval_episode(
                    model,
                    env,
                    seed=seed + index,
                    deterministic=deterministic,
                    curriculum_difficulty=3,
                    evaluation_training_step=evaluation_training_step,
                    reset_options=case.reset_options,
                    trace_every=trace_every,
                    collect_trajectory=True,
                )
                score = self._score_case(case, result)
                case_summaries.append({"case": case.name, **score, "result": result})
                delivery_scores.append(score["delivery_score"])
                path_scores.append(score["path_score"])
                thermal_scores.append(score["thermal_score"])
                safety_scores.append(score.get("safety_score", 0.0))
            finally:
                env.close()

        mean_delivery = float(sum(delivery_scores) / max(1, len(delivery_scores)))
        mean_path = float(sum(path_scores) / max(1, len(path_scores)))
        mean_thermal = float(sum(thermal_scores) / max(1, len(thermal_scores)))
        mean_safety = float(sum(safety_scores) / max(1, len(safety_scores)))
        final_score = (
            self.delivery_weight * mean_delivery
            + self.path_weight * mean_path
            + self.thermal_weight * mean_thermal
            + self.safety_weight * mean_safety
        )

        return {
            "level_name": self.level_name,
            "score": float(final_score),
            "delivery_score": mean_delivery,
            "path_score": mean_path,
            "thermal_score": mean_thermal,
            "safety_score": mean_safety,
            "cases": case_summaries,
        }

    def _score_case(self, case: ScenarioCase, result: Dict[str, Any]) -> Dict[str, float]:
        trajectory = result.get("trajectory", [])
        graph = result.get("graph_snapshot")
        if graph is None:
            raise RuntimeError("Missing graph snapshot in evaluation result")

        delivery_score = 1.0 if result.get("delivery_success", False) else 0.0
        delivery_steps = int(result.get("steps", 0))

        shipment_scores: List[float] = []
        thermal_scores: List[float] = []
        safety_scores: List[float] = []

        final_info = trajectory[-1][3] if trajectory else {}
        shipment_status = final_info.get("per_shipment_status", {})
        vehicle_status = final_info.get("per_vehicle_status", {})

        for shipment_index, shipment in enumerate(case.reset_options.get("shipment_cargo_types", [])):
            shipment_id = shipment_index
            shipment_info = shipment_status.get(shipment_id, {})
            vehicle_id = int(shipment_info.get("current_vehicle_id", -1))
            destination = int(shipment_info.get("destination_node", case.reset_options.get("shipment_destinations", [0])[shipment_index]))
            start_node = int(case.reset_options.get("vehicle_start_nodes", [0])[min(shipment_index, len(case.reset_options.get("vehicle_start_nodes", [0])) - 1)])

            delivery_step = self._delivery_step(trajectory, shipment_id)
            actual_steps = delivery_step if delivery_step is not None else delivery_steps
            shortest_steps = int(nx.shortest_path_length(graph, start_node, destination, weight="current_weight"))
            shipment_scores.append(self._path_score(shortest_steps, actual_steps))

            thermal_scores.append(self._thermal_score(shipment_info, shipment))

            if shipment_id in case.risky_shipment_ids:
                safety_scores.append(self._safety_score(case, trajectory, shipment_id, delivery_step, graph))

        if not shipment_scores:
            shipment_scores = [0.0]
        if not thermal_scores:
            thermal_scores = [0.0]

        path_score = float(sum(shipment_scores) / len(shipment_scores))
        thermal_score = float(sum(thermal_scores) / len(thermal_scores))
        safety_score = float(sum(safety_scores) / len(safety_scores)) if safety_scores else 1.0

        return {
            "delivery_score": float(delivery_score),
            "path_score": path_score,
            "thermal_score": thermal_score,
            "safety_score": safety_score,
        }

    def _delivery_step(self, trajectory: List[Any], shipment_id: int) -> int | None:
        for step_index, (_, _, _, info) in enumerate(trajectory, start=1):
            if info.get("per_shipment_status", {}).get(shipment_id, {}).get("is_delivered"):
                return step_index
        return None

    def _path_score(self, shortest_steps: int, actual_steps: int) -> float:
        if shortest_steps <= 0 or actual_steps <= 0:
            return 0.0
        return max(0.0, min(1.0, float(shortest_steps) / float(max(shortest_steps, actual_steps))))

    def _thermal_score(self, shipment_info: Dict[str, Any], cargo_type: Any) -> float:
        if not shipment_info:
            return 0.0
        cargo_name = _cargo_type_name(cargo_type)
        bounds = CARGO_SPECS[cargo_name]
        tolerance = max(1, int(bounds["tolerance_steps"]))
        excursion_duration = float(shipment_info.get("excursion_duration", 0.0))
        cargo_temp = float(shipment_info.get("cargo_temp", bounds["low"]))
        upper_overrun = max(0.0, cargo_temp - float(bounds["high"]))
        lower_overrun = max(0.0, float(bounds["low"]) - cargo_temp)
        range_penalty = excursion_duration + 3.0 * (upper_overrun + lower_overrun)
        return max(0.0, 1.0 - min(1.0, range_penalty / float(tolerance)))

    def _safety_score(
        self,
        case: ScenarioCase,
        trajectory: List[Any],
        shipment_id: int,
        delivery_step: int | None,
        graph: nx.Graph,
    ) -> float:
        shipment_destinations = case.reset_options.get("shipment_destinations", [])
        vehicle_starts = case.reset_options.get("vehicle_start_nodes", [])
        vehicle_id = min(shipment_id, max(0, len(vehicle_starts) - 1))
        if not vehicle_starts or shipment_id >= len(shipment_destinations):
            return 0.0

        start_node = int(vehicle_starts[vehicle_id])
        destination = int(shipment_destinations[shipment_id])
        risky_limit = float(CARGO_SPECS[_cargo_type_name(case.reset_options.get("shipment_cargo_types", ["blood"])[shipment_id])]["high"])
        final_info = trajectory[-1][3] if trajectory else {}
        shipment_info = final_info.get("per_shipment_status", {}).get(shipment_id, {})
        cargo_temp = float(shipment_info.get("cargo_temp", risky_limit))
        if cargo_temp <= risky_limit:
            return 1.0

        vehicle_path = [start_node]
        for _, _, _, info in trajectory:
            vehicle_snapshot = info.get("per_vehicle_status", {}).get(vehicle_id, {})
            if "location" in vehicle_snapshot:
                vehicle_path.append(int(vehicle_snapshot["location"]))

        depot_nodes = set(graph.graph.get("cold_depot_nodes", []))
        visited_depot = any(node in depot_nodes for node in vehicle_path)
        returned_to_hub = any(node == 0 for node in vehicle_path[1:])
        shortest_steps = int(nx.shortest_path_length(graph, start_node, destination, weight="current_weight"))
        actual_steps = delivery_step if delivery_step is not None else len(vehicle_path) - 1
        if visited_depot or returned_to_hub:
            route_score = self._path_score(shortest_steps, actual_steps)
            return max(0.0, min(1.0, 0.5 + 0.5 * route_score))
        return 0.0


class EasyGrader(ScenarioGraderBase):
    level_name = "easy"
    delivery_weight = 0.6
    path_weight = 0.3
    thermal_weight = 0.1
    cases = (
        ScenarioCase(
            name="easy-1",
            config=ColdChainConfig(n_vehicles=1, n_nodes=8, n_shipments=1, max_shipments=1, max_steps=80),
            reset_options={
                "vehicle_start_nodes": [1],
                "shipment_destinations": [6],
                "shipment_cargo_types": ["vaccine"],
                "shipment_cargo_temps": [4.0],
                "shipment_assignments": [0],
                "load_shipments_on_start": True,
            },
        ),
        ScenarioCase(
            name="easy-2",
            config=ColdChainConfig(n_vehicles=1, n_nodes=8, n_shipments=1, max_shipments=1, max_steps=80),
            reset_options={
                "vehicle_start_nodes": [2],
                "shipment_destinations": [5],
                "shipment_cargo_types": ["insulin"],
                "shipment_cargo_temps": [4.2],
                "shipment_assignments": [0],
                "load_shipments_on_start": True,
            },
        ),
        ScenarioCase(
            name="easy-3",
            config=ColdChainConfig(n_vehicles=1, n_nodes=8, n_shipments=1, max_shipments=1, max_steps=80),
            reset_options={
                "vehicle_start_nodes": [3],
                "shipment_destinations": [7],
                "shipment_cargo_types": ["blood"],
                "shipment_cargo_temps": [5.1],
                "shipment_assignments": [0],
                "load_shipments_on_start": True,
            },
        ),
    )


class MiddleGrader(ScenarioGraderBase):
    level_name = "middle"
    delivery_weight = 0.45
    path_weight = 0.35
    thermal_weight = 0.20
    cases = (
        ScenarioCase(
            name="middle-1",
            config=ColdChainConfig(n_vehicles=2, n_nodes=12, n_shipments=2, max_shipments=2, max_steps=120),
            reset_options={
                "vehicle_start_nodes": [1, 4],
                "shipment_destinations": [8, 10],
                "shipment_cargo_types": ["blood", "insulin"],
                "shipment_cargo_temps": [5.2, 4.3],
                "shipment_assignments": [0, 1],
                "load_shipments_on_start": True,
            },
        ),
        ScenarioCase(
            name="middle-2",
            config=ColdChainConfig(n_vehicles=2, n_nodes=12, n_shipments=2, max_shipments=2, max_steps=120),
            reset_options={
                "vehicle_start_nodes": [2, 5],
                "shipment_destinations": [9, 11],
                "shipment_cargo_types": ["vaccine", "blood"],
                "shipment_cargo_temps": [4.4, 5.4],
                "shipment_assignments": [0, 1],
                "load_shipments_on_start": True,
            },
        ),
        ScenarioCase(
            name="middle-3",
            config=ColdChainConfig(n_vehicles=2, n_nodes=12, n_shipments=2, max_shipments=2, max_steps=120),
            reset_options={
                "vehicle_start_nodes": [3, 6],
                "shipment_destinations": [7, 11],
                "shipment_cargo_types": ["insulin", "blood"],
                "shipment_cargo_temps": [4.1, 5.5],
                "shipment_assignments": [0, 1],
                "load_shipments_on_start": True,
            },
        ),
    )


class HardGrader(ScenarioGraderBase):
    level_name = "hard"
    delivery_weight = 0.35
    path_weight = 0.30
    thermal_weight = 0.20
    safety_weight = 0.15
    cases = (
        ScenarioCase(
            name="hard-1",
            config=ColdChainConfig(n_vehicles=5, n_nodes=18, n_shipments=5, max_shipments=5, max_steps=180),
            reset_options={
                "vehicle_start_nodes": [1, 2, 3, 4, 5],
                "shipment_destinations": [10, 11, 12, 13, 14],
                "shipment_cargo_types": ["blood", "blood", "insulin", "vaccine", "blood"],
                "shipment_cargo_temps": [4.4, 4.7, 4.2, 4.1, 5.4],
                "shipment_assignments": [0, 1, 2, 3, 4],
                "load_shipments_on_start": True,
            },
            risky_shipment_ids=(),
        ),
        ScenarioCase(
            name="hard-2",
            config=ColdChainConfig(n_vehicles=5, n_nodes=18, n_shipments=5, max_shipments=5, max_steps=180),
            reset_options={
                "vehicle_start_nodes": [2, 4, 6, 8, 10],
                "shipment_destinations": [15, 14, 13, 12, 11],
                "shipment_cargo_types": ["vaccine", "blood", "insulin", "blood", "blood"],
                "shipment_cargo_temps": [4.0, 5.2, 4.4, 5.1, 5.5],
                "shipment_assignments": [0, 1, 2, 3, 4],
                "load_shipments_on_start": True,
            },
            risky_shipment_ids=(),
        ),
        ScenarioCase(
            name="hard-3",
            config=ColdChainConfig(n_vehicles=5, n_nodes=18, n_shipments=5, max_shipments=5, max_steps=180),
            reset_options={
                "vehicle_start_nodes": [5, 6, 7, 8, 9],
                "shipment_destinations": [10, 9, 8, 7, 6],
                "shipment_cargo_types": ["vaccine", "insulin", "vaccine", "blood", "blood"],
                "shipment_cargo_temps": [8.6, 4.6, 4.0, 5.7, 5.9],
                "shipment_assignments": [0, 1, 2, 3, 4],
                "load_shipments_on_start": True,
            },
            risky_shipment_ids=(0,),
        ),
    )


def _shipment_delivery_steps(trajectory: List[Any], n_shipments: int) -> Dict[int, int]:
    delivery_steps: Dict[int, int] = {}
    for step_index, (_, _, _, info) in enumerate(trajectory, start=1):
        shipment_status = info.get("per_shipment_status", {})
        for shipment_id in range(n_shipments):
            if shipment_id in delivery_steps:
                continue
            if shipment_status.get(shipment_id, {}).get("is_delivered", False):
                delivery_steps[shipment_id] = step_index
    return delivery_steps


def _cargo_multiplier(cargo_type: str) -> float:
    return {
        "organ": 3.0,
        "blood": 1.8,
        "vaccine": 1.3,
        "insulin": 1.2,
    }.get(str(cargo_type), 1.0)


def _episode_metrics(case: ScenarioCase, result: Dict[str, Any]) -> Dict[str, float]:
    trajectory = result.get("trajectory", [])
    final_info = trajectory[-1][3] if trajectory else {}
    shipment_status = final_info.get("per_shipment_status", {})
    shipment_count = int(case.config.n_shipments)
    delivery_steps = _shipment_delivery_steps(trajectory, shipment_count)
    reset_opts = case.reset_options

    delivered_count = 0
    thermal_ok_count = 0
    efficiency_parts: List[float] = []
    triage_weights_total = 0.0
    triage_weights_delivered = 0.0

    graph = result.get("graph_snapshot")
    vehicle_starts = list(reset_opts.get("vehicle_start_nodes", [0]))
    destinations = list(reset_opts.get("shipment_destinations", []))
    assignments = list(reset_opts.get("shipment_assignments", []))

    for shipment_id in range(shipment_count):
        status = shipment_status.get(shipment_id, {})
        is_delivered = bool(status.get("is_delivered", False))
        is_destroyed = bool(status.get("is_destroyed", False))
        cargo_type = str(status.get("cargo_type", "vaccine"))
        priority = int(status.get("priority", 0))
        current_temp = float(status.get("cargo_temp", 4.0))
        lower = float(status.get("temp_lower_bound", CARGO_SPECS[cargo_type]["low"]))
        upper = float(status.get("temp_upper_bound", CARGO_SPECS[cargo_type]["high"]))

        triage_weight = (priority + 1.0) * _cargo_multiplier(cargo_type)
        triage_weights_total += triage_weight
        if is_delivered:
            triage_weights_delivered += triage_weight

        if is_delivered:
            delivered_count += 1
            if lower <= current_temp <= upper and not is_destroyed:
                thermal_ok_count += 1

            if graph is not None and shipment_id < len(destinations):
                vehicle_index = assignments[shipment_id] if shipment_id < len(assignments) else min(shipment_id, len(vehicle_starts) - 1)
                vehicle_index = max(0, min(vehicle_index, len(vehicle_starts) - 1))
                start_node = int(vehicle_starts[vehicle_index])
                destination = int(destinations[shipment_id])
                shortest_steps = int(nx.shortest_path_length(graph, start_node, destination, weight="current_weight"))
                actual_steps = int(delivery_steps.get(shipment_id, case.config.max_steps))
                efficiency_parts.append(max(0.0, min(1.0, float(shortest_steps) / float(max(shortest_steps, actual_steps)))))

    delivery_ratio = float(delivered_count / max(1, shipment_count))
    thermal_ratio = float(thermal_ok_count / max(1, delivered_count)) if delivered_count > 0 else 0.0
    efficiency_ratio = float(sum(efficiency_parts) / len(efficiency_parts)) if efficiency_parts else 0.0
    speed_ratio = max(0.0, min(1.0, 1.0 - float(result.get("steps", 0)) / float(case.config.max_steps)))

    delivered_by_step: List[tuple[float, int]] = []
    for shipment_id in range(shipment_count):
        status = shipment_status.get(shipment_id, {})
        if bool(status.get("is_delivered", False)):
            cargo_type = str(status.get("cargo_type", "vaccine"))
            priority = int(status.get("priority", 0))
            weight = (priority + 1.0) * _cargo_multiplier(cargo_type)
            delivered_by_step.append((weight, int(delivery_steps.get(shipment_id, case.config.max_steps))))
    delivered_by_step.sort(key=lambda pair: pair[1])
    delivered_order_weights = [pair[0] for pair in delivered_by_step]

    all_weights: List[float] = []
    for shipment_id in range(shipment_count):
        status = shipment_status.get(shipment_id, {})
        cargo_type = str(status.get("cargo_type", "vaccine"))
        priority = int(status.get("priority", 0))
        all_weights.append((priority + 1.0) * _cargo_multiplier(cargo_type))
    ideal_weights = sorted(all_weights, reverse=True)

    dcg = sum(weight / float(index + 1) for index, weight in enumerate(delivered_order_weights))
    ideal_dcg = sum(weight / float(index + 1) for index, weight in enumerate(ideal_weights))
    order_score = float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
    coverage_score = float(triage_weights_delivered / triage_weights_total) if triage_weights_total > 0 else 0.0
    triage_score = 0.6 * coverage_score + 0.4 * order_score

    return {
        "delivery_ratio": delivery_ratio,
        "thermal_ratio": thermal_ratio,
        "efficiency_ratio": efficiency_ratio,
        "speed_ratio": speed_ratio,
        "triage_score": triage_score,
    }


def _run_case_episodes(
    model,
    case: ScenarioCase,
    *,
    seeds: Sequence[int],
    deterministic: bool,
    evaluation_training_step: int,
    trace_every: int,
) -> List[Dict[str, Any]]:
    from evaluation.eval_contract import build_eval_env, run_eval_episode

    results: List[Dict[str, Any]] = []
    env = build_eval_env(case.config)
    try:
        for seed in seeds:
            result = run_eval_episode(
                model,
                env,
                seed=int(seed),
                deterministic=deterministic,
                curriculum_difficulty=3,
                evaluation_training_step=evaluation_training_step,
                reset_options=case.reset_options,
                trace_every=trace_every,
                collect_trajectory=True,
            )
            results.append(result)
    finally:
        env.close()
    return results


def _aggregate_case_metrics(case: ScenarioCase, results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    metrics = [_episode_metrics(case, result) for result in results]
    if not metrics:
        return {
            "delivery_ratio": 0.0,
            "thermal_ratio": 0.0,
            "efficiency_ratio": 0.0,
            "speed_ratio": 0.0,
            "triage_score": 0.0,
        }

    keys = ["delivery_ratio", "thermal_ratio", "efficiency_ratio", "speed_ratio", "triage_score"]
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def _score_easy(agg: Dict[str, float]) -> float:
    return 0.60 * agg["delivery_ratio"] + 0.20 * agg["thermal_ratio"] + 0.20 * agg["speed_ratio"]


def _score_moderate(agg: Dict[str, float]) -> float:
    score = (
        0.45 * agg["delivery_ratio"]
        + 0.25 * agg["thermal_ratio"]
        + 0.15 * agg["efficiency_ratio"]
        + 0.10 * agg["speed_ratio"]
    )
    if agg["delivery_ratio"] < 0.40:
        score = min(score, 0.50)
    return score


def _score_hard(agg_main: Dict[str, float], adv_delivery_ratio: float) -> float:
    score = (
        0.40 * agg_main["delivery_ratio"]
        + 0.25 * agg_main["thermal_ratio"]
        + 0.20 * agg_main["efficiency_ratio"]
        + 0.15 * agg_main["speed_ratio"]
    )
    if adv_delivery_ratio < 0.30:
        score = min(score, 0.35)
    return score


def _score_extreme(agg: Dict[str, float]) -> float:
    return (
        0.30 * agg["triage_score"]
        + 0.25 * agg["thermal_ratio"]
        + 0.20 * agg["efficiency_ratio"]
        + 0.15 * agg["speed_ratio"]
        + 0.10 * agg["delivery_ratio"]
    )


def run_full_evaluation(
    model,
    *,
    seed: int = 42,
    deterministic: bool = True,
    evaluation_training_step: int = 100000,
    trace_every: int = 0,
) -> Dict[str, Any]:
    thresholds = {
        "easy": 0.80,
        "moderate": 0.65,
        "hard": 0.50,
        "extreme": 0.35,
    }

    easy_case = ScenarioCase(
        name="easy-tier",
        config=ColdChainConfig(
            n_vehicles=1,
            n_shipments=1,
            max_shipments=1,
            n_nodes=10,
            n_cold_depots=3,
            max_steps=200,
            weather_events_enabled=False,
            breakdown_probability=0.0,
            refrigeration_degradation_prob=0.0,
        ),
        reset_options={
            "vehicle_start_nodes": [1],
            "shipment_destinations": [8],
            "shipment_cargo_types": ["vaccine"],
            "shipment_cargo_temps": [4.0],
            "shipment_assignments": [0],
            "shipment_deadlines": [200],
            "load_shipments_on_start": True,
        },
    )

    moderate_case = ScenarioCase(
        name="moderate-tier",
        config=ColdChainConfig(
            n_vehicles=2,
            n_shipments=3,
            max_shipments=3,
            n_nodes=14,
            n_cold_depots=3,
            max_steps=180,
            weather_events_enabled=False,
            breakdown_probability=0.003,
            refrigeration_degradation_prob=0.005,
        ),
        reset_options={
            "vehicle_start_nodes": [1, 4],
            "shipment_destinations": [11, 12, 9],
            "shipment_cargo_types": ["vaccine", "insulin", "blood"],
            "shipment_cargo_temps": [4.0, 4.1, 5.2],
            "shipment_assignments": [0, 1, 0],
            "shipment_deadlines": [160, 170, 165],
            "forced_weather_only": True,
            "forced_weather_events": [{"step": 35, "event": "HEATWAVE", "duration": 12}],
            "load_shipments_on_start": True,
        },
    )

    hard_case = ScenarioCase(
        name="hard-tier",
        config=ColdChainConfig(
            n_vehicles=3,
            n_shipments=5,
            max_shipments=5,
            n_nodes=18,
            n_cold_depots=2,
            max_steps=170,
            weather_events_enabled=True,
            breakdown_probability=0.01,
            refrigeration_degradation_prob=0.01,
        ),
        reset_options={
            "vehicle_start_nodes": [1, 2, 4],
            "shipment_destinations": [13, 14, 15, 10, 11],
            "shipment_cargo_types": ["vaccine", "insulin", "blood", "vaccine", "blood"],
            "shipment_cargo_temps": [4.0, 4.3, 5.1, 4.2, 5.0],
            "shipment_assignments": [0, 1, 2, 0, 1],
            "shipment_deadlines": [90, 95, 100, 85, 92],
            "load_shipments_on_start": True,
        },
    )

    hard_adversarial_case = ScenarioCase(
        name="hard-tier-adversarial",
        config=hard_case.config,
        reset_options={
            **hard_case.reset_options,
            "forced_weather_only": True,
            "forced_weather_events": [
                {"step": 1, "event": "STORM", "duration": 18},
                {"step": 25, "event": "HEATWAVE", "duration": 14},
            ],
            "forced_breakdowns": [{"step": 12, "vehicle_ids": [0]}],
        },
    )

    extreme_case = ScenarioCase(
        name="extreme-tier",
        config=ColdChainConfig(
            n_vehicles=5,
            n_shipments=8,
            max_shipments=8,
            n_nodes=24,
            n_cold_depots=2,
            max_steps=220,
            weather_events_enabled=True,
            breakdown_probability=0.015,
            refrigeration_degradation_prob=0.02,
        ),
        reset_options={
            "vehicle_start_nodes": [1, 2, 3, 4, 5],
            "shipment_destinations": [18, 19, 20, 21, 22, 16, 17, 23],
            "shipment_cargo_types": ["organ", "organ", "blood", "blood", "vaccine", "insulin", "blood", "vaccine"],
            "shipment_cargo_temps": [2.5, 2.8, 5.4, 5.3, 4.1, 4.2, 5.2, 4.0],
            "shipment_priorities": [2, 2, 2, 1, 1, 0, 1, 0],
            "shipment_assignments": [0, 1, 2, 3, 4, 0, 1, 2],
            "shipment_deadlines": [110, 112, 145, 150, 160, 170, 155, 175],
            "forced_weather_only": True,
            "forced_weather_events": [
                {"step": 0, "event": "STORM", "duration": 20},
                {"step": 40, "event": "HEATWAVE", "duration": 18},
            ],
            "forced_breakdowns": [{"step": 18, "vehicle_ids": [2]}, {"step": 45, "vehicle_ids": [4]}],
            "load_shipments_on_start": True,
        },
    )

    report_rows: List[Dict[str, Any]] = []
    stop_early = False

    easy_results = _run_case_episodes(
        model,
        easy_case,
        seeds=[seed + idx for idx in range(5)],
        deterministic=deterministic,
        evaluation_training_step=evaluation_training_step,
        trace_every=trace_every,
    )
    easy_agg = _aggregate_case_metrics(easy_case, easy_results)
    easy_score = _score_easy(easy_agg)
    easy_pass = easy_score >= thresholds["easy"]
    report_rows.append({"tier": "easy", "threshold": thresholds["easy"], "score": easy_score, "pass": easy_pass, **easy_agg})
    if not easy_pass:
        stop_early = True

    if not stop_early:
        moderate_results = _run_case_episodes(
            model,
            moderate_case,
            seeds=[seed + 10 + idx for idx in range(5)],
            deterministic=deterministic,
            evaluation_training_step=evaluation_training_step,
            trace_every=trace_every,
        )
        moderate_agg = _aggregate_case_metrics(moderate_case, moderate_results)
        moderate_score = _score_moderate(moderate_agg)
        moderate_pass = moderate_score >= thresholds["moderate"]
        report_rows.append(
            {
                "tier": "moderate",
                "threshold": thresholds["moderate"],
                "score": moderate_score,
                "pass": moderate_pass,
                "delivery_cap_applied": bool(moderate_agg["delivery_ratio"] < 0.40),
                **moderate_agg,
            }
        )
        if not moderate_pass:
            stop_early = True

    if not stop_early:
        hard_results = _run_case_episodes(
            model,
            hard_case,
            seeds=[seed + 20 + idx for idx in range(5)],
            deterministic=deterministic,
            evaluation_training_step=evaluation_training_step,
            trace_every=trace_every,
        )
        hard_adv_results = _run_case_episodes(
            model,
            hard_adversarial_case,
            seeds=[911, 1337, 2025, 4444, 8888],
            deterministic=deterministic,
            evaluation_training_step=evaluation_training_step,
            trace_every=trace_every,
        )
        hard_agg = _aggregate_case_metrics(hard_case, hard_results)
        hard_adv_agg = _aggregate_case_metrics(hard_adversarial_case, hard_adv_results)
        hard_score = _score_hard(hard_agg, hard_adv_agg["delivery_ratio"])
        hard_pass = hard_score >= thresholds["hard"]
        report_rows.append(
            {
                "tier": "hard",
                "threshold": thresholds["hard"],
                "score": hard_score,
                "pass": hard_pass,
                "adversarial_delivery_ratio": hard_adv_agg["delivery_ratio"],
                "adversarial_cap_applied": bool(hard_adv_agg["delivery_ratio"] < 0.30),
                **hard_agg,
            }
        )
        if not hard_pass:
            stop_early = True

    if not stop_early:
        extreme_results = _run_case_episodes(
            model,
            extreme_case,
            seeds=[seed + 30 + idx for idx in range(5)],
            deterministic=deterministic,
            evaluation_training_step=evaluation_training_step,
            trace_every=trace_every,
        )
        extreme_agg = _aggregate_case_metrics(extreme_case, extreme_results)
        extreme_score = _score_extreme(extreme_agg)
        extreme_pass = extreme_score >= thresholds["extreme"]
        report_rows.append({"tier": "extreme", "threshold": thresholds["extreme"], "score": extreme_score, "pass": extreme_pass, **extreme_agg})

    return {
        "stopped_early": stop_early,
        "rows": report_rows,
        "thresholds": thresholds,
    }