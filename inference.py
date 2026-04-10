from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience dependency
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

from core.config import ColdChainConfig
from server.env import ColdChainEnv


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
DEFAULT_TASK_NAME = "coldchain-gym"
DEFAULT_BENCHMARK_NAME = "coldchain-gym"
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "20"))

ACTION_NAMES = {
    0: "WAIT",
    1: "REROUTE",
    2: "DIVERT_COLD_DEPOT",
    3: "SWAP_VEHICLE",
    4: "EXPEDITE",
    5: "ABORT",
}

SYSTEM_PROMPT = """You control a cold-chain delivery environment.
Return exactly one JSON object with integer fields vehicle_index, action_type, target_index.
Use only legal actions from the provided candidate_actions list.
Prioritize, in order:
1. Deliver active shipments before deadlines.
2. Reduce temperature excursion risk.
3. Avoid oscillating between equivalent reroutes.
4. Use WAIT only when movement would clearly be worse.
Do not include markdown or any explanatory text."""


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _fmt_float(value: float) -> str:
    return f"{float(value):.2f}"


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _format_error(error: str | None) -> str:
    if error is None or not str(error).strip():
        return "null"
    return _single_line(str(error))


def _action_to_str(action: dict[str, int]) -> str:
    action_name = ACTION_NAMES.get(action["action_type"], "UNKNOWN")
    return (
        f"vehicle_index={action['vehicle_index']},"
        f"action_type={action['action_type']}:{action_name},"
        f"target_index={action['target_index']}"
    )


def _legal_actions(mask: list[int], config: ColdChainConfig) -> list[dict[str, int]]:
    n_nodes = config.n_nodes
    actions: list[dict[str, int]] = []
    for flat_index, enabled in enumerate(mask):
        if not enabled:
            continue
        vehicle_index = flat_index // (6 * n_nodes)
        action_type = (flat_index % (6 * n_nodes)) // n_nodes
        target_index = flat_index % n_nodes
        actions.append(
            {
                "vehicle_index": int(vehicle_index),
                "action_type": int(action_type),
                "target_index": int(target_index),
            }
        )
    return actions


def _candidate_action_summary(legal_actions: list[dict[str, int]], max_items: int = 24) -> list[dict[str, int | str]]:
    summary: list[dict[str, int | str]] = []
    for action in legal_actions[:max_items]:
        summary.append(
            {
                "vehicle_index": action["vehicle_index"],
                "action_type": action["action_type"],
                "action_name": ACTION_NAMES.get(action["action_type"], "UNKNOWN"),
                "target_index": action["target_index"],
            }
        )
    return summary


def _shipment_risk(shipment: dict[str, Any]) -> float:
    temp = float(shipment.get("cargo_temp", 0.0))
    low = float(shipment.get("temp_lower_bound", 0.0))
    high = float(shipment.get("temp_upper_bound", 0.0))
    deadline = float(shipment.get("time_to_deadline", 9999))
    excursion_count = float(shipment.get("excursion_count", 0.0))
    excursion_duration = float(shipment.get("excursion_duration", 0.0))
    destroyed = int(shipment.get("is_destroyed", 0))
    delivered = int(shipment.get("is_delivered", 0))
    if destroyed or delivered:
        return -1e9

    temperature_penalty = 0.0
    if temp < low:
        temperature_penalty += (low - temp) * 25.0
    if temp > high:
        temperature_penalty += (temp - high) * 25.0

    margin_penalty = 0.0
    margin_penalty += max(0.0, 1.0 - (temp - low)) * 3.0
    margin_penalty += max(0.0, 1.0 - (high - temp)) * 3.0

    urgency_bonus = max(0.0, 50.0 - deadline)
    return urgency_bonus + temperature_penalty + margin_penalty + excursion_count * 10.0 + excursion_duration * 2.0


def _priority_shipment(observation: dict[str, Any]) -> dict[str, Any] | None:
    shipments = observation.get("shipments", [])
    active = [shipment for shipment in shipments if not int(shipment.get("is_delivered", 0)) and not int(shipment.get("is_destroyed", 0))]
    if not active:
        return None
    return max(active, key=_shipment_risk)


def _vehicle_by_id(observation: dict[str, Any], vehicle_id: int) -> dict[str, Any] | None:
    for vehicle in observation.get("vehicles", []):
        if int(vehicle.get("id", -1)) == int(vehicle_id):
            return vehicle
    return None


def _shipment_by_id(observation: dict[str, Any], shipment_id: int) -> dict[str, Any] | None:
    for shipment in observation.get("shipments", []):
        if int(shipment.get("id", -1)) == int(shipment_id):
            return shipment
    return None


def _action_score(action: dict[str, int], observation: dict[str, Any], config: ColdChainConfig) -> float:
    vehicle_index = int(action["vehicle_index"])
    action_type = int(action["action_type"])
    target_index = int(action["target_index"])
    global_noop_vehicle = int(config.n_vehicles)

    if vehicle_index == global_noop_vehicle:
        return -25.0

    vehicle = _vehicle_by_id(observation, vehicle_index)
    if vehicle is None:
        return -1e9

    score = 0.0
    onboard_ids = [int(shipment_id) for shipment_id in vehicle.get("shipments_onboard", []) if int(shipment_id) >= 0]
    onboard_shipments = [shipment for shipment_id in onboard_ids if (shipment := _shipment_by_id(observation, shipment_id)) is not None]
    urgent_onboard = max((_shipment_risk(shipment) for shipment in onboard_shipments), default=-1e9)
    priority = _priority_shipment(observation)

    if action_type == 0:
        score -= 4.0
        if vehicle.get("status") == 2:
            score -= 2.0
    elif action_type == 1:
        score += 1.0
        if onboard_shipments:
            primary = max(onboard_shipments, key=_shipment_risk)
            destination = int(primary.get("destination_node", 0))
            if target_index == destination:
                score += 20.0
            else:
                score -= 6.0
        elif priority is not None:
            if target_index == int(priority.get("destination_node", 0)):
                score += 8.0
        steps_to_destination = float(vehicle.get("steps_to_destination", 9999))
        if target_index == int(vehicle.get("location", -1)):
            score -= 12.0
        score += max(0.0, 12.0 - min(12.0, steps_to_destination)) * 0.5
    elif action_type == 2:
        score += 10.0 if urgent_onboard > 0 else -8.0
        if target_index == int(vehicle.get("nearest_cold_depot_node", -1)):
            score += 8.0
    elif action_type == 3:
        score += 2.0 if onboard_shipments else -5.0
    elif action_type == 4:
        in_transit = int(vehicle.get("status", -1)) == 1
        score += 6.0 if in_transit else -10.0
        if urgent_onboard > 0:
            score += 6.0
    elif action_type == 5:
        score += 3.0 if urgent_onboard > 20.0 else -20.0

    if priority is not None and onboard_shipments:
        if any(int(shipment.get("id", -1)) == int(priority.get("id", -2)) for shipment in onboard_shipments):
            score += 6.0

    return score


def _ranked_actions(legal_actions: list[dict[str, int]], observation: dict[str, Any], config: ColdChainConfig, limit: int = 12) -> list[dict[str, int]]:
    ranked = sorted(
        legal_actions,
        key=lambda action: (_action_score(action, observation, config), -action["vehicle_index"], -action["action_type"]),
        reverse=True,
    )
    return ranked[:limit]


def _build_prompt(observation: dict[str, Any], candidate_actions: list[dict[str, int]], task_name: str) -> str:
    vehicles = observation.get("vehicles", [])
    shipments = observation.get("shipments", [])
    global_state = observation.get("global_state", {})
    priority = _priority_shipment(observation)

    prompt_payload = {
        "task": task_name,
        "global_state": global_state,
        "vehicles": vehicles,
        "shipments": shipments,
        "legal_action_count": len(candidate_actions),
        "priority_shipment": priority,
        "candidate_actions": _candidate_action_summary(candidate_actions),
        "instructions": "Choose exactly one action from candidate_actions. Prefer the priority shipment and avoid route thrashing.",
    }
    return json.dumps(prompt_payload, separators=(",", ":"))


def _extract_action(response_text: str) -> dict[str, int]:
    payload = json.loads(response_text)
    return {
        "vehicle_index": int(payload["vehicle_index"]),
        "action_type": int(payload["action_type"]),
        "target_index": int(payload["target_index"]),
    }


def _choose_fallback_action(candidate_actions: list[dict[str, int]], observation: dict[str, Any], config: ColdChainConfig) -> dict[str, int]:
    ranked = _ranked_actions(candidate_actions, observation, config, limit=1)
    if ranked:
        return ranked[0]
    return {"vehicle_index": int(config.n_vehicles), "action_type": 0, "target_index": 0}


def _request_action(
    client: OpenAI,
    model_name: str,
    observation: dict[str, Any],
    candidate_actions: list[dict[str, int]],
    task_name: str,
    benchmark_name: str,
    request_timeout: float,
) -> dict[str, int]:
    prompt = _build_prompt(observation, candidate_actions, task_name)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"benchmark={benchmark_name}\n"
                    f"Return only JSON.\n"
                    f"{prompt}"
                ),
            },
        ],
        temperature=0,
        timeout=request_timeout,
    )
    content = completion.choices[0].message.content or ""
    return _extract_action(content)


def _validate_action(action: dict[str, int], candidate_actions: list[dict[str, int]]) -> bool:
    return action in candidate_actions


def _build_client() -> OpenAI:
    if not API_KEY:
        raise ValueError("Set HF_TOKEN, OPENAI_API_KEY, or OPENAI_TOKEN in .env")
    if API_BASE_URL.startswith("<") or MODEL_NAME.startswith("<"):
        raise ValueError("API_BASE_URL and MODEL_NAME must be configured for LLM inference")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven inference for coldchain-gym")
    parser.add_argument("--task-name", type=str, default=DEFAULT_TASK_NAME)
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ColdChainConfig(max_steps=int(args.max_steps))
    env = ColdChainEnv(config=config)
    rewards: list[float] = []
    success = False
    steps = 0
    end_score = 0.0
    start_emitted = False
    info: dict[str, Any] = {}
    grader_scores: dict[str, Any] = {}
    termination_reason = "unknown"
    client: OpenAI | None = None
    client_error: str | None = None

    try:
        try:
            client = _build_client()
        except Exception as exc:
            client = None
            client_error = str(exc)
        print(f"[START] task={args.task_name} env={args.benchmark} model={MODEL_NAME}", flush=True)
        start_emitted = True
        if client is None:
            print(f"[INFO] LLM client unavailable, using heuristic fallback policy ({_single_line(client_error or 'unknown error')})", flush=True)

        obs, info = env.reset(seed=int(args.seed))
        done = False

        while not done:
            observation = env._core._get_obs(reward=0.0, done=False, message="llm planning", info=info).model_dump()
            legal_actions = _legal_actions(observation.get("action_mask", []), config)
            if not legal_actions:
                raise RuntimeError("No legal actions available")
            candidate_actions = _ranked_actions(legal_actions, observation, config)

            action_error: str | None = None
            if client is not None:
                try:
                    action = _request_action(
                        client=client,
                        model_name=MODEL_NAME,
                        observation=observation,
                        candidate_actions=candidate_actions,
                        task_name=args.task_name,
                        benchmark_name=args.benchmark,
                        request_timeout=float(args.request_timeout),
                    )
                except Exception as exc:
                    action_error = str(exc)
                    action = _choose_fallback_action(candidate_actions, observation, config)
            else:
                action = _choose_fallback_action(candidate_actions, observation, config)

            if not _validate_action(action, candidate_actions):
                if action_error is None:
                    action_error = "Model produced an illegal action; replaced with fallback"
                action = _choose_fallback_action(candidate_actions, observation, config)

            obs, reward, terminated, truncated, info = env.step(
                [action["vehicle_index"], action["action_type"], action["target_index"]]
            )
            done = bool(terminated or truncated)
            steps += 1
            rewards.append(float(reward))

            env_error = info.get("last_action_error")

            print(
                "[STEP] "
                f"step={steps} "
                f"action={_action_to_str(action)} "
                f"reward={_fmt_float(reward)} "
                f"done={_bool_text(done)} "
                f"error={_format_error(env_error)}"
                ,
                flush=True,
            )

        grader_scores = info.get("grader_scores", {})
        raw_score = grader_scores.get("composite", 1.0 if info.get("delivery_success") else 0.0)
        end_score = max(0.0, min(1.0, float(raw_score)))
        success = bool(info.get("delivery_success", False))
        termination_reason = str(info.get("termination_reason", "unknown"))
    except Exception:
        success = False
    finally:
        env.close()
        if not start_emitted:
            print(f"[START] task={args.task_name} env={args.benchmark} model={MODEL_NAME}", flush=True)
        reward_text = ",".join(_fmt_float(reward) for reward in rewards)
        print(
            "[END] "
            f"success={_bool_text(success)} "
            f"steps={steps} "
            f"termination_reason={termination_reason} "
            f"score={_fmt_float(end_score)} "
            f"delivery_score={_fmt_float(float(grader_scores.get('delivery', 0.0)))} "
            f"thermal_score={_fmt_float(float(grader_scores.get('thermal', 0.0)))} "
            f"efficiency_score={_fmt_float(float(grader_scores.get('efficiency', 0.0)))} "
            f"rewards={reward_text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
