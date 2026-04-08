from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.graders import run_full_evaluation


def _parse_seeds(seeds_text: str) -> list[int]:
    seeds: list[int] = []
    for piece in seeds_text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        seeds.append(int(piece))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _row_by_tier(rows: list[dict[str, Any]], tier: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("tier", "")).lower() == tier.lower():
            return row
    return {}


def _score_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic inference with reproducible tier scores")
    parser.add_argument("--model-path", type=str, default="models/ppo_phase3.zip")
    parser.add_argument("--seeds", type=str, default="42,101,202,303,404")
    parser.add_argument("--eval-training-step", type=int, default=50000)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    seeds = _parse_seeds(args.seeds)
    model = MaskablePPO.load(args.model_path)

    per_seed: list[dict[str, Any]] = []

    print("=" * 78)
    print("REPRODUCIBLE INFERENCE REPORT")
    print("=" * 78)
    print(f"model           : {args.model_path}")
    print(f"deterministic   : True")
    print(f"eval_step       : {args.eval_training_step}")
    print(f"seeds           : {seeds}")

    for seed in seeds:
        report = run_full_evaluation(
            model,
            seed=seed,
            deterministic=True,
            evaluation_training_step=int(args.eval_training_step),
            trace_every=int(args.trace_every),
        )
        rows = report.get("rows", [])

        easy = float(_row_by_tier(rows, "easy").get("score", 0.0))
        moderate = float(_row_by_tier(rows, "moderate").get("score", 0.0))
        hard = float(_row_by_tier(rows, "hard").get("score", 0.0))
        extreme = float(_row_by_tier(rows, "extreme").get("score", 0.0))
        overall = _mean([easy, moderate, hard, extreme])

        seed_result = {
            "seed": int(seed),
            "easy": easy,
            "moderate": moderate,
            "hard": hard,
            "extreme": extreme,
            "overall": overall,
            "stopped_early": bool(report.get("stopped_early", False)),
            "rows": rows,
        }
        seed_result["digest"] = _score_digest(seed_result)
        per_seed.append(seed_result)

        print(
            f"seed={seed:4d} | easy={easy:.4f} moderate={moderate:.4f} "
            f"hard={hard:.4f} extreme={extreme:.4f} overall={overall:.4f} "
            f"digest={seed_result['digest']}"
        )

    summary = {
        "easy_mean": _mean([x["easy"] for x in per_seed]),
        "moderate_mean": _mean([x["moderate"] for x in per_seed]),
        "hard_mean": _mean([x["hard"] for x in per_seed]),
        "extreme_mean": _mean([x["extreme"] for x in per_seed]),
        "overall_mean": _mean([x["overall"] for x in per_seed]),
        "overall_stdev": float(statistics.pstdev([x["overall"] for x in per_seed])) if len(per_seed) > 1 else 0.0,
    }
    summary["digest"] = _score_digest(summary)

    print("-" * 78)
    print(
        f"mean | easy={summary['easy_mean']:.4f} moderate={summary['moderate_mean']:.4f} "
        f"hard={summary['hard_mean']:.4f} extreme={summary['extreme_mean']:.4f} "
        f"overall={summary['overall_mean']:.4f} stdev={summary['overall_stdev']:.4f} "
        f"digest={summary['digest']}"
    )
    print("=" * 78)

    payload = {
        "model_path": args.model_path,
        "deterministic": True,
        "eval_training_step": int(args.eval_training_step),
        "seeds": seeds,
        "summary": summary,
        "per_seed": per_seed,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {output_path}")


if __name__ == "__main__":
    main()
