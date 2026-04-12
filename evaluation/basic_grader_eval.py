from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO

from graders.new_levels import run_full_evaluation


def _load_model(model_path: str) -> MaskablePPO | None:
    if not os.path.exists(model_path):
        return None
    return MaskablePPO.load(model_path)


def _print_level_summary(name: str, summary: dict[str, object]) -> None:
    print(f"\n{name.upper()} GRADER")
    print(f"  Score          : {float(summary['score']):.4f}")
    print(f"  Delivery Score : {float(summary['delivery_score']):.4f}")
    print(f"  Path Score     : {float(summary['path_score']):.4f}")
    print(f"  Thermal Score  : {float(summary['thermal_score']):.4f}")
    print(f"  Safety Score   : {float(summary['safety_score']):.4f}")
    for case in summary.get("cases", []):
        case_name = case.get("case", "case")
        print(
            f"    - {case_name}: delivery={float(case['delivery_score']):.4f}, "
            f"path={float(case['path_score']):.4f}, thermal={float(case['thermal_score']):.4f}, "
            f"safety={float(case.get('safety_score', 0.0)):.4f}"
        )


def _print_full_eval_report(report: dict[str, object]) -> None:
    rows = report.get("rows", [])
    stopped_early = bool(report.get("stopped_early", False))

    print("\n" + "-" * 92)
    print(f"{'Tier':<10}{'Score':>10}{'Threshold':>12}{'Pass':>8}{'Delivery':>12}{'Thermal':>10}{'Eff':>8}{'Speed':>8}{'Triage':>10}")
    print("-" * 92)
    for row in rows:
        print(
            f"{str(row.get('tier', '-')):<10}"
            f"{float(row.get('score', 0.0)):>10.4f}"
            f"{float(row.get('threshold', 0.0)):>12.2f}"
            f"{('PASS' if row.get('pass', False) else 'FAIL'):>8}"
            f"{float(row.get('delivery_ratio', 0.0)):>12.4f}"
            f"{float(row.get('thermal_ratio', 0.0)):>10.4f}"
            f"{float(row.get('efficiency_ratio', 0.0)):>8.4f}"
            f"{float(row.get('speed_ratio', 0.0)):>8.4f}"
            f"{float(row.get('triage_score', 0.0)):>10.4f}"
        )
        if row.get("delivery_cap_applied", False):
            print("  note: moderate cap applied (delivery_ratio < 0.40 => max score 0.50)")
        if row.get("adversarial_cap_applied", False):
            print(
                "  note: hard adversarial cap applied "
                f"(adversarial_delivery_ratio={float(row.get('adversarial_delivery_ratio', 0.0)):.4f} < 0.30 => max score 0.35)"
            )

    print("-" * 92)
    if stopped_early:
        print("Evaluation stopped early after a tier failure.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/ppo_phase3.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-training-step", type=int, default=100000)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--trace-every", type=int, default=0)
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("TIERED GRADER EVALUATION")
    print("=" * 72)

    model = _load_model(args.model_path)
    if model is None:
        for fallback in ["models/ppo_phase2.zip", "models/ppo_phase1.zip"]:
            model = _load_model(fallback)
            if model is not None:
                print(f"Using fallback model: {fallback}")
                break
    if model is None:
        print(f"!!! Error: could not find a model at {args.model_path} or fallback checkpoints.")
        return

    print(f"✓ Loaded model: {args.model_path if os.path.exists(args.model_path) else 'fallback checkpoint'}")

    report = run_full_evaluation(
        model,
        seed=args.seed,
        deterministic=args.deterministic,
        evaluation_training_step=args.eval_training_step,
        trace_every=args.trace_every,
    )
    _print_full_eval_report(report)

    rows = report.get("rows", [])
    overall = sum(float(row.get("score", 0.0)) for row in rows) / max(1, len(rows))
    print("\n" + "-" * 72)
    print(f"OVERALL SCORE (COMPLETED TIERS): {overall:.4f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
