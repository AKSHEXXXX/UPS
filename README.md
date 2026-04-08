---
title: coldchain-gym
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# UPS

The UPS models pharmaceutical cold-chain logistics where an RL agent dispatches refrigerated vehicles under weather, traffic, and equipment uncertainty.

## What This Project Solves

- Route perishable cargo with dynamic constraints.
- Balance delivery success, thermal integrity, and operational efficiency.
- Stress-test policies with deterministic hazard injection (forced weather and forced breakdown schedules).

## Repository Layout

- `algorithms/`: RL training scripts (`ppo_training.py`) and related algorithm entrypoints.
- `evaluation/`: evaluation contract and shared evaluation utilities (`eval_contract.py`).
- `graders/`: grader execution scripts (`basic_grader_eval.py`).
- `core/`: simulation logic (config, graph, reward, shipment/vehicle/weather systems, grader internals).
- `server/`: OpenEnv/FastAPI adapters and Gymnasium wrapper.
- `tests/`: unit/integration/regression tests.
- `artifacts/visualizations/`: generated plots and debugging visuals.

Canonical entrypoints are organized by folder and should be run directly from those paths.

## Action and Observation Interfaces

Action space is flattened discrete control over vehicle, action type, and node target.

- Action tuple semantics: `[vehicle_index, action_type, target_index]`
- Action types:
  - `0`: WAIT
  - `1`: REROUTE
  - `2`: DIVERT_COLD_DEPOT
  - `3`: SWAP_VEHICLE
  - `4`: EXPEDITE
  - `5`: ABORT

Observations are dictionary-based tensors with fixed shapes (global features, per-vehicle state, per-shipment state), flattened where needed for PPO pipelines.

## Current Model and Grading Behavior

Latest graded snapshot (deterministic, `seed=42`, `eval_training_step=50000`):

- Easy: `0.9974` (PASS, threshold `0.80`)
- Moderate: `0.8899` (PASS, threshold `0.65`)
- Hard: `0.5369` (PASS, threshold `0.50`)
- Extreme: `0.4182` (PASS, threshold `0.35`)
- Overall completed-tier score: `0.7106`

Recent environment and evaluator updates include:

- Deterministic forced weather/breakdown injection for adversarial validation.
- Tiered Easy/Moderate/Hard/Extreme evaluation with threshold and cap rules.
- Delivery-focused diagnostics and exploit checks integrated into training workflow.

## Setup

```bash
pip install -e .
```

Or with uv:

```bash
uv sync
```

## Run Commands

Train PPO phase:

```bash
python algorithms/ppo_training.py --phase 1 --seed 42
```

Run grader evaluation:

```bash
python graders/basic_grader_eval.py --model-path models/ppo_phase3.zip --seed 42 --eval-training-step 50000 --deterministic
```

Run OpenEnv server:

```bash
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run targeted tests:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_eval_contract.py tests/test_graders.py -q
```

## Notes

- Root infra files (`pyproject.toml`, `setup.py`, `Dockerfile`, `openenv.yaml`) remain at root for packaging and container tooling.
- `plan.md` tracks the implementation timeline and current behavior summary.
