---
title: coldchain-gym
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---
## HAVE TO TRAIN ON DIFFERENT MODELS USING HF AND GROQ

# UPS ColdChain-Gym

UPS ColdChain-Gym is a reinforcement learning environment for pharmaceutical cold-chain delivery. The agent controls refrigerated vehicle actions under uncertain traffic, weather shocks, equipment faults, and delivery deadlines.

This project is designed for reproducible evaluation and deployment-ready submission workflows (OpenEnv + Hugging Face Space).

## Problem Statement

Pharma cargo quality depends on both route decisions and thermal safety. A high-performing policy must:

- Deliver shipments on time.
- Keep cargo within temperature bounds.
- Recover from disruptions (weather, breakdowns).
- Avoid short-horizon exploits that inflate reward but fail delivery objectives.

## Environment Description

Each episode simulates a city graph with a hub, cold depots, vehicles, and shipments.

- Graph: weighted network with dynamic edge weights (traffic/weather effects).
- Fleet: refrigerated vehicles with fuel, route state, and cooling status.
- Cargo: heterogeneous shipment types (vaccine, insulin, blood, organ) with strict thermal constraints.
- Events: stochastic and forced disruptions (weather and breakdown schedules).

The environment supports curriculum difficulty and deterministic stress testing for robust policy grading.

## Action Space

Action space is a flattened discrete index:

- `Discrete((n_vehicles + 1) * 6 * n_nodes)`

Flattened action decodes to:

- `[vehicle_index, action_type, target_index]`

Action types:

1. `0` WAIT
2. `1` REROUTE
3. `2` DIVERT_COLD_DEPOT
4. `3` SWAP_VEHICLE
5. `4` EXPEDITE
6. `5` ABORT

Invalid actions are masked through `MaskablePPO` action masks.

## Observation Space

Observation is dictionary-based and flattened for PPO training:

- `global`: shape `(7,)`
- `vehicles`: shape `(n_vehicles, 9 + max_cargo_per_vehicle)`
- `shipments`: shape `(max_shipments, 13)`

Global features include ambient temperature, traffic/weather signals, and episode progress.

## Grading and Current Baseline

Tiered deterministic grading (`easy`, `moderate`, `hard`, `extreme`) combines delivery ratio, thermal integrity, efficiency, speed, and triage robustness.

Latest reference snapshot (`seed=42`, `eval_training_step=50000`):

- Easy: `0.9974` (PASS)
- Moderate: `0.8899` (PASS)
- Hard: `0.5369` (PASS)
- Extreme: `0.4182` (PASS)
- Overall completed-tier score: `0.7106`

## Project Structure

- `algorithms/`: PPO training entrypoints and callbacks.
- `evaluation/`: evaluation contract and deterministic rollout helpers.
- `graders/`: tiered grading CLI.
- `core/`: simulation, reward, graph, shipment, vehicle, and grader internals.
- `server/`: FastAPI/OpenEnv serving layer and Gym wrapper.
- `Scripts/`: utility scripts (`Prevalidation.py`, `inference_repro.py`).
- `tests/`: unit/integration/regression tests.
- `models/`: checkpoints used for training and inference.

## Setup

### Option A: uv (recommended)

```bash
uv sync
```

### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### 1) Train

```bash
python algorithms/ppo_training.py --phase 1 --seed 42
```

### 2) Grade a checkpoint

```bash
python graders/basic_grader_eval.py \
  --model-path models/ppo_phase3.zip \
  --seed 42 \
  --eval-training-step 50000 \
  --deterministic
```

### 3) Reproducible inference report

```bash
python Scripts/inference_repro.py \
  --model-path models/ppo_phase3.zip \
  --seeds 42,101,202,303,404 \
  --eval-training-step 50000 \
  --output-json outputs/inference_repro.json
```

This script prints per-seed tier scores, aggregate statistics, and reproducibility digests.

### 4) Run the OpenEnv server

```bash
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Validation and Deployment

Local validation:

```bash
openenv validate . -v
```

Push to Hugging Face Space:

```bash
openenv push .
```

Current public deployment:

- https://huggingface.co/spaces/AKSHEXXXX/coldchain-gym

## Testing

Run targeted smoke tests:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_eval_contract.py tests/test_graders.py -q
```

## Hackathon Highlights

- Robust RL benchmark for safety-critical logistics.
- Deterministic adversarial grading for repeatable leaderboard-style scoring.
- End-to-end reproducibility: fixed seeds, deterministic eval contract, JSON reporting.
- Deployment-ready OpenEnv package with web interface support.
