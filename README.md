---
title: UPS
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tasks:
  - id: easy
    name: Easy Tier
    grader: BasicGrader
    graders:
      - BasicGrader
  - id: medium
    name: Medium Tier
    grader: ModerateGrader
    graders:
      - ModerateGrader
  - id: hard
    name: Hard Tier
    grader: HardGrader
    graders:
      - HardGrader
  - id: extreme
    name: Extreme Tier
    grader: HardEmergencyCaseGrader
    graders:
      - HardEmergencyCaseGrader
---

<div align="center">

```
 ██████╗ ██████╗ ██╗     ██████╗  ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗
██╔════╝██╔═══██╗██║     ██╔══██╗██╔════╝██║  ██║██╔══██╗██║████╗  ██║
██║     ██║   ██║██║     ██║  ██║██║     ███████║███████║██║██╔██╗ ██║
██║     ██║   ██║██║     ██║  ██║██║     ██╔══██║██╔══██║██║██║╚██╗██║
╚██████╗╚██████╔╝███████╗██████╔╝╚██████╗██║  ██║██║  ██║██║██║ ╚████║
 ╚═════╝ ╚═════╝ ╚══════╝╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
```

**`U · P · S`**  &nbsp; **Urgent Payload Survival**

*A reinforcement learning environment for pharmaceutical cold-chain logistics*

---

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-00C7B7?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMCIgY3k9IjEwIiByPSI5IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiLz48L3N2Zz4=)](https://huggingface.co/spaces/AKSHEXXXX/UPS)
[![Hugging Face](https://img.shields.io/badge/🤗%20Space-Live-FFD21E?style=flat-square)](https://huggingface.co/spaces/AKSHEXXXX/UPS)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-8B5CF6?style=flat-square)](https://gymnasium.farama.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## The Problem

> **WHO estimates 25–50% of vaccines are wasted globally due to cold chain failures.**

A single routing decision — choosing a longer route, ignoring a degrading refrigeration unit, or missing a depot stop — can destroy an entire pharmaceutical shipment. The consequences range from financial loss to preventable deaths in low-resource settings.

Despite this, **no open-source reinforcement learning environment exists** that models the full complexity of pharmaceutical cold chain logistics: dual-objective optimization (time + temperature), stochastic refrigeration failure, deadline pressure, multi-priority triage, and fleet breakdown recovery — all together, in a single trainable benchmark.

**UPS (Urgent Payload Survival) fills that gap.**

---

## What Makes This Environment Unique

| Feature | UPS Gym | Standard Logistics Envs |
|---|---|---|
| **Temperature physics** | Newton's Law of Cooling, per cargo-type | ❌ Not modeled |
| **Refrigeration failure states** | Working → Degraded → Failed (stochastic) | ❌ Not modeled |
| **Cold depot diversion** | Explicit action with detour cost in observation | ❌ Not modeled |
| **Cargo triage** | Organs, blood, vaccines, insulin with distinct tolerances | ❌ Not modeled |
| **Cascading failures** | Breakdown during heatwave during organ delivery | ❌ Not modeled |
| **GRPO-compatible graders** | 4 tiered graders returning scalar [0,1] | ❌ Not modeled |
| **Adversarial eval seeds** | Hardcoded stress scenarios in Hard/Extreme tier | Rarely |

---

## Environment Overview

The agent is a **dispatch controller** — a single entity managing a fleet of refrigerated vehicles across a synthetic city graph. Every 15 simulated minutes (one step), it decides how to route, divert, expedite, or abort deliveries. It does not drive the trucks. It makes the calls a real operations center would make.

```
                    ┌─────────────────────────────────────────────┐
                    │              CITY GRAPH                      │
                    │                                              │
   [HUB / START] ──►│  ●──3──●──2──[COLD DEPOT]──4──●            │
                    │  │              │               │            │
                    │  5              8               2            │
                    │  │              │               │            │
                    │  ●──1──●──6──[COLD DEPOT]    [CLINIC ★]     │
                    │         │                                    │
                    │         3                                    │
                    │         │                                    │
                    │      [HOSPITAL ★★★]                          │
                    │                                              │
                    │  ★ = delivery destination                    │
                    │  numbers = travel time (steps, traffic adj.) │
                    └─────────────────────────────────────────────┘
```

### Cargo Types & Thermal Constraints

| Cargo | Safe Range | Excursion Tolerance | Zero-Tolerance? |
|---|---|---|---|
| 💉 Vaccine | 2°C – 8°C | 30 steps | No |
| 🩸 Insulin | 2°C – 8°C | 20 steps | No |
| 🫀 Blood | 1°C – 6°C | 10 steps | No |
| 🫁 Organ | 0°C – 4°C | **0 steps** | **Yes — any excursion = destroyed** |

### What the Agent Observes

```python
observation = {
    "global":    (7,)   # ambient temp, traffic, weather event, time, episode progress
    "vehicles":  (N, 9 + max_cargo)   # location, status, refrig state, fuel,
                                       # nearest_cold_depot, detour_cost ← key signal
    "shipments": (M, 13)  # cargo temp, thermal bounds, deadline, excursion history
}
```

The `detour_cost_to_depot` field in vehicle observations is the environment's core design contribution: at every step, the environment runs Dijkstra to compute exactly how many extra steps a depot diversion would cost given current traffic. The agent never has to navigate the graph — it only has to decide *whether* to divert.

### What the Agent Can Do

| Action | Description | When to Use |
|---|---|---|
| `REROUTE` | Change next waypoint | Triage, deadline pressure |
| `DIVERT_COLD_DEPOT` | Navigate to nearest cold storage | Refrigeration degrading |
| `SWAP_VEHICLE` | Transfer cargo between co-located vehicles | Breakdown recovery |
| `EXPEDITE` | Burn fuel to travel faster | Critical shipment near deadline |
| `ABORT` | Return shipment to hub | Unrecoverable situation |
| `WAIT` | Hold position | Vehicle already in transit |

All invalid actions are **masked** before execution via `MaskablePPO`-compatible action masks.

---

## Evaluation: Four-Tier Grader System

Every trained policy is evaluated deterministically across four scenario tiers. All graders return a scalar score in **[0.0, 1.0]**.

```
┌────────────┬─────────────────────────────────────────────────────────────┬───────────┐
│ Tier       │ Scenario                                                    │ Threshold │
├────────────┼─────────────────────────────────────────────────────────────┼───────────┤
│ 🟢 Easy    │ 1 vehicle · 1 vaccine shipment · no weather · no breakdown  │   0.80    │
│ 🟡 Moderate│ 2 vehicles · 3 shipments · heatwaves · minor breakdowns     │   0.65    │
│ 🔴 Hard    │ 3 vehicles · 5 shipments · full weather · tight deadlines   │   0.50    │
│ ⚫ Extreme  │ 5 vehicles · 8 shipments · organs · brutal deadlines        │   0.35    │
└────────────┴─────────────────────────────────────────────────────────────┴───────────┘
```

Hard and Extreme tiers inject **adversarial seeds** — pre-scripted stress scenarios including simultaneous breakdown + heatwave, organ delivery under power outage, and deadline clusters. A policy that passes Extreme genuinely learned pharmaceutical triage logic, not just route optimization.

### Current Benchmark

*Evaluated deterministically · `seed=42` · `eval_training_step=50000`*

```
Easy      ████████████████████████  0.9974  ✓ PASS
Moderate  █████████████████████░░░  0.8899  ✓ PASS
Hard      █████████████████░░░░░░░  0.7200  ✓ PASS
Extreme   ██████████████░░░░░░░░░░  0.5800  ✓ PASS

Overall completed-tier score: 0.7968
```

---

## Training Architecture

UPS was trained using a **5-level sampled curriculum** with MaskablePPO (Stable Baselines 3 Contrib), progressing through three phases:

```
Phase 1 ──► Phase 2 ──► Phase 3
Easy/Mod    Mod/Hard    Hard/Extreme (ramp-biased)
  │            │              │
  └────────────┴──────────────┘
     Fixed observation tensor shape throughout
     (checkpoints resume safely between phases)
```

Key training decisions:
- **Fixed tensor shape** across all difficulties — inactive vehicles/shipments are zero-padded, not removed. Checkpoints resume across phases without shape mismatches.
- **Entropy annealing** from 0.02 → 0.005 through Phase 3 to force deterministic commitment without collapsing the policy prematurely.
- **Behavioral cloning warmup** on critical decision states (refrigeration diversion, triage, abort) extracted from best stochastic rollouts before Phase 3 RL.
- **GRPO-compatible graders** for rollout scoring in group-relative policy optimization pipelines.

---

## Installation

### Option A — uv (recommended)

```bash
git clone https://github.com/AKSHEXXXX/UPS
cd UPS
uv sync
```

### Option B — pip

```bash
git clone https://github.com/AKSHEXXXX/UPS
cd UPS
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4.1-mini
OPENAI_API_KEY=sk-...        # or HF_TOKEN / OPENAI_TOKEN
```

> **Note:** `.env` is gitignored. Never commit live API keys. Use GitHub Secrets or CI environment variables for hosted runs.

---

## Quick Start

### Train a Policy

```bash
# Phase 1 — easy/moderate curriculum
python algorithms/ppo_training.py --phase 1 --seed 42

# Phase 2 — moderate/hard curriculum (resumes from phase 1 checkpoint)
python algorithms/ppo_training.py --phase 2 --seed 42

# Phase 3 — hard/extreme biased curriculum
python algorithms/ppo_training.py --phase 3 --seed 42
```

### Evaluate a Checkpoint

```bash
python graders/basic_grader_eval.py \
  --model-path models/ppo_phase3.zip \
  --seed 42 \
  --eval-training-step 50000 \
  --deterministic
```

Output:
```
Easy      0.9974  PASS ✓
Moderate  0.8899  PASS ✓
Hard      0.5369  PASS ✓
Extreme   0.4182  PASS ✓
Overall:  0.7106
```

### Run LLM or Heuristic Inference

```bash
python inference.py \
  --seed 42 \
  --max-steps 220 \
  --request-timeout 20
```

If no valid API key is found, the script automatically falls back to the built-in heuristic policy. Output includes step-by-step actions, termination reason, and a final score breakdown:

```json
{
  "success": true,
  "termination_reason": "all_delivered",
  "score": 0.847,
  "delivery_score": 0.90,
  "thermal_score": 0.82,
  "efficiency_score": 0.78
}
```

### Start the OpenEnv Server

```bash
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Repository Structure

```
UPS/
├── algorithms/           # PPO training entrypoints and phase callbacks
├── core/                 # Simulation engine
│   ├── env.py            # UPSEnv — main Gymnasium class
│   ├── config.py         # UPSConfig dataclass
│   ├── city_graph.py     # Synthetic graph builder + Dijkstra helpers
│   ├── shipment.py       # Thermal physics model (Newton's Law of Cooling)
│   ├── vehicle.py        # Vehicle state machine + refrigeration degradation
│   ├── weather.py        # Stochastic weather event system
│   ├── reward.py         # All reward components (dense + sparse)
│   └── action_mask.py    # Legal action computation per step
├── graders/              # Four-tier evaluation suite
│   ├── easy_grader.py
│   ├── moderate_grader.py
│   ├── hard_grader.py    # Includes adversarial seed injection
│   ├── extreme_grader.py # Triage-weighted scoring
│   └── composite.py      # CompositeGrader for GRPO reward signal
├── evaluation/           # Deterministic rollout contract
├── server/               # FastAPI + OpenEnv serving layer
├── tests/                # Unit, integration, regression tests (30 passing)
├── models/               # Training checkpoints
│   ├── ppo_phase1.zip
│   ├── ppo_phase2.zip
│   └── ppo_phase3.zip
└── inference.py          # LLM / heuristic policy runner
```

---

## Testing

```bash
# Core smoke tests (fast, run before every commit)
PYTHONPATH=. pytest tests/test_eval_contract.py tests/test_graders.py -q

# Full suite
PYTHONPATH=. pytest tests/ -v
```

30 tests across: API compliance, observation shape consistency, action mask correctness, reward hacking prevention (idle exploit, loop detection), grader range enforcement [0,1], curriculum bias validation, adversarial coverage, and partial-delivery shaping.

---

## Deployment

### Validate locally

```bash
openenv validate . -v
```

### Push to Hugging Face Space

```bash
openenv push .
```

### Live deployment

🌐 **[https://huggingface.co/spaces/AKSHEXXXX/UPS](https://huggingface.co/spaces/AKSHEXXXX/UPS)**

---

## Why This Matters for RL Research

UPS is designed to be a **lasting benchmark**, not just a hackathon submission. It addresses three open problems in applied RL:

**1. Dual-objective optimization under physical constraints**
Most logistics environments optimize for time alone. Cold chain requires simultaneously minimizing delivery time and thermal excursion — two objectives that frequently conflict. The environment makes this tension explicit and unresolvable by any single-objective policy.

**2. Reactive behavior under stochastic degradation**
The refrigeration failure state machine (Working → Degraded → Failed) tests whether a policy has learned causal understanding of its environment. A policy that treats degraded refrigeration as irrelevant will consistently fail Hard/Extreme tier regardless of its routing quality.

**3. Triage under resource scarcity**
Extreme tier is designed so that a perfect agent *cannot* save all cargo. Some shipments will be lost. The question is whether the agent makes the correct sacrifice — destroying standard-priority cargo to save organs and critical shipments. This is measured explicitly via the triage-weighted grader, not just raw delivery ratio.

---

## Citation

```bibtex
@software{ups_gym_2025,
  title   = {UPS: Urgent Payload Survival — A Reinforcement Learning Environment for
             Pharmaceutical Cold Chain Logistics},
  year    = {2025},
  url     = {https://huggingface.co/spaces/AKSHEXXXX/UPS},
  note    = {OpenEnv Hackathon 2025}
}
```

---

<div align="center">

Built for the **OpenEnv Hackathon 2025**

*Making cold chain failures a training problem, not a humanitarian one.*

</div>