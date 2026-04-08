# ColdChain Gym (OpenEnv + Gymnasium)

## Environment Description and Motivation

ColdChain Gym models pharmaceutical cold-chain logistics where an RL agent dispatches refrigerated vehicles under uncertainty. The objective is to maximize successful deliveries while maintaining thermal integrity and controlling operational cost.

Why this environment matters:
- Real-world inspired constraints: refrigeration degradation, weather/traffic shifts, depot detours.
- Safety-critical decisions: late or thermally compromised cargo can be destroyed.
- Multi-objective RL benchmark: delivery performance, temperature safety, and efficiency must all be balanced.

The project supports two runtime modes:
- OpenEnv server/client mode for hackathon-style API usage.
- Gymnasium wrapper mode for local RL algorithm training and evaluation.

## Repository Organization

Code is organized by purpose:
- `core/`: OpenEnv core simulation modules (models, reward, physics, masking, graders).
- `server/`: FastAPI/OpenEnv serving layer.
- `coldchain_gym/`: Gymnasium adapter package used by RL pipelines.
- `tests/`: API, behavior, and comprehensive environment validation.

Top-level infra files (`pyproject.toml`, `setup.py`, `openenv.yaml`, `Dockerfile`) remain at repository root intentionally because packaging and OpenEnv tooling expect them there.

## Action Space Definition

Gymnasium action space:
- `MultiDiscrete([n_vehicles + 1, 6, n_nodes])`

Action tuple format:
- `[vehicle_index, action_type, target_index]`

Action types:
- `0`: WAIT
- `1`: REROUTE
- `2`: DIVERT_COLD_DEPOT
- `3`: SWAP_VEHICLE
- `4`: EXPEDITE
- `5`: ABORT

Action masking:
- Illegal actions are masked via `env.action_masks()`.
- If an illegal action is passed, the environment forces WAIT and logs the mask event.

## Observation Space Definition

Gymnasium observation is a dict with fixed-shape tensors:
- `global`: telemetry dict
  - `ambient_temperature`, `time_of_day`, `traffic_multiplier`, `weather_event`, `hub_cold_storage_temp`, `steps_elapsed`, `steps_remaining`
- `vehicles`: array shape `(n_vehicles, 9 + max_cargo_per_vehicle)`
  - location/status/refrigeration/fuel/steps-to-waypoint/onboard cargo slots/depot and destination metrics
- `shipments`: array shape `(max_shipments, 13)`
  - cargo temperature bounds, deadlines, carrier assignment, destination, priority/type, excursion and delivery state

The OpenEnv side returns structured typed models; the Gym wrapper converts them into numpy arrays for RL libraries.

## Task Descriptions and Expected Difficulty

1. Basic dispatch control
- Goal: deliver cargo with minimal destruction under standard conditions.
- Expected difficulty: Easy to Medium.

2. Multi-vehicle coordination under stochastic events
- Goal: coordinate reroutes, depot diversions, and expedites as weather and refrigeration dynamics evolve.
- Expected difficulty: Medium to Hard.

3. High-pressure thermal safety and deadline tradeoffs
- Goal: maintain temperature integrity with limited fuel/time while avoiding costly abort patterns.
- Expected difficulty: Hard.

## Setup and Usage Instructions

### 1) Install

```bash
pip install -e .
```

### 2) Run OpenEnv server locally

```bash
PYTHONPATH=. uvicorn gym.server.app:app --host 0.0.0.0 --port 8000
```

### 3) Use Gymnasium wrapper

```python
from core.config import ColdChainConfig
from server.env import ColdChainEnv

env = ColdChainEnv(config=ColdChainConfig())
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step([0, 0, 0])
```

### 4) Run tests

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_phase11_comprehensive.py tests/test_debug_systems.py -q
```

## Baseline Scores

Baseline scores below were measured on default config over 20 episodes (`seed=0..19`).

Policy definitions:
- `Random legal`: sample uniformly from current legal action mask.
- `First legal`: deterministic first legal action from current mask.

| Policy | Avg Episode Reward | CompositeGrader | BasicGrader | ModerateGrader | HardGrader |
|---|---:|---:|---:|---:|---:|
| Random legal | -141.659 | 0.000 | 0.000 | 0.000 | 0.000 |
| First legal | -25.200 | 0.200 | 0.000 | 0.000 | 0.200 |

Interpretation:
- Naive baselines underperform significantly, leaving clear room for learning-based policies.
- Hard/Composite scores improve slightly with deterministic legality-only behavior, but remain far from robust dispatch quality.

## Notes for Judges

- Import fallback chains were removed to keep imports explicit and deterministic.
- GRPO-specific modules/docs were removed for a cleaner, focused submission.
- The same environment logic is shared across OpenEnv and Gymnasium interfaces.
