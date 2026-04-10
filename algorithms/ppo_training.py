import os
import argparse
import sys
import json
from pathlib import Path
from collections import Counter
from dataclasses import replace
import importlib
from typing import Any, Dict, Optional
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.env import ColdChainEnv, CurriculumWrapper
from core.config import ColdChainConfig
from evaluation.eval_contract import build_eval_env, run_eval_episode
from core.graders import CompositeGrader, DeliverySuccessGrader, run_full_evaluation


def _ensure_gym_version_for_sb3() -> None:
    """SB3 save utilities inspect gym.__version__; our local package layout can shadow gym."""
    try:
        legacy_gym = importlib.import_module("gym")
    except Exception:
        return
    if not hasattr(legacy_gym, "__version__"):
        legacy_gym.__version__ = getattr(gym, "__version__", "0.0")


def _phase_ppo_hyperparams(phase: int) -> dict[str, float | int]:
    if phase == 1:
        return {
            "learning_rate": 1e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 4,
            "ent_coef_start": 0.05,
            "ent_coef_end": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
        }
    if phase == 2:
        return {
            "learning_rate": 7e-5,
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 8,
            "ent_coef_start": 0.05,
            "ent_coef_end": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
        }
    # Laptop-friendly phase-3 profile: keep the sharper PPO settings from phase3.md,
    # but avoid the heavy 4096-step rollout / 512 batch that would spike memory use.
    return {
        "learning_rate": 3e-5,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 12,
        "ent_coef_start": 0.01,
        "ent_coef_end": 0.001,
        "vf_coef": 0.6,
        "gamma": 0.997,
        "gae_lambda": 0.97,
        "clip_range": 0.15,
    }


def _is_phase3_continuation(phase: int, resume_path: str | None) -> bool:
    if int(phase) != 3 or not resume_path:
        return False
    filename = os.path.basename(str(resume_path)).lower()
    return "ppo_phase3" in filename


def _apply_loaded_ppo_hyperparams(model: MaskablePPO, hyperparams: dict[str, float | int]) -> None:
    model.n_steps = int(hyperparams["n_steps"])
    model.batch_size = int(hyperparams["batch_size"])
    model.n_epochs = int(hyperparams["n_epochs"])
    model.ent_coef = float(hyperparams["ent_coef_start"])
    model.vf_coef = float(hyperparams["vf_coef"])
    model.gamma = float(hyperparams["gamma"])
    model.gae_lambda = float(hyperparams["gae_lambda"])
    model.learning_rate = float(hyperparams["learning_rate"])
    model.lr_schedule = FloatSchedule(float(hyperparams["learning_rate"]))
    model.clip_range = FloatSchedule(float(hyperparams["clip_range"]))
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )
    current_lr = float(model.lr_schedule(1.0))
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = current_lr


def _infer_checkpoint_max_steps(resume_path: str) -> Optional[int]:
    """Extract expected max_steps from flattened observation-space bounds in a saved checkpoint."""
    if not resume_path or not os.path.exists(resume_path):
        return None
    try:
        checkpoint_model = MaskablePPO.load(resume_path)
    except Exception:
        return None
    obs_space = getattr(checkpoint_model, "observation_space", None)
    if not isinstance(obs_space, gym.spaces.Box):
        return None
    try:
        high = np.asarray(obs_space.high, dtype=np.float32).reshape(-1)
        if high.size < 7:
            return None
        inferred = int(round(float(high[5])))
        return inferred if inferred > 0 else None
    except Exception:
        return None

def get_mask(env):
    return env.unwrapped.action_masks()

class EntropyAnnealingCallback(BaseCallback):
    """Linearly anneal entropy based on timestep progress."""

    def __init__(
        self,
        start_ent=0.05,
        end_ent=0.01,
        total_steps=120000,
        min_entropy_floor=None,
        competence_callback: "TierAlignedMetricsCallback | None" = None,
        competence_threshold: float = 0.0,
        verbose=0,
    ):
        super().__init__(verbose)
        self.start_ent = float(start_ent)
        self.end_ent = float(end_ent)
        self.total_steps = max(int(total_steps), 1)
        self.min_entropy_floor = float(self.end_ent if min_entropy_floor is None else min_entropy_floor)
        self.competence_callback = competence_callback
        self.competence_threshold = float(competence_threshold)

    def _on_step(self) -> bool:
        if self.competence_callback is not None:
            competence = float(getattr(self.competence_callback, "latest_extreme_stochastic_delivery", 0.0))
            if competence <= self.competence_threshold:
                self.model.ent_coef = max(self.start_ent, self.min_entropy_floor)
                return True
        progress = min(float(self.num_timesteps) / float(self.total_steps), 1.0)
        current_ent = self.start_ent + (self.end_ent - self.start_ent) * progress
        self.model.ent_coef = max(current_ent, self.min_entropy_floor)
        return True


class DeterministicGapCallback(BaseCallback):
    """Track deterministic-vs-stochastic delivery gap throughout training."""

    def __init__(self, config, curriculum_difficulty=5, check_every_steps=5000, n_episodes=6, seed=42, verbose=1):
        super().__init__(verbose)
        self.config = config
        self.curriculum_difficulty = int(curriculum_difficulty)
        self.check_every_steps = max(int(check_every_steps), 1)
        self.n_episodes = max(int(n_episodes), 1)
        self.seed = int(seed)
        self._next_check_step = self.check_every_steps
        self.latest_deterministic_delivery = 0.0
        self.latest_stochastic_delivery = 0.0
        self.latest_gap = 0.0

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_check_step:
            self._run_check(self._next_check_step)
            self._next_check_step += self.check_every_steps
        return True

    def _run_check(self, training_step: int) -> None:
        eval_config = replace(self.config, current_training_step=int(training_step))
        eval_env = build_eval_env(eval_config)
        delivery_by_mode = {}

        try:
            for deterministic in (True, False):
                deliveries = 0
                for episode_index in range(self.n_episodes):
                    result = run_eval_episode(
                        self.model,
                        eval_env,
                        seed=self.seed + 6000 + training_step + episode_index,
                        deterministic=bool(deterministic),
                        curriculum_difficulty=self.curriculum_difficulty,
                        evaluation_training_step=int(training_step),
                        trace_every=0,
                        collect_trajectory=False,
                    )
                    deliveries += int(bool(result.get("delivery_success", False)))
                key = "deterministic" if deterministic else "stochastic"
                delivery_by_mode[key] = deliveries / float(self.n_episodes)
        finally:
            eval_env.close()

        self.latest_deterministic_delivery = float(delivery_by_mode.get("deterministic", 0.0))
        self.latest_stochastic_delivery = float(delivery_by_mode.get("stochastic", 0.0))
        self.latest_gap = max(0.0, self.latest_stochastic_delivery - self.latest_deterministic_delivery)

        print(
            f"[DetVsSto @ {training_step}] det={self.latest_deterministic_delivery:.2%}, "
            f"sto={self.latest_stochastic_delivery:.2%}, gap={self.latest_gap:.2%}"
        )


def _phase3_ramped_weights(progress: float) -> dict[int, float]:
    # Start moderate-heavy; ramp to hard/extreme-heavy through training.
    start = np.asarray([0.05, 0.20, 0.30, 0.30, 0.15], dtype=np.float64)
    end = np.asarray([0.00, 0.00, 0.10, 0.35, 0.55], dtype=np.float64)
    t = float(min(max(progress, 0.0), 1.0))
    values = start + (end - start) * t
    values = values / float(values.sum())
    return {index + 1: float(values[index]) for index in range(5)}


def _phase3_focus_extreme_weights() -> dict[int, float]:
    # Deterministic-alignment continuation pass: maximize hard/extreme exposure.
    return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.20, 5: 0.80}


class CurriculumRampCallback(BaseCallback):
    """Gradually shift phase-3 sampling toward hard/extreme difficulties."""

    def __init__(self, curriculum_env: CurriculumWrapper, total_steps: int, verbose=0):
        super().__init__(verbose)
        self.curriculum_env = curriculum_env
        self.total_steps = max(int(total_steps), 1)

    def _on_step(self) -> bool:
        progress = min(float(self.num_timesteps) / float(self.total_steps), 1.0)
        self.curriculum_env.difficulty_weights = _phase3_ramped_weights(progress)
        return True

class AnnealingCallback(BaseCallback):
    """Anneals the destruction penalty scale over time."""
    def __init__(self, steps_to_full=30000, verbose=0):
        super().__init__(verbose)
        self.steps_to_full = steps_to_full

    def _on_step(self) -> bool:
        # Update current step in config for reward calculation
        if hasattr(self.training_env.unwrapped, 'envs'):
            # Handling vectorized environments
            for env in self.training_env.unwrapped.envs:
                conf = env.unwrapped.config
                conf.current_training_step = self.num_timesteps
        else:
            conf = self.training_env.unwrapped.config
            conf.current_training_step = self.num_timesteps
            
        if self.num_timesteps % 1000 == 0:
            # We can't easily get the last info from training_env if it's vectorized
            # but we can at least see the current step
            pass
        return True

class PhasedMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.lengths = []
        
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(np.mean([info['r'] for info in self.model.ep_info_buffer]))
            self.lengths.append(np.mean([info['l'] for info in self.model.ep_info_buffer]))


class IllegalActionCheckCallback(BaseCallback):
    """Run a deterministic masked validation sweep every N training steps."""

    def __init__(self, config, curriculum_difficulty=3, check_every_steps=5000, n_episodes=3, seed=42, verbose=1):
        super().__init__(verbose)
        self.config = config
        self.curriculum_difficulty = int(curriculum_difficulty)
        self.check_every_steps = max(int(check_every_steps), 1)
        self.n_episodes = max(int(n_episodes), 1)
        self.seed = int(seed)
        self._next_check_step = self.check_every_steps

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_check_step:
            self._run_check(self._next_check_step)
            self._next_check_step += self.check_every_steps
        return True

    def _run_check(self, training_step: int) -> None:
        eval_config = replace(self.config, current_training_step=int(training_step))
        eval_env = build_eval_env(eval_config)

        illegal_actions = 0
        total_actions = 0
        delivered = 0

        try:
            for episode_index in range(self.n_episodes):
                obs, _ = eval_env.reset(
                    seed=self.seed + training_step + episode_index,
                    options={"curriculum_difficulty": self.curriculum_difficulty},
                )
                raw_env = eval_env.unwrapped
                raw_env.config.current_training_step = int(training_step)
                done = False

                while not done:
                    mask = raw_env.action_masks()
                    action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                    action_int = int(action)
                    if action_int >= len(mask) or mask[action_int] == 0:
                        illegal_actions += 1
                        valid_actions = np.flatnonzero(mask).tolist()
                        print(
                            f"🚨 ILLEGAL ACTION DETECTED at step {training_step} "
                            f"episode {episode_index}: action={action_int}, valid_actions={valid_actions}"
                        )

                    obs, _, terminated, truncated, info = eval_env.step(action_int)
                    total_actions += 1
                    done = bool(terminated or truncated)

                delivered += int(bool(info.get("delivery_success", False)))
        finally:
            eval_env.close()

        delivery_rate = delivered / float(self.n_episodes)
        print(
            f"[IllegalActionCheck @ {training_step}] episodes={self.n_episodes}, "
            f"illegal_actions={illegal_actions}/{total_actions}, delivery_rate={delivery_rate:.2%}"
        )


class DeliveryOnlyCallback(BaseCallback):
    """Measure strict true-delivery rate (all_delivered termination) during training."""

    def __init__(self, config, curriculum_difficulty=3, check_every_steps=5000, n_episodes=10, seed=42, verbose=1):
        super().__init__(verbose)
        self.config = config
        self.curriculum_difficulty = int(curriculum_difficulty)
        self.check_every_steps = max(int(check_every_steps), 1)
        self.n_episodes = max(int(n_episodes), 1)
        self.seed = int(seed)
        self._next_check_step = self.check_every_steps
        self.latest_true_delivery_rate = 0.0

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_check_step:
            self.latest_true_delivery_rate = self._run_check(self._next_check_step)
            self._next_check_step += self.check_every_steps
        return True

    def _run_check(self, training_step: int) -> float:
        eval_config = replace(self.config, current_training_step=int(training_step))
        eval_env = build_eval_env(eval_config)
        true_deliveries = 0
        reasons = Counter()

        try:
            for episode_index in range(self.n_episodes):
                result = run_eval_episode(
                    self.model,
                    eval_env,
                    seed=self.seed + 3000 + training_step + episode_index,
                    deterministic=True,
                    curriculum_difficulty=self.curriculum_difficulty,
                    evaluation_training_step=int(training_step),
                    trace_every=0,
                    collect_trajectory=False,
                )
                reason = str(result.get("termination_reason", "unknown"))
                reasons[reason] += 1
                true_deliveries += int(reason == "all_delivered")
        finally:
            eval_env.close()

        rate = true_deliveries / float(self.n_episodes)
        print(
            f"[DeliveryOnly @ {training_step}] true_delivery_rate={rate:.2%}, "
            f"termination_reasons={dict(reasons)}"
        )
        return rate


class TierAlignedMetricsCallback(BaseCallback):
    """Evaluate hard/extreme tiers during phase 3 and expose extreme competence to other callbacks."""

    def __init__(self, config, check_every_steps=10000, seed=42, verbose=1):
        super().__init__(verbose)
        self.config = config
        self.check_every_steps = max(int(check_every_steps), 1)
        self.seed = int(seed)
        self._next_check_step = self.check_every_steps
        self.latest_report: dict[str, Any] = {}
        self.latest_extreme_stochastic_delivery = 0.0

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_check_step:
            self.latest_report = run_tier_aligned_validation(
                self.model,
                self.config,
                seed=self.seed,
                evaluation_training_step=int(self._next_check_step),
                n_episodes=6,
                include_official_grader=False,
            )
            extreme = self.latest_report.get("extreme", {})
            self.latest_extreme_stochastic_delivery = float(extreme.get("stochastic_delivery_rate", 0.0))
            self._next_check_step += self.check_every_steps
        return True


class ExploitDetectorCallback(BaseCallback):
    """Stop training if all_destroyed dominates episode terminations."""

    def __init__(self, check_every=2000, destroy_threshold=0.5, verbose=1):
        super().__init__(verbose)
        self.check_every = max(int(check_every), 1)
        self.destroy_threshold = float(destroy_threshold)
        self.termination_counts = Counter()
        self.total_episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            reason = info.get("termination_reason")
            if reason:
                self.termination_counts[str(reason)] += 1
                self.total_episodes += 1

        if self.num_timesteps % self.check_every == 0 and self.total_episodes > 0:
            destroy_rate = self.termination_counts.get("all_destroyed", 0) / float(self.total_episodes)
            print(f"[{self.num_timesteps}] Termination breakdown:")
            for reason, count in self.termination_counts.items():
                pct = 100.0 * count / float(self.total_episodes)
                print(f"   {reason:<25} {pct:.1f}%")

            if destroy_rate > self.destroy_threshold:
                print(f"\n🚨 EXPLOIT DETECTED — all_destroyed={destroy_rate * 100:.1f}%")
                print(f"   Threshold: {self.destroy_threshold * 100:.0f}%")
                print("   Stopping training. Fix reward architecture before continuing.")
                return False

            self.termination_counts.clear()
            self.total_episodes = 0

        return True


def pre_training_gate(config, seed=42, n_steps=500, curriculum_difficulty=3):
    """Random masked rollout to ensure destruction exploit is unprofitable before training."""
    max_non_delivery_reward = 2.0 + 1.0 + 1.5 + 0.3
    destruction_penalty_at_floor = -50.0 * 0.5
    exploit_proof = abs(destruction_penalty_at_floor) > max_non_delivery_reward

    gate_pass = exploit_proof
    print("\n=== PRE-TRAINING GATE ===")
    print(f"  destruction floor   : {destruction_penalty_at_floor}  {'✅' if exploit_proof else '🚨 FAIL'}")
    print(f"  max non-delivery rew: {max_non_delivery_reward}")
    print(f"  exploit-proof check : {'✅ PASS' if exploit_proof else '🚨 FAIL — increase base penalty'}")

    rng = np.random.default_rng(seed)
    sweep_limit = max(1, int(min(3, curriculum_difficulty)))
    for diff in range(1, sweep_limit + 1):
        env = ColdChainEnv(config=replace(config))
        env = CurriculumWrapper(env)
        env.difficulty = int(diff)
        env.successes_to_advance = 10**9
        env = ActionMasker(env, get_mask)
        env = gym.wrappers.FlattenObservation(env)

        termination_counts = Counter()
        total_episodes = 0
        obs, _ = env.reset(seed=seed + diff)

        for _ in range(int(n_steps)):
            mask = env.unwrapped.action_masks()
            valid_actions = np.flatnonzero(mask)
            if len(valid_actions) == 0:
                break
            action = int(rng.choice(valid_actions))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                reason = str(info.get("termination_reason", "unknown"))
                termination_counts[reason] += 1
                total_episodes += 1
                obs, _ = env.reset()

        env.close()
        destroy_rate = termination_counts.get("all_destroyed", 0) / float(max(total_episodes, 1))
        passed = destroy_rate < 0.1
        gate_pass = gate_pass and passed
        print(f"  difficulty {diff}: all_destroyed rate={destroy_rate * 100:.1f}% {'✅' if passed else '🚨 FAIL'}")
        print(f"    terminations: {dict(termination_counts)}")

    print(f"\n  GATE: {'✅ PASS — safe to train' if gate_pass else '🚨 FAIL — fix reward before training'}")
    return gate_pass


def run_termination_probe(model, config, seed=42, n_episodes=10, curriculum_difficulty=3):
    print(f"\n[Termination Probe] episodes={n_episodes}, difficulty={curriculum_difficulty}")
    probe_env = build_eval_env(config)
    reasons = Counter()
    true_deliveries = 0

    try:
        eval_training_step = max(int(model.num_timesteps), int(config.penalty_anneal_steps))
        for episode_index in range(n_episodes):
            result = run_eval_episode(
                model,
                probe_env,
                seed=seed + 9000 + episode_index,
                deterministic=True,
                curriculum_difficulty=curriculum_difficulty,
                evaluation_training_step=eval_training_step,
                trace_every=0,
                collect_trajectory=False,
            )
            reason = str(result.get("termination_reason", "unknown"))
            reasons[reason] += 1
            true_deliveries += int(reason == "all_delivered")
            print(
                f"  ep={episode_index:02d} steps={result.get('steps', -1):3d} "
                f"reward={result.get('total_reward', 0.0):7.2f} reason={reason}"
            )
    finally:
        probe_env.close()

    print(f"  termination_reason_counts={dict(reasons)}")
    print(f"  true_delivery_rate={true_deliveries / float(n_episodes):.2%}")
    return dict(reasons)


def _row_by_tier(rows, tier_name):
    for row in rows:
        if str(row.get("tier", "")).lower() == str(tier_name).lower():
            return row
    return {}


def robust_tier_score(report):
    rows = report.get("rows", [])
    easy = float(_row_by_tier(rows, "easy").get("score", 0.0))
    moderate = float(_row_by_tier(rows, "moderate").get("score", 0.0))
    hard = float(_row_by_tier(rows, "hard").get("score", 0.0))
    extreme = float(_row_by_tier(rows, "extreme").get("score", 0.0))
    # Bias model quality toward difficult conditions.
    return 0.10 * easy + 0.20 * moderate + 0.35 * hard + 0.35 * extreme


def _base_training_space_config(phase: int) -> ColdChainConfig:
    config_kwargs = dict(
        n_vehicles=5,
        n_nodes=24,
        n_shipments=8,
        max_shipments=8,
        max_steps=220,
        penalty_anneal_steps=30000,
        penalty_initial_scale=0.10,
        weather_events_enabled=False,
        breakdown_probability=0.0,
        refrigeration_degradation_prob=0.0,
    )
    if phase >= 2:
        config_kwargs.update(
            weather_events_enabled=True,
            breakdown_probability=0.004,
            refrigeration_degradation_prob=0.008,
            penalty_anneal_steps=40000,
        )
    if phase >= 3:
        config_kwargs.update(
            max_steps=240,
            breakdown_probability=0.012,
            refrigeration_degradation_prob=0.015,
            penalty_anneal_steps=60000,
            penalty_initial_scale=0.15,
            enable_difficulty_reward_normalization=True,
            refrigeration_reaction_bonus=0.45,
            refrigeration_reaction_penalty=0.22,
        )
    return ColdChainConfig(**config_kwargs)


def _scenario_options(**kwargs: Any) -> Dict[str, Any]:
    return dict(kwargs)


def _training_scenario_library() -> dict[int, list[dict[str, Any]]]:
    return {
        1: [
            _scenario_options(
                scenario_family="easy_route_a",
                active_vehicle_count=1,
                active_shipment_count=1,
                vehicle_start_nodes=[1],
                shipment_destinations=[8],
                shipment_cargo_types=["vaccine"],
                shipment_cargo_temps=[4.0],
                shipment_assignments=[0],
                shipment_deadlines=[140],
                load_shipments_on_start=True,
            ),
            _scenario_options(
                scenario_family="easy_route_b",
                active_vehicle_count=1,
                active_shipment_count=1,
                vehicle_start_nodes=[2],
                shipment_destinations=[9],
                shipment_cargo_types=["insulin"],
                shipment_cargo_temps=[4.2],
                shipment_assignments=[0],
                shipment_deadlines=[145],
                load_shipments_on_start=True,
            ),
        ],
        2: [
            _scenario_options(
                scenario_family="moderate_multi_drop",
                active_vehicle_count=2,
                active_shipment_count=3,
                vehicle_start_nodes=[1, 4],
                shipment_destinations=[11, 12, 9],
                shipment_cargo_types=["vaccine", "insulin", "blood"],
                shipment_cargo_temps=[4.0, 4.1, 5.2],
                shipment_assignments=[0, 1, 0],
                shipment_deadlines=[170, 180, 175],
                load_shipments_on_start=True,
            ),
            _scenario_options(
                scenario_family="moderate_heatwave",
                active_vehicle_count=2,
                active_shipment_count=3,
                vehicle_start_nodes=[2, 5],
                shipment_destinations=[10, 13, 8],
                shipment_cargo_types=["blood", "vaccine", "insulin"],
                shipment_cargo_temps=[5.0, 4.0, 4.1],
                shipment_assignments=[0, 1, 1],
                shipment_deadlines=[165, 185, 178],
                forced_weather_only=True,
                forced_weather_events=[{"step": 25, "event": "HEATWAVE", "duration": 10}],
                load_shipments_on_start=True,
            ),
        ],
        3: [
            _scenario_options(
                scenario_family="medium_storm_first",
                active_vehicle_count=2,
                active_shipment_count=3,
                vehicle_start_nodes=[1, 3],
                shipment_destinations=[12, 13, 15],
                shipment_cargo_types=["vaccine", "blood", "organ"],
                shipment_cargo_temps=[4.0, 5.1, 2.8],
                shipment_priorities=[1, 1, 2],
                shipment_assignments=[0, 1, 0],
                shipment_deadlines=[145, 150, 140],
                forced_weather_only=True,
                forced_weather_events=[{"step": 18, "event": "STORM", "duration": 10}],
                load_shipments_on_start=True,
            ),
            _scenario_options(
                scenario_family="medium_heatwave_late",
                active_vehicle_count=2,
                active_shipment_count=3,
                vehicle_start_nodes=[2, 6],
                shipment_destinations=[14, 16, 17],
                shipment_cargo_types=["organ", "blood", "vaccine"],
                shipment_cargo_temps=[2.6, 5.3, 4.0],
                shipment_priorities=[2, 1, 0],
                shipment_assignments=[0, 1, 1],
                shipment_deadlines=[150, 155, 165],
                forced_weather_only=True,
                forced_weather_events=[{"step": 35, "event": "HEATWAVE", "duration": 12}],
                load_shipments_on_start=True,
            ),
        ],
        4: [
            _scenario_options(
                scenario_family="hard_clean",
                active_vehicle_count=3,
                active_shipment_count=5,
                vehicle_start_nodes=[1, 2, 4],
                shipment_destinations=[13, 14, 15, 10, 11],
                shipment_cargo_types=["vaccine", "insulin", "blood", "vaccine", "blood"],
                shipment_cargo_temps=[4.0, 4.3, 5.1, 4.2, 5.0],
                shipment_assignments=[0, 1, 2, 0, 1],
                shipment_deadlines=[90, 95, 100, 85, 92],
                load_shipments_on_start=True,
            ),
            _scenario_options(
                scenario_family="hard_heatwave",
                active_vehicle_count=3,
                active_shipment_count=5,
                vehicle_start_nodes=[1, 2, 4],
                shipment_destinations=[13, 14, 15, 10, 11],
                shipment_cargo_types=["vaccine", "insulin", "blood", "vaccine", "blood"],
                shipment_cargo_temps=[4.0, 4.3, 5.1, 4.2, 5.0],
                shipment_assignments=[0, 1, 2, 0, 1],
                shipment_deadlines=[90, 95, 100, 85, 92],
                forced_weather_only=True,
                forced_weather_events=[{"step": 35, "event": "HEATWAVE", "duration": 12}],
                load_shipments_on_start=True,
            ),
            _scenario_options(
                scenario_family="hard_adversarial_breakdown",
                active_vehicle_count=3,
                active_shipment_count=5,
                vehicle_start_nodes=[1, 2, 4],
                shipment_destinations=[13, 14, 15, 10, 11],
                shipment_cargo_types=["vaccine", "insulin", "blood", "vaccine", "blood"],
                shipment_cargo_temps=[4.0, 4.3, 5.1, 4.2, 5.0],
                shipment_assignments=[0, 1, 2, 0, 1],
                shipment_deadlines=[90, 95, 100, 85, 92],
                forced_weather_only=True,
                forced_weather_events=[
                    {"step": 1, "event": "STORM", "duration": 18},
                    {"step": 25, "event": "HEATWAVE", "duration": 14},
                ],
                forced_breakdowns=[{"step": 12, "vehicle_ids": [0]}],
                load_shipments_on_start=True,
            ),
        ],
        5: [
            _scenario_options(
                scenario_family="extreme_clean",
                active_vehicle_count=5,
                active_shipment_count=8,
                vehicle_start_nodes=[1, 2, 3, 4, 5],
                shipment_destinations=[18, 19, 20, 21, 22, 16, 17, 23],
                shipment_cargo_types=["organ", "organ", "blood", "blood", "vaccine", "insulin", "blood", "vaccine"],
                shipment_cargo_temps=[2.5, 2.8, 5.4, 5.3, 4.1, 4.2, 5.2, 4.0],
                shipment_priorities=[2, 2, 2, 1, 1, 0, 1, 0],
                shipment_assignments=[0, 1, 2, 3, 4, 0, 1, 2],
                shipment_deadlines=[110, 112, 145, 150, 160, 170, 155, 175],
                load_shipments_on_start=True,
            ),
            _scenario_options(
                scenario_family="extreme_weather_stack",
                active_vehicle_count=5,
                active_shipment_count=8,
                vehicle_start_nodes=[1, 2, 3, 4, 5],
                shipment_destinations=[18, 19, 20, 21, 22, 16, 17, 23],
                shipment_cargo_types=["organ", "organ", "blood", "blood", "vaccine", "insulin", "blood", "vaccine"],
                shipment_cargo_temps=[2.5, 2.8, 5.4, 5.3, 4.1, 4.2, 5.2, 4.0],
                shipment_priorities=[2, 2, 2, 1, 1, 0, 1, 0],
                shipment_assignments=[0, 1, 2, 3, 4, 0, 1, 2],
                shipment_deadlines=[110, 112, 145, 150, 160, 170, 155, 175],
                forced_weather_only=True,
                forced_weather_events=[
                    {"step": 0, "event": "STORM", "duration": 20},
                    {"step": 40, "event": "HEATWAVE", "duration": 18},
                ],
                load_shipments_on_start=True,
            ),
            _scenario_options(
                scenario_family="extreme_adversarial_breakdowns",
                active_vehicle_count=5,
                active_shipment_count=8,
                vehicle_start_nodes=[1, 2, 3, 4, 5],
                shipment_destinations=[18, 19, 20, 21, 22, 16, 17, 23],
                shipment_cargo_types=["organ", "organ", "blood", "blood", "vaccine", "insulin", "blood", "vaccine"],
                shipment_cargo_temps=[2.5, 2.8, 5.4, 5.3, 4.1, 4.2, 5.2, 4.0],
                shipment_priorities=[2, 2, 2, 1, 1, 0, 1, 0],
                shipment_assignments=[0, 1, 2, 3, 4, 0, 1, 2],
                shipment_deadlines=[110, 112, 145, 150, 160, 170, 155, 175],
                forced_weather_only=True,
                forced_weather_events=[
                    {"step": 0, "event": "STORM", "duration": 20},
                    {"step": 40, "event": "HEATWAVE", "duration": 18},
                ],
                forced_breakdowns=[{"step": 18, "vehicle_ids": [2]}, {"step": 45, "vehicle_ids": [4]}],
                load_shipments_on_start=True,
            ),
        ],
    }


def _difficulty_name(difficulty: int) -> str:
    return {
        1: "easy",
        2: "moderate",
        3: "medium",
        4: "hard",
        5: "extreme",
    }.get(int(difficulty), f"difficulty_{difficulty}")


def _scenario_family_name(reset_options: dict[str, Any] | None, difficulty: int) -> str:
    if reset_options and reset_options.get("scenario_family"):
        return str(reset_options["scenario_family"])
    return f"{_difficulty_name(difficulty)}_default"


def _active_shipment_statuses(result: dict[str, Any]) -> list[dict[str, Any]]:
    per_shipment = result.get("final_info", {}).get("per_shipment_status") if result.get("final_info") else None
    if not per_shipment:
        per_shipment = result.get("per_shipment_status")
    if not per_shipment:
        return []
    active = []
    for shipment in per_shipment.values():
        if bool(shipment.get("is_active", True)):
            active.append(dict(shipment))
    return active


def _delivery_counts(result: dict[str, Any]) -> tuple[int, int]:
    active_shipments = _active_shipment_statuses(result)
    if not active_shipments:
        return (0, 0)
    delivered = sum(1 for shipment in active_shipments if bool(shipment.get("is_delivered", False)))
    return delivered, len(active_shipments)


def _normalized_episode_quality(result: dict[str, Any], difficulty: int) -> float:
    delivered, total = _delivery_counts(result)
    delivered_ratio = float(delivered) / float(max(total, 1))
    reward = float(result.get("total_reward", 0.0))
    grader_scores = result.get("grader_scores", {}) or {}
    composite = float(grader_scores.get("composite", 0.0))
    thermal = float(grader_scores.get("thermal", 0.0))
    difficulty_boost = 1.0 + 0.20 * max(int(difficulty) - 3, 0)
    success_bonus = 2.0 if bool(result.get("delivery_success", False)) else 0.0
    return difficulty_boost * (success_bonus + 1.5 * delivered_ratio + composite + 0.5 * thermal + 0.01 * reward)


def _select_harvest_candidates(
    tier_results: list[dict[str, Any]],
    difficulty: int,
    top_percentile: float = 0.35,
) -> list[dict[str, Any]]:
    if not tier_results:
        return []
    scored = sorted(
        tier_results,
        key=lambda result: _normalized_episode_quality(result, difficulty),
        reverse=True,
    )
    successful = [result for result in scored if bool(result.get("delivery_success", False))]
    keep_count = max(1, int(np.ceil(len(scored) * float(top_percentile))))
    selected = successful[:]
    for result in scored[:keep_count]:
        if result not in selected:
            selected.append(result)
    return selected


def _rank_of_action(probabilities: np.ndarray, action: int) -> int:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    action = int(action)
    if action < 0 or action >= scores.size:
        return scores.size
    return 1 + int(np.sum(scores > scores[action]))


def _classify_failure(result: dict[str, Any]) -> str:
    termination = str(result.get("termination_reason", "unknown"))
    if termination == "all_destroyed":
        return "all_destroyed"
    if termination == "max_steps":
        delivered, total = _delivery_counts(result)
        ratio = float(delivered) / float(max(total, 1))
        if ratio >= 0.5:
            return "timeout_many_delivered"
        return "timeout_few_delivered"

    masked_ratio = 0.0
    steps = max(int(result.get("steps", 0)), 1)
    final_info = result.get("final_info", {}) or {}
    illegal_actions = int(final_info.get("illegal_action_count", 0))
    masked_ratio = float(illegal_actions) / float(steps)
    if masked_ratio >= 0.15:
        return "illegal_masked_action_heavy"

    vehicles = final_info.get("per_vehicle_status", {}) if final_info else {}
    if any(
        int(vehicle.get("refrig_status", 0)) != 0 and len(vehicle.get("shipments_onboard", [])) > 0
        for vehicle in vehicles.values()
    ):
        return "refrigeration_mishandling"

    return termination


def _top2_action_snapshot(model, obs: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(obs))
        dist = model.policy.get_distribution(obs_tensor, action_masks=np.asarray(mask))
        probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
    top2 = np.argsort(probs)[-2:][::-1]
    return {
        "top_actions": [int(idx) for idx in top2.tolist()],
        "top_probs": [float(probs[idx]) for idx in top2.tolist()],
    }


def harvest_imitation_dataset(
    model,
    config: ColdChainConfig,
    *,
    seed: int,
    episodes_per_scenario: int = 4,
    top_percentile: float = 0.35,
    extreme_focus: bool = False,
    artifact_path: str | None = None,
) -> dict[str, Any]:
    eval_step = max(int(model.num_timesteps), int(config.penalty_anneal_steps))
    library = _training_scenario_library()
    tier_weights = {4: 1.0, 5: 1.75}
    if bool(extreme_focus):
        tier_weights = {4: 0.7, 5: 3.0}
    selected_results: dict[int, list[dict[str, Any]]] = {4: [], 5: []}

    for difficulty in (4, 5):
        tier_rollouts: list[dict[str, Any]] = []
        for scenario_index, reset_options in enumerate(library[difficulty]):
            eval_env = build_eval_env(config)
            try:
                for episode_index in range(int(episodes_per_scenario)):
                    rollout = run_eval_episode(
                        model,
                        eval_env,
                        seed=seed + difficulty * 1000 + scenario_index * 100 + episode_index,
                        deterministic=False,
                        curriculum_difficulty=difficulty,
                        evaluation_training_step=eval_step,
                        reset_options=reset_options,
                        collect_trajectory=True,
                    )
                    rollout["scenario_family"] = _scenario_family_name(reset_options, difficulty)
                    rollout["final_info"] = dict(eval_env.unwrapped._last_info)
                    rollout["grader_scores"] = dict(rollout["final_info"].get("grader_scores", {}))
                    tier_rollouts.append(rollout)
            finally:
                eval_env.close()
        selected_results[difficulty] = _select_harvest_candidates(
            tier_rollouts,
            difficulty=difficulty,
            top_percentile=top_percentile,
        )

    obs_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    mask_rows: list[np.ndarray] = []
    weight_rows: list[float] = []
    tier_rows: list[int] = []
    family_rows: list[str] = []
    action_rank_rows: list[int] = []
    action_prob_rows: list[float] = []
    summary: dict[str, Any] = {"episodes": {}, "sample_count": 0}

    for difficulty, results in selected_results.items():
        difficulty_name = _difficulty_name(difficulty)
        summary["episodes"][difficulty_name] = {
            "selected_episodes": len(results),
            "successful_episodes": sum(int(bool(result.get("delivery_success", False))) for result in results),
        }
        for result in results:
            quality = _normalized_episode_quality(result, difficulty)
            trajectory = list(result.get("trajectory", []))
            for step in trajectory:
                obs = np.asarray(step["obs"], dtype=np.float32)
                action = int(step["action"])
                mask = np.asarray(step["mask"], dtype=np.int8)
                with torch.no_grad():
                    probs = model.policy.get_distribution(
                        model.policy.obs_to_tensor(obs)[0],
                        action_masks=np.asarray(mask),
                    ).distribution.probs.detach().cpu().numpy().reshape(-1)
                obs_rows.append(obs)
                action_rows.append(action)
                mask_rows.append(mask)
                tier_rows.append(int(difficulty))
                family_rows.append(str(result.get("scenario_family", difficulty_name)))
                weight_rows.append(float(tier_weights[difficulty] * max(quality, 0.05)))
                action_rank_rows.append(_rank_of_action(probs, action))
                action_prob_rows.append(float(probs[action]))

    if not obs_rows:
        return {
            "obs": np.zeros((0,), dtype=np.float32),
            "actions": np.zeros((0,), dtype=np.int64),
            "masks": np.zeros((0,), dtype=np.int8),
            "weights": np.zeros((0,), dtype=np.float32),
            "tiers": np.zeros((0,), dtype=np.int64),
            "families": np.asarray([], dtype=object),
            "action_ranks": np.zeros((0,), dtype=np.int64),
            "action_probs": np.zeros((0,), dtype=np.float32),
            "summary": summary,
            "artifact_path": artifact_path,
        }

    dataset = {
        "obs": np.stack(obs_rows).astype(np.float32),
        "actions": np.asarray(action_rows, dtype=np.int64),
        "masks": np.stack(mask_rows).astype(np.int8),
        "weights": np.asarray(weight_rows, dtype=np.float32),
        "tiers": np.asarray(tier_rows, dtype=np.int64),
        "families": np.asarray(family_rows, dtype=object),
        "action_ranks": np.asarray(action_rank_rows, dtype=np.int64),
        "action_probs": np.asarray(action_prob_rows, dtype=np.float32),
        "summary": summary,
        "artifact_path": artifact_path,
    }
    dataset["summary"]["sample_count"] = int(dataset["actions"].shape[0])

    if artifact_path:
        os.makedirs(os.path.dirname(artifact_path) or ".", exist_ok=True)
        np.savez_compressed(
            artifact_path,
            obs=dataset["obs"],
            actions=dataset["actions"],
            masks=dataset["masks"],
            weights=dataset["weights"],
            tiers=dataset["tiers"],
            families=dataset["families"],
            action_ranks=dataset["action_ranks"],
            action_probs=dataset["action_probs"],
        )

    return dataset


def _load_imitation_dataset(path: str) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    return {
        "obs": np.asarray(payload["obs"], dtype=np.float32),
        "actions": np.asarray(payload["actions"], dtype=np.int64),
        "masks": np.asarray(payload["masks"], dtype=np.int8),
        "weights": np.asarray(payload["weights"], dtype=np.float32),
        "tiers": np.asarray(payload["tiers"], dtype=np.int64),
        "families": np.asarray(payload["families"], dtype=object),
        "action_ranks": np.asarray(payload["action_ranks"], dtype=np.int64),
        "action_probs": np.asarray(payload["action_probs"], dtype=np.float32),
    }


def run_behavior_cloning_warmstart(
    model,
    dataset: dict[str, Any],
    *,
    epochs: int = 3,
    batch_size: int = 128,
) -> dict[str, Any]:
    if int(dataset["actions"].shape[0]) == 0:
        return {"epochs": 0, "loss": 0.0, "samples": 0}

    model.policy.set_training_mode(True)
    obs_array = np.asarray(dataset["obs"], dtype=np.float32)
    actions = torch.as_tensor(dataset["actions"], device=model.device, dtype=torch.long)
    masks = torch.as_tensor(np.asarray(dataset["masks"], dtype=np.float32), device=model.device)
    weights = torch.as_tensor(np.asarray(dataset["weights"], dtype=np.float32), device=model.device)
    num_samples = int(actions.shape[0])
    index_array = np.arange(num_samples)
    latest_loss = 0.0

    for _ in range(max(int(epochs), 1)):
        np.random.shuffle(index_array)
        for start in range(0, num_samples, int(batch_size)):
            batch_indices = index_array[start : start + int(batch_size)]
            batch_obs, _ = model.policy.obs_to_tensor(obs_array[batch_indices])
            batch_actions = actions[batch_indices]
            batch_masks = masks[batch_indices]
            batch_weights = weights[batch_indices]
            dist = model.policy.get_distribution(batch_obs, action_masks=batch_masks)
            logits = dist.distribution.logits
            losses = F.cross_entropy(logits, batch_actions, reduction="none")
            loss = torch.sum(losses * batch_weights) / torch.clamp(batch_weights.sum(), min=1e-6)
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            model.policy.optimizer.step()
            latest_loss = float(loss.detach().cpu().item())

    model.policy.set_training_mode(False)
    return {"epochs": int(epochs), "loss": float(latest_loss), "samples": int(num_samples)}


def _dataset_rank_summary(dataset: dict[str, Any]) -> dict[str, float]:
    ranks = np.asarray(dataset.get("action_ranks", []), dtype=np.int64)
    if ranks.size == 0:
        return {"top1": 0.0, "top2": 0.0, "lower": 0.0}
    return {
        "top1": float(np.mean(ranks == 1)),
        "top2": float(np.mean(ranks == 2)),
        "lower": float(np.mean(ranks > 2)),
    }


def _flat_action_index(vehicle_index: int, action_type: int, target_index: int, n_nodes: int) -> int:
    return int(vehicle_index) * (6 * int(n_nodes)) + int(action_type) * int(n_nodes) + int(target_index)


def _decode_action_type(action_index: int, n_nodes: int) -> int:
    return int((int(action_index) % (6 * int(n_nodes))) // int(n_nodes))


def _status_lookup(status_map: dict[str, Any], key: int) -> dict[str, Any]:
    if not isinstance(status_map, dict):
        return {}
    if key in status_map:
        return dict(status_map[key])
    key_str = str(int(key))
    if key_str in status_map:
        return dict(status_map[key_str])
    return {}


def _first_valid_action(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask)
    return int(valid[0]) if len(valid) else 0


def _greedy_expert_flat_action(raw_env, mask: np.ndarray) -> int:
    cfg = raw_env.config
    n_nodes = int(cfg.n_nodes)
    depot_nodes = list(getattr(raw_env.graph, "graph", {}).get("cold_depot_nodes", [])) if raw_env.graph is not None else []

    for vehicle in raw_env.vehicles:
        if not bool(getattr(vehicle, "is_active", True)):
            continue
        status_value = int(getattr(getattr(vehicle, "status", 0), "value", getattr(vehicle, "status", 0)))
        if status_value == 3:
            continue

        onboard = [int(sid) for sid in getattr(vehicle, "shipments_onboard", []) if int(sid) >= 0]
        refrig_value = int(getattr(getattr(vehicle, "refrig_status", 0), "value", getattr(vehicle, "refrig_status", 0)))
        nearest_depot = int(getattr(vehicle, "nearest_cold_depot_node", depot_nodes[0] if depot_nodes else 0))
        detour_cost = float(getattr(vehicle, "detour_cost_to_depot", 9999.0))

        # Rule 1: failed refrigeration + active cargo onboard => divert now.
        if refrig_value == 2 and onboard:
            candidate = _flat_action_index(vehicle.id, 2, nearest_depot, n_nodes)
            if candidate < len(mask) and int(mask[candidate]) == 1:
                return candidate

        # Rule 2: degraded refrigeration and cheap diversion => divert.
        if refrig_value == 1 and onboard and detour_cost < 8.0:
            candidate = _flat_action_index(vehicle.id, 2, nearest_depot, n_nodes)
            if candidate < len(mask) and int(mask[candidate]) == 1:
                return candidate

        # Rule 3: deadline pressure => expedite when possible else reroute direct.
        for shipment_id in onboard:
            if shipment_id >= len(raw_env.shipments):
                continue
            shipment = raw_env.shipments[shipment_id]
            if bool(getattr(shipment, "is_delivered", False)) or bool(getattr(shipment, "is_destroyed", False)):
                continue
            if float(getattr(shipment, "time_to_deadline", 9999)) < 10.0:
                expedite = _flat_action_index(vehicle.id, 4, 0, n_nodes)
                if expedite < len(mask) and int(mask[expedite]) == 1:
                    return expedite
                reroute = _flat_action_index(vehicle.id, 1, int(getattr(shipment, "destination_node", 0)), n_nodes)
                if reroute < len(mask) and int(mask[reroute]) == 1:
                    return reroute

        # Rule 4: idle vehicle with pending shipments => reroute to highest priority destination.
        if status_value == 0:
            undelivered = [
                s
                for s in raw_env.shipments
                if bool(getattr(s, "is_active", True))
                and not bool(getattr(s, "is_delivered", False))
                and not bool(getattr(s, "is_destroyed", False))
            ]
            if undelivered:
                target = max(undelivered, key=lambda s: int(getattr(s, "priority", 0)))
                reroute = _flat_action_index(vehicle.id, 1, int(getattr(target, "destination_node", 0)), n_nodes)
                if reroute < len(mask) and int(mask[reroute]) == 1:
                    return reroute

    # Rule 5: default wait/no-op.
    wait_noop = _flat_action_index(cfg.n_vehicles, 0, 0, n_nodes)
    if wait_noop < len(mask) and int(mask[wait_noop]) == 1:
        return wait_noop
    return _first_valid_action(mask)


def harvest_expert_trajectories(
    config: ColdChainConfig,
    *,
    seed: int,
    n_episodes: int = 500,
    min_delivery_score: float = 0.40,
) -> dict[str, Any]:
    library = _training_scenario_library()
    all_scores: list[float] = []
    kept_results: list[dict[str, Any]] = []

    for episode_index in range(max(int(n_episodes), 1)):
        eval_env = build_eval_env(config)
        difficulty = 5
        reset_options = dict(library[difficulty][episode_index % len(library[difficulty])])
        try:
            obs, _ = eval_env.reset(
                seed=seed + 7000 + episode_index,
                options={"curriculum_difficulty": difficulty, **reset_options},
            )
            done = False
            total_reward = 0.0
            trajectory: list[dict[str, Any]] = []
            final_info: dict[str, Any] = {}
            while not done:
                raw_env = eval_env.unwrapped
                mask = np.asarray(raw_env.action_masks(), dtype=np.int8)
                action_int = _greedy_expert_flat_action(raw_env, mask)
                pre_obs = np.asarray(obs, dtype=np.float32)
                next_obs, reward, terminated, truncated, info = eval_env.step(int(action_int))
                trajectory.append(
                    {
                        "obs": pre_obs,
                        "action": int(action_int),
                        "mask": np.asarray(mask, dtype=np.int8),
                        "next_obs": np.asarray(next_obs, dtype=np.float32),
                        "reward": float(reward),
                        "info": dict(info),
                    }
                )
                total_reward += float(reward)
                final_info = dict(info)
                obs = next_obs
                done = bool(terminated or truncated)

            delivery_score = float(DeliverySuccessGrader(trajectory).score())
            all_scores.append(delivery_score)
            if delivery_score >= float(min_delivery_score):
                kept_results.append(
                    {
                        "delivery_score": delivery_score,
                        "composite_score": float(CompositeGrader(trajectory).score()),
                        "trajectory": trajectory,
                        "delivery_success": bool(final_info.get("delivery_success", False)),
                        "termination_reason": str(final_info.get("termination_reason", "unknown")),
                        "final_info": final_info,
                        "scenario_family": str(reset_options.get("scenario_family", "extreme_unknown")),
                        "total_reward": float(total_reward),
                    }
                )
        finally:
            eval_env.close()

    if all_scores:
        print(
            "[Phase4 Harvest] "
            f"kept={len(kept_results)}/{len(all_scores)}, "
            f"delivery_score(min/mean/max)=({min(all_scores):.3f}/{float(np.mean(all_scores)):.3f}/{max(all_scores):.3f})"
        )
    return {"results": kept_results, "scores": all_scores}


def extract_critical_states(
    harvest_results: list[dict[str, Any]],
    *,
    n_nodes: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"diversion": [], "triage": [], "abort": []}

    for episode in harvest_results:
        for step in episode.get("trajectory", []):
            info = dict(step.get("info", {}))
            action = int(step.get("action", 0))
            action_type = _decode_action_type(action, n_nodes)
            vehicle_index = int(action // (6 * n_nodes))

            vehicle_status = _status_lookup(info.get("per_vehicle_status", {}), vehicle_index)
            refrig_status = int(vehicle_status.get("refrig_status", 0))

            shipment_status_map = info.get("per_shipment_status", {}) or {}
            critical_undelivered = 0
            for shipment in shipment_status_map.values():
                if int(shipment.get("priority", 0)) == 2 and not bool(shipment.get("is_delivered", False)) and not bool(shipment.get("is_destroyed", False)):
                    critical_undelivered += 1

            row = {
                "obs": np.asarray(step.get("obs"), dtype=np.float32),
                "mask": np.asarray(step.get("mask"), dtype=np.float32),
                "action": int(action),
            }
            if refrig_status in (1, 2) and action_type == 2:
                grouped["diversion"].append(row)
            if critical_undelivered >= 2 and action_type == 1:
                grouped["triage"].append(row)
            if action_type == 5:
                grouped["abort"].append(row)

    print(
        "[Phase4 CriticalStates] "
        f"diversion={len(grouped['diversion'])}, "
        f"triage={len(grouped['triage'])}, abort={len(grouped['abort'])}"
    )
    return grouped


def margin_loss(logits: torch.Tensor, target_action: torch.Tensor, action_mask: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    masked_logits = logits.masked_fill(action_mask <= 0.0, -1e9)
    target_logit = logits.gather(1, target_action.unsqueeze(1)).squeeze(1)
    masked_logits_no_target = masked_logits.scatter(1, target_action.unsqueeze(1), -1e9)
    best_other = masked_logits_no_target.max(dim=1).values
    margin_violation = F.relu(best_other - target_logit + float(margin))
    ce = F.cross_entropy(logits, target_action)
    return ce + 0.5 * margin_violation.mean()


def measure_logit_margin(model, states: list[dict[str, Any]], limit: int = 100) -> float:
    if not states:
        return 0.0
    sample = states[: max(1, min(int(limit), len(states)))]
    margins: list[float] = []
    with torch.no_grad():
        for row in sample:
            obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(row["obs"], dtype=np.float32))
            mask_tensor = torch.as_tensor(np.asarray(row["mask"], dtype=np.float32), device=model.device).unsqueeze(0)
            logits = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor).distribution.logits
            masked = logits.masked_fill(mask_tensor <= 0.0, -1e9)
            top2 = torch.topk(masked, k=2, dim=1).values.squeeze(0)
            margins.append(float((top2[0] - top2[1]).detach().cpu().item()))
    return float(np.mean(margins)) if margins else 0.0


def _hard_tier_score(model, *, seed: int, evaluation_training_step: int) -> float:
    report = run_full_evaluation(
        model,
        seed=int(seed),
        deterministic=True,
        evaluation_training_step=int(evaluation_training_step),
        trace_every=0,
    )
    return float(_row_by_tier(report.get("rows", []), "hard").get("score", 0.0))


def _write_phase4_report(seed: int, payload: dict[str, Any]) -> str:
    report_dir = os.path.join("artifacts", "phase4")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"phase4_report_seed{int(seed)}.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[Phase4] Report saved: {report_path}")
    return report_path


def run_targeted_bc_alignment(
    model,
    critical_states: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    evaluation_training_step: int,
    hard_guard_threshold: float = 0.60,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for state_type in ("diversion", "triage", "abort"):
        states = list(critical_states.get(state_type, []))
        if len(states) < 20:
            summaries[state_type] = {"skipped": True, "reason": f"insufficient samples ({len(states)})"}
            print(f"[Phase4 BC] skip {state_type}: only {len(states)} states")
            continue

        pre_margin = measure_logit_margin(model, states)
        obs_array = np.asarray([row["obs"] for row in states], dtype=np.float32)
        action_array = np.asarray([int(row["action"]) for row in states], dtype=np.int64)
        mask_array = np.asarray([row["mask"] for row in states], dtype=np.float32)

        old_lrs = [float(group["lr"]) for group in model.policy.optimizer.param_groups]
        for group in model.policy.optimizer.param_groups:
            group["lr"] = 5e-5

        model.policy.set_training_mode(True)
        order = np.arange(action_array.shape[0])
        latest_loss = 0.0
        for _ in range(3):
            np.random.shuffle(order)
            for start in range(0, len(order), 64):
                batch_idx = order[start : start + 64]
                batch_obs, _ = model.policy.obs_to_tensor(obs_array[batch_idx])
                batch_actions = torch.as_tensor(action_array[batch_idx], device=model.device, dtype=torch.long)
                batch_masks = torch.as_tensor(mask_array[batch_idx], device=model.device, dtype=torch.float32)
                dist = model.policy.get_distribution(batch_obs, action_masks=batch_masks)
                logits = dist.distribution.logits
                loss = margin_loss(logits, batch_actions, batch_masks, margin=2.0)
                model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.3)
                model.policy.optimizer.step()
                latest_loss = float(loss.detach().cpu().item())
        model.policy.set_training_mode(False)

        for group, old_lr in zip(model.policy.optimizer.param_groups, old_lrs):
            group["lr"] = old_lr

        post_margin = measure_logit_margin(model, states)
        hard_score = _hard_tier_score(model, seed=seed, evaluation_training_step=evaluation_training_step)
        summaries[state_type] = {
            "skipped": False,
            "count": len(states),
            "pre_margin": pre_margin,
            "post_margin": post_margin,
            "loss": latest_loss,
            "hard_score": hard_score,
        }
        print(
            f"[Phase4 BC] {state_type}: count={len(states)}, pre_margin={pre_margin:.3f}, "
            f"post_margin={post_margin:.3f}, hard_score={hard_score:.4f}"
        )
        if hard_score < float(hard_guard_threshold):
            summaries["hard_guard_failed"] = True
            summaries["failed_at"] = state_type
            return summaries
    summaries["hard_guard_failed"] = False
    return summaries


def bc_completion_check(
    model,
    config: ColdChainConfig,
    *,
    seed: int,
    evaluation_training_step: int,
    hard_threshold: float = 0.65,
) -> dict[str, Any]:
    tier_report = run_tier_aligned_validation(
        model,
        config,
        seed=seed,
        evaluation_training_step=evaluation_training_step,
        n_episodes=20,
        include_official_grader=False,
    )
    det_extreme = float(tier_report.get("extreme", {}).get("deterministic_delivery_rate", 0.0))
    sto_extreme = float(tier_report.get("extreme", {}).get("stochastic_delivery_rate", 0.0))
    hard_score = _hard_tier_score(model, seed=seed, evaluation_training_step=evaluation_training_step)

    checks = {
        "det_extreme_gt_5pct": det_extreme > 0.05,
        "hard_score_gate": hard_score >= float(hard_threshold),
        "sto_extreme_ge_15pct": sto_extreme >= 0.15,
    }
    print(
        "[Phase4 BC Gate] "
        f"det_extreme={det_extreme:.2%}, hard={hard_score:.4f}, hard_threshold={float(hard_threshold):.4f}, "
        f"sto_extreme={sto_extreme:.2%}, checks={checks}"
    )
    return {
        "det_extreme": det_extreme,
        "hard_score": hard_score,
        "sto_extreme": sto_extreme,
        "checks": checks,
        "pass_all": bool(all(checks.values())),
    }


def _phase_difficulty_weights(phase: int) -> dict[int, float]:
    if phase == 1:
        return {1: 1.0}
    if phase == 2:
        return {1: 0.20, 2: 0.55, 3: 0.25}
    return {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.35, 5: 0.50}


def _configure_curriculum_wrapper(env: CurriculumWrapper, phase: int) -> CurriculumWrapper:
    env.max_difficulty = 5
    env.difficulty_weights = _phase_difficulty_weights(phase)
    env.scenario_library = _training_scenario_library()
    env.replay_prob = 0.0
    env.replay_min_difficulty = 1
    env.successes_to_advance = 10**9
    env.difficulty = max(_phase_difficulty_weights(phase), key=_phase_difficulty_weights(phase).get)
    env.log_sampling = phase >= 3
    return env


def _phase3_probe_case(config: ColdChainConfig, difficulty: int) -> dict[str, Any]:
    return dict(_training_scenario_library()[difficulty][0])


def run_tier_aligned_validation(
    model,
    config,
    *,
    seed=42,
    evaluation_training_step: int | None = None,
    n_episodes: int = 10,
    include_official_grader: bool = True,
) -> dict[str, Any]:
    eval_step = (
        max(int(model.num_timesteps), int(config.penalty_anneal_steps))
        if evaluation_training_step is None
        else int(evaluation_training_step)
    )
    library = _training_scenario_library()
    summary: dict[str, Any] = {}
    print("\n[Tier-Aligned Validation]")

    official_report = None
    if include_official_grader:
        official_report = run_full_evaluation(
            model,
            seed=seed,
            deterministic=True,
            evaluation_training_step=eval_step,
            trace_every=0,
        )

    for tier_name, difficulty in (("hard", 4), ("extreme", 5)):
        reset_options_list = library[difficulty]
        results_by_mode: dict[str, list[dict[str, Any]]] = {"deterministic": [], "stochastic": []}
        action_snapshots: list[dict[str, Any]] = []

        for deterministic in (True, False):
            eval_env = build_eval_env(config)
            try:
                for episode_index in range(int(n_episodes)):
                    reset_options = dict(reset_options_list[episode_index % len(reset_options_list)])
                    result = run_eval_episode(
                        model,
                        eval_env,
                        seed=seed + difficulty * 100 + episode_index,
                        deterministic=deterministic,
                        curriculum_difficulty=difficulty,
                        evaluation_training_step=eval_step,
                        reset_options=reset_options,
                        collect_trajectory=bool(deterministic and episode_index < 3),
                    )
                    result["scenario_family"] = _scenario_family_name(reset_options, difficulty)
                    results_by_mode["deterministic" if deterministic else "stochastic"].append(result)
                    if deterministic and episode_index < 3:
                        for step in result.get("trajectory", [])[:2]:
                            snapshot = _top2_action_snapshot(model, step["obs"], step["mask"])
                            with torch.no_grad():
                                probs = model.policy.get_distribution(
                                    model.policy.obs_to_tensor(step["obs"])[0],
                                    action_masks=np.asarray(step["mask"]),
                                ).distribution.probs.detach().cpu().numpy().reshape(-1)
                            action_snapshots.append(
                                {
                                    "scenario_family": result["scenario_family"],
                                    "sampled_action": int(step["action"]),
                                    "sampled_action_prob": float(probs[int(step["action"])]),
                                    "sampled_action_rank": _rank_of_action(probs, int(step["action"])),
                                    **snapshot,
                                }
                            )
            finally:
                eval_env.close()

        det_results = results_by_mode["deterministic"]
        sto_results = results_by_mode["stochastic"]
        det_delivery = float(np.mean([1.0 if result.get("delivery_success", False) else 0.0 for result in det_results]))
        sto_delivery = float(np.mean([1.0 if result.get("delivery_success", False) else 0.0 for result in sto_results]))
        gap = max(0.0, sto_delivery - det_delivery)
        all_destroyed_share = float(np.mean([1.0 if result.get("termination_reason") == "all_destroyed" else 0.0 for result in det_results]))
        mean_steps = float(np.mean([float(result.get("steps", 0)) for result in det_results]))
        timeout_missed = []
        failure_buckets = Counter()
        for result in det_results:
            failure_buckets[_classify_failure(result)] += 1
            if str(result.get("termination_reason")) == "max_steps":
                delivered, total = _delivery_counts(result)
                timeout_missed.append(float(max(total - delivered, 0)))
        grader_score = 0.0
        if official_report is not None:
            grader_score = float(_row_by_tier(official_report.get("rows", []), tier_name).get("score", 0.0))

        summary[tier_name] = {
            "deterministic_delivery_rate": det_delivery,
            "stochastic_delivery_rate": sto_delivery,
            "gap": gap,
            "all_destroyed_share": all_destroyed_share,
            "mean_steps": mean_steps,
            "avg_missed_shipments_on_timeout": float(np.mean(timeout_missed)) if timeout_missed else 0.0,
            "failure_buckets": dict(failure_buckets),
            "grader_score": grader_score,
            "action_rank_diagnostics": action_snapshots[:6],
        }
        print(
            f"  > {tier_name}: det={det_delivery:.2%}, sto={sto_delivery:.2%}, gap={gap:.2%}, "
            f"steps={mean_steps:.1f}, all_destroyed={all_destroyed_share:.2%}, "
            f"timeout_missed={summary[tier_name]['avg_missed_shipments_on_timeout']:.2f}, "
            f"failures={dict(failure_buckets)}"
        )
        if action_snapshots:
            first = action_snapshots[0]
            print(
                f"    top2={first['top_actions']} probs={[round(p, 3) for p in first['top_probs']]} "
                f"sampled_rank={first['sampled_action_rank']} sampled_prob={first['sampled_action_prob']:.3f}"
            )

    return summary


def run_phase3_tier_gate(model, config, seed=42) -> dict[str, Any]:
    eval_step = max(int(model.num_timesteps), int(config.penalty_anneal_steps))
    thresholds = {
        "hard": {"deterministic_delivery": 0.65, "gap": 0.30, "destroyed_cap": 0.10},
        "extreme": {"deterministic_delivery": 0.55, "gap": 0.35, "destroyed_cap": 0.10},
    }
    summary = run_tier_aligned_validation(
        model,
        config,
        seed=seed,
        evaluation_training_step=eval_step,
        n_episodes=10,
        include_official_grader=True,
    )
    print("\n[Phase 3 Tier Gate]")
    for tier_name in ("hard", "extreme"):
        tier = summary[tier_name]
        passed = (
            float(tier["deterministic_delivery_rate"]) >= thresholds[tier_name]["deterministic_delivery"]
            and float(tier["gap"]) <= thresholds[tier_name]["gap"]
            and float(tier["all_destroyed_share"]) <= thresholds[tier_name]["destroyed_cap"]
        )
        tier["pass"] = bool(passed)
        print(
            f"  > {tier_name}: det={tier['deterministic_delivery_rate']:.2%}, "
            f"sto={tier['stochastic_delivery_rate']:.2%}, gap={tier['gap']:.2%}, "
            f"all_destroyed={tier['all_destroyed_share']:.2%}, grader={tier['grader_score']:.4f}, "
            f"{'PASS' if passed else 'FAIL'}"
        )
    return summary


def run_acceptance_seed_sweep(model, seeds, evaluation_training_step, deterministic=True, trace_every=0):
    print("\n[Acceptance Seed Sweep]")
    print(f"  > Seeds: {list(seeds)}")

    per_seed_reports = []
    for seed in seeds:
        report = run_full_evaluation(
            model,
            seed=int(seed),
            deterministic=bool(deterministic),
            evaluation_training_step=int(evaluation_training_step),
            trace_every=int(trace_every),
        )
        score = robust_tier_score(report)
        rows = report.get("rows", [])
        hard = float(_row_by_tier(rows, "hard").get("score", 0.0))
        extreme = float(_row_by_tier(rows, "extreme").get("score", 0.0))
        per_seed_reports.append({"seed": int(seed), "robust_score": score, "hard": hard, "extreme": extreme})
        print(f"  > seed={seed}: robust={score:.4f}, hard={hard:.4f}, extreme={extreme:.4f}")

    robust_scores = [x["robust_score"] for x in per_seed_reports]
    hard_scores = [x["hard"] for x in per_seed_reports]
    extreme_scores = [x["extreme"] for x in per_seed_reports]

    summary = {
        "robust_score_mean": float(np.mean(robust_scores)) if robust_scores else 0.0,
        "robust_score_min": float(np.min(robust_scores)) if robust_scores else 0.0,
        "hard_mean": float(np.mean(hard_scores)) if hard_scores else 0.0,
        "hard_min": float(np.min(hard_scores)) if hard_scores else 0.0,
        "extreme_mean": float(np.mean(extreme_scores)) if extreme_scores else 0.0,
        "extreme_min": float(np.min(extreme_scores)) if extreme_scores else 0.0,
        "per_seed": per_seed_reports,
    }

    print(
        "  > Summary: "
        f"robust_mean={summary['robust_score_mean']:.4f}, robust_min={summary['robust_score_min']:.4f}, "
        f"hard_mean={summary['hard_mean']:.4f}, hard_min={summary['hard_min']:.4f}, "
        f"extreme_mean={summary['extreme_mean']:.4f}, extreme_min={summary['extreme_min']:.4f}"
    )
    return summary


def audit_is_active_handling(config: ColdChainConfig, seed: int = 42, episodes: int = 4) -> bool:
    """Validate that inactive entities remain masked/inactive throughout reset+step."""
    print("\n[is_active Audit]")
    checks_passed = True
    scenarios = [(2, 3), (3, 5)]

    for scenario_index, (active_vehicles, active_shipments) in enumerate(scenarios):
        env = ColdChainEnv(config=replace(config))
        try:
            for episode in range(max(int(episodes), 1)):
                obs, info = env.reset(
                    seed=seed + scenario_index * 100 + episode,
                    options={
                        "curriculum_difficulty": 5,
                        "active_vehicle_count": int(active_vehicles),
                        "active_shipment_count": int(active_shipments),
                    },
                )
                _ = obs
                info_active_vehicles = int(info.get("active_vehicle_count", -1))
                info_active_shipments = int(info.get("active_shipment_count", -1))
                if info_active_vehicles != int(active_vehicles) or info_active_shipments != int(active_shipments):
                    checks_passed = False
                    print(
                        "  ! FAIL: info active counts mismatch "
                        f"expected=({active_vehicles},{active_shipments}) "
                        f"actual=({info_active_vehicles},{info_active_shipments})"
                    )

                mask = env.action_masks()
                n_nodes = env.config.n_nodes
                for vehicle_id in range(int(active_vehicles), env.config.n_vehicles):
                    row_start = vehicle_id * 6 * n_nodes
                    row_end = row_start + 6 * n_nodes
                    if int(np.sum(mask[row_start:row_end])) != 0:
                        checks_passed = False
                        print(f"  ! FAIL: inactive vehicle {vehicle_id} has legal actions in mask")

                for shipment in env.shipments:
                    if shipment.id >= int(active_shipments) and bool(getattr(shipment, "is_active", True)):
                        checks_passed = False
                        print(f"  ! FAIL: shipment {shipment.id} expected inactive but marked active")

                valid_actions = np.flatnonzero(mask)
                if len(valid_actions) > 0:
                    action = int(valid_actions[0])
                    _, _, _, _, step_info = env.step(action)
                    if int(step_info.get("active_vehicle_count", -1)) != int(active_vehicles):
                        checks_passed = False
                        print("  ! FAIL: active_vehicle_count changed after step")
                    if int(step_info.get("active_shipment_count", -1)) != int(active_shipments):
                        checks_passed = False
                        print("  ! FAIL: active_shipment_count changed after step")
        finally:
            env.close()

    print(f"  > result: {'PASS' if checks_passed else 'FAIL'}")
    return checks_passed


def run_training_phase(
    phase,
    steps,
    resume_path=None,
    seed=42,
    enable_refrigeration_reaction=False,
    enable_bc_warmstart=True,
    enable_tier_aligned_validation=True,
    enable_phase4_alignment=False,
):
    print(f"\n--- [Phase {phase}] Training for {steps} steps ---")

    curriculum_difficulty = 1 if phase == 1 else (3 if phase == 2 else 5)
    phase3_continuation = _is_phase3_continuation(phase, resume_path)
    ppo_hyperparams = _phase_ppo_hyperparams(phase)
    config = _base_training_space_config(phase)
    if phase == 3 and resume_path and os.path.exists(resume_path):
        checkpoint_max_steps = _infer_checkpoint_max_steps(resume_path)
        if checkpoint_max_steps is not None and checkpoint_max_steps != int(config.max_steps):
            config.max_steps = int(checkpoint_max_steps)
            print(
                "[Phase 3] Adjusted max_steps to match checkpoint observation space: "
                f"{config.max_steps}"
            )
    if phase == 3 and bool(enable_refrigeration_reaction):
        config.enable_refrigeration_reaction = True
        print("[Phase 3] Refrigeration reaction shaping ENABLED.")
    if phase == 3 and bool(config.enable_difficulty_reward_normalization):
        print(
            "[Phase 3] Difficulty-aware reward normalization ENABLED "
            f"(alpha={config.reward_norm_alpha}, warmup={config.reward_norm_warmup_steps}, clip={config.reward_norm_clip})."
        )
    if phase3_continuation:
        print("[Phase 3] Continuation mode: extreme-focused curriculum + stronger BC alignment.")
    
    # Run exploit-proof gate on stable base difficulty so it measures reward-path
    # exploitability rather than random destruction noise at hard difficulty.
    pre_gate_difficulty = 1
    if not pre_training_gate(config, seed=seed, n_steps=500, curriculum_difficulty=pre_gate_difficulty):
        raise RuntimeError("Pre-training gate failed. Reward architecture still exploitable.")

    masking_clean = True
    if phase == 3:
        masking_clean = audit_is_active_handling(config, seed=seed, episodes=3)

    env = ColdChainEnv(config=config)
    curriculum_env = CurriculumWrapper(env)
    curriculum_env = _configure_curriculum_wrapper(curriculum_env, phase)
    env = ActionMasker(curriculum_env, get_mask) # Add masking wrapper
    env = Monitor(env, info_keywords=("delivery_success",))
    env = gym.wrappers.FlattenObservation(env)
    
    anneal_callback = AnnealingCallback(steps_to_full=30000)
    check_every = 500 if phase == 1 else (3000 if phase == 2 else 5000)
    tier_metrics_callback = None
    if phase == 3 and enable_tier_aligned_validation:
        tier_metrics_callback = TierAlignedMetricsCallback(
            config,
            check_every_steps=max(check_every, 10000),
            seed=seed,
        )
    ent_callback = EntropyAnnealingCallback(
        start_ent=float(ppo_hyperparams["ent_coef_start"]),
        end_ent=float(ppo_hyperparams["ent_coef_end"]),
        total_steps=max(steps, 1),
        min_entropy_floor=float(ppo_hyperparams["ent_coef_end"]),
        competence_callback=tier_metrics_callback if phase == 3 else None,
        competence_threshold=0.0,
    )
    illegal_action_callback = IllegalActionCheckCallback(
        config,
        curriculum_difficulty=curriculum_difficulty,
        check_every_steps=check_every,
        n_episodes=3,
        seed=seed,
    )
    delivery_only_callback = DeliveryOnlyCallback(
        config,
        curriculum_difficulty=curriculum_difficulty,
        check_every_steps=check_every,
        n_episodes=10,
        seed=seed,
    )
    det_gap_callback = DeterministicGapCallback(
        config,
        curriculum_difficulty=curriculum_difficulty,
        check_every_steps=check_every,
        n_episodes=6 if phase == 3 else 4,
        seed=seed,
    )
    callbacks = [anneal_callback, ent_callback, illegal_action_callback, det_gap_callback, delivery_only_callback, PhasedMetricsCallback()]
    if tier_metrics_callback is not None:
        callbacks.insert(3, tier_metrics_callback)
    if phase >= 2:
        callbacks.insert(3, ExploitDetectorCallback(check_every=2000, destroy_threshold=0.5))
    if phase == 3 and masking_clean:
        if phase3_continuation:
            curriculum_env.difficulty_weights = _phase3_focus_extreme_weights()
            print(f"[Phase 3] Extreme-focused curriculum enabled: {curriculum_env.difficulty_weights}")
        else:
            # Replace abrupt hard/extreme jump with a progressive curriculum ramp.
            curriculum_env.difficulty_weights = _phase3_ramped_weights(0.0)
            callbacks.insert(2, CurriculumRampCallback(curriculum_env=curriculum_env, total_steps=steps))
            print("[Phase 3] Curriculum ramp enabled.")
    elif phase == 3:
        print("[Phase 3] Curriculum ramp skipped because is_active audit failed.")
    
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        model = MaskablePPO.load(resume_path, env=env)
        _apply_loaded_ppo_hyperparams(model, ppo_hyperparams)
        print(
            "Applied phase-specific PPO settings after load: "
            f"lr={float(ppo_hyperparams['learning_rate']):.1e}, "
            f"n_steps={int(ppo_hyperparams['n_steps'])}, "
            f"batch_size={int(ppo_hyperparams['batch_size'])}, "
            f"n_epochs={int(ppo_hyperparams['n_epochs'])}, "
            f"vf_coef={float(ppo_hyperparams['vf_coef']):.2f}, "
            f"gamma={float(ppo_hyperparams['gamma']):.3f}, "
            f"gae_lambda={float(ppo_hyperparams['gae_lambda']):.2f}, "
            f"clip_range={float(ppo_hyperparams['clip_range']):.2f}"
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=int(ppo_hyperparams["n_steps"]),
            batch_size=int(ppo_hyperparams["batch_size"]),
            n_epochs=int(ppo_hyperparams["n_epochs"]),
            ent_coef=float(ppo_hyperparams["ent_coef_start"]),
            learning_rate=float(ppo_hyperparams["learning_rate"]),
            vf_coef=float(ppo_hyperparams["vf_coef"]),
            gamma=float(ppo_hyperparams["gamma"]),
            gae_lambda=float(ppo_hyperparams["gae_lambda"]),
            clip_range=float(ppo_hyperparams["clip_range"]),
            verbose=1,
            seed=seed
        )

    if phase == 3 and bool(enable_phase4_alignment):
        pre_bc_checkpoint = "models/ppo_phase3_pre_bc_backup.zip"
        os.makedirs("models", exist_ok=True)
        _ensure_gym_version_for_sb3()
        model.save(pre_bc_checkpoint)
        print(f"[Phase4] Saved pre-BC backup checkpoint: {pre_bc_checkpoint}")

        eval_step = max(int(model.num_timesteps), int(config.penalty_anneal_steps))
        baseline_hard = _hard_tier_score(model, seed=seed, evaluation_training_step=eval_step)
        hard_guard_threshold = 0.60 if baseline_hard >= 0.60 else max(0.50, baseline_hard - 0.03)
        hard_gate_threshold = 0.65 if baseline_hard >= 0.65 else max(0.50, baseline_hard - 0.02)
        print(
            "[Phase4] Hard baseline="
            f"{baseline_hard:.4f}, guard_threshold={hard_guard_threshold:.4f}, gate_threshold={hard_gate_threshold:.4f}"
        )

        harvest_episodes = 500 if int(steps) >= 15000 else 120
        print(f"[Phase4] Harvest episodes={harvest_episodes}")
        harvest = harvest_expert_trajectories(config, seed=seed, n_episodes=harvest_episodes, min_delivery_score=0.40)
        harvested_results = list(harvest.get("results", []))
        critical_states = extract_critical_states(harvested_results, n_nodes=int(config.n_nodes))

        phase4_payload: dict[str, Any] = {
            "seed": int(seed),
            "steps": int(steps),
            "pre_bc_checkpoint": pre_bc_checkpoint,
            "baseline": {
                "hard": float(baseline_hard),
                "guard_threshold": float(hard_guard_threshold),
                "gate_threshold": float(hard_gate_threshold),
            },
            "harvest": {
                "episodes": int(harvest_episodes),
                "kept": int(len(harvested_results)),
                "score_min": float(min(harvest.get("scores", [0.0])) if harvest.get("scores") else 0.0),
                "score_mean": float(np.mean(harvest.get("scores", [0.0])) if harvest.get("scores") else 0.0),
                "score_max": float(max(harvest.get("scores", [0.0])) if harvest.get("scores") else 0.0),
            },
            "critical_states": {
                "diversion": int(len(critical_states.get("diversion", []))),
                "triage": int(len(critical_states.get("triage", []))),
                "abort": int(len(critical_states.get("abort", []))),
            },
        }

        if len(critical_states.get("diversion", [])) < 30:
            print("[Phase4] WARNING: diversion states < 30. Observation augmentation may be needed before strict BC alignment.")

        bc_summary = run_targeted_bc_alignment(
            model,
            critical_states,
            seed=seed,
            evaluation_training_step=eval_step,
            hard_guard_threshold=hard_guard_threshold,
        )

        if bool(bc_summary.get("hard_guard_failed", False)):
            print("[Phase4] Hard-tier guard failed during BC. Restoring pre-BC checkpoint.")
            model = MaskablePPO.load(pre_bc_checkpoint, env=env)
            _apply_loaded_ppo_hyperparams(model, ppo_hyperparams)
            phase4_payload["bc_summary"] = bc_summary
            phase4_payload["bc_gate"] = {"pass_all": False, "reason": "hard_guard_failed"}
            _write_phase4_report(seed, phase4_payload)
        else:
            gate = bc_completion_check(
                model,
                config,
                seed=seed,
                evaluation_training_step=eval_step,
                hard_threshold=hard_gate_threshold,
            )
            phase4_payload["bc_summary"] = bc_summary
            phase4_payload["bc_gate"] = gate
            if not bool(gate.get("pass_all", False)):
                print("[Phase4] BC completion gate FAILED. Restoring pre-BC checkpoint before RL continuation.")
                model = MaskablePPO.load(pre_bc_checkpoint, env=env)
                _apply_loaded_ppo_hyperparams(model, ppo_hyperparams)
                _write_phase4_report(seed, phase4_payload)
            else:
                print("[Phase4] BC completion gate PASSED. Proceeding to RL continuation.")
                _write_phase4_report(seed, phase4_payload)

    if phase == 3 and bool(enable_bc_warmstart) and not bool(enable_phase4_alignment):
        artifact_path = os.path.join("models", f"phase3_harvest_seed{seed}.npz")
        dataset = harvest_imitation_dataset(
            model,
            config,
            seed=seed,
            episodes_per_scenario=4,
            top_percentile=0.50 if phase3_continuation else 0.35,
            extreme_focus=bool(phase3_continuation),
            artifact_path=artifact_path,
        )
        rank_summary = _dataset_rank_summary(dataset)
        print(
            "[Phase 3] Harvested imitation dataset: "
            f"samples={dataset['summary']['sample_count']}, "
            f"hard_eps={dataset['summary']['episodes'].get('hard', {}).get('selected_episodes', 0)}, "
            f"extreme_eps={dataset['summary']['episodes'].get('extreme', {}).get('selected_episodes', 0)}, "
            f"rank_top1={rank_summary['top1']:.2%}, rank_top2={rank_summary['top2']:.2%}"
        )
        bc_summary = run_behavior_cloning_warmstart(
            model,
            dataset,
            epochs=5 if phase3_continuation else 3,
            batch_size=min(128, int(ppo_hyperparams["batch_size"])),
        )
        print(
            "[Phase 3] BC warmstart complete: "
            f"epochs={bc_summary['epochs']}, samples={bc_summary['samples']}, loss={bc_summary['loss']:.4f}"
        )
    
    model.learn(total_timesteps=steps, callback=callbacks)
    
    save_path = f"models/ppo_phase{phase}.zip"
    os.makedirs("models", exist_ok=True)
    _ensure_gym_version_for_sb3()
    model.save(save_path)
    print(f"✓ Phase {phase} complete. Model saved to {save_path}")
    
    # Validation
    validation = validate_model(
        model,
        config,
        seed,
        curriculum_difficulty=curriculum_difficulty,
        delivery_only_rate=delivery_only_callback.latest_true_delivery_rate,
        tier_aligned=bool(phase == 3 and enable_tier_aligned_validation),
    )
    if phase == 3:
        det_rate = float(validation.get("deterministic_delivery_rate", 0.0))
        if det_rate <= 0.05 and not bool(config.enable_refrigeration_reaction):
            print(
                "! Deterministic delivery remains near zero. "
                "Re-run phase 3 with --enable-refrigeration-reaction for targeted shaping."
            )
        run_termination_probe(model, config, seed=seed, n_episodes=10, curriculum_difficulty=4)
        tier_gate = run_phase3_tier_gate(model, config, seed=seed)
        eval_step = max(int(model.num_timesteps), int(config.penalty_anneal_steps))
        sweep = run_acceptance_seed_sweep(
            model,
            seeds=[42, 101, 202, 303, 404],
            evaluation_training_step=eval_step,
            deterministic=True,
        )
        if (
            tier_gate["hard"]["pass"]
            and tier_gate["extreme"]["pass"]
            and sweep["hard_mean"] >= 0.60
            and sweep["extreme_mean"] >= 0.45
            and sweep["robust_score_mean"] >= 0.58
        ):
            robust_path = "models/ppo_phase3_robust_ready.zip"
            model.save(robust_path)
            print(f"✓ Robustness gate passed. Snapshot saved to {robust_path}")
        else:
            print("! Robustness gate not met. Keep training on phase 3 with harder seeds.")
    return model

def validate_model(model, config, seed, curriculum_difficulty=3, delivery_only_rate=0.0, tier_aligned=False):
    print("\n[Validation Check]")
    if tier_aligned and int(curriculum_difficulty) >= 5:
        report = run_tier_aligned_validation(
            model,
            config,
            seed=seed,
            evaluation_training_step=max(int(model.num_timesteps), int(config.penalty_anneal_steps)),
            n_episodes=10,
            include_official_grader=False,
        )
        extreme = report.get("extreme", {})
        print(f"  > DeliveryOnlyCallback true delivery rate: {delivery_only_rate:.2%}")
        if float(extreme.get("deterministic_delivery_rate", 0.0)) <= 0.0:
            print("  ! WARNING: No deterministic extreme deliveries in tier-aligned validation.")
        return {
            "delivery_rate": float(extreme.get("deterministic_delivery_rate", 0.0)),
            "deterministic_delivery_rate": float(extreme.get("deterministic_delivery_rate", 0.0)),
            "stochastic_delivery_rate": float(extreme.get("stochastic_delivery_rate", 0.0)),
            "tier_report": report,
        }

    eval_env = build_eval_env(config)
    eval_training_step = max(int(model.num_timesteps), int(config.penalty_anneal_steps))
    seed_results = []

    for offset in range(5):
        result = run_eval_episode(
            model,
            eval_env,
            seed=seed + 100 + offset,
            deterministic=True,
            curriculum_difficulty=curriculum_difficulty,
            evaluation_training_step=eval_training_step,
            trace_every=0,
            collect_trajectory=False,
        )
        seed_results.append(result)

    rewards = [r["total_reward"] for r in seed_results]
    delivery_rate = float(np.mean([1.0 if r.get("delivery_success", False) else 0.0 for r in seed_results]))
    avg_reward = float(np.mean(rewards))
    median_reward = float(np.median(rewards))
    std_reward = float(np.std(rewards))
    avg_steps = float(np.mean([r["steps"] for r in seed_results]))
    action_mix = {k: int(sum(r["action_type_counts"][k] for r in seed_results)) for k in range(6)}
    termination_reasons = Counter(str(r.get("termination_reason", "unknown")) for r in seed_results)

    print(f"  > Contract: deterministic=True, difficulty={curriculum_difficulty}, eval_step={eval_training_step}")
    print(f"  > Seeds: {len(seed_results)}, Delivery Rate: {delivery_rate:.2%}, Mean Reward: {avg_reward:.2f}, Median Reward: {median_reward:.2f}, Std: {std_reward:.2f}, Avg Steps: {avg_steps:.1f}")
    print(f"  > Action mix [WAIT,REROUTE,DIVERT,SWAP,EXPEDITE,ABORT]: {action_mix}")
    print(f"  > Termination reasons: {dict(termination_reasons)}")
    print(f"  > DeliveryOnlyCallback true delivery rate: {delivery_only_rate:.2%}")
    if delivery_rate <= 0.0:
        print("  ! WARNING: No deliveries in deterministic validation seeds.")

    # Compare stochastic vs deterministic delivery rate to expose robustness gaps.
    delivery_by_mode = {}
    for deterministic in (False, True):
        delivered = 0
        gap_env = build_eval_env(config)
        try:
            for offset in range(5):
                obs, _ = gap_env.reset(seed=seed + 500 + offset, options={"curriculum_difficulty": curriculum_difficulty})
                done = False
                while not done:
                    mask = gap_env.unwrapped.action_masks()
                    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
                    obs, _, terminated, truncated, info = gap_env.step(int(action))
                    done = bool(terminated or truncated)
                delivered += int(bool(info.get("delivery_success", False)))
        finally:
            gap_env.close()
        label = "deterministic" if deterministic else "stochastic"
        mode_rate = delivered / 5
        delivery_by_mode[label] = mode_rate
        print(f"  > {label:>14} delivery rate: {mode_rate:.2%}")

    if curriculum_difficulty >= 5:
        deterministic_rate = float(delivery_by_mode.get("deterministic", 0.0))
        stochastic_rate = float(delivery_by_mode.get("stochastic", 0.0))
        gap = max(0.0, stochastic_rate - deterministic_rate)
        all_destroyed_share = 0.0
        if seed_results:
            all_destroyed_share = termination_reasons.get("all_destroyed", 0) / float(len(seed_results))
        top_termination_share = 0.0
        if seed_results:
            top_termination_share = max(termination_reasons.values()) / float(len(seed_results))

        gate_checks = {
            "deterministic delivery rate > 40%": deterministic_rate > 0.40,
            "stochastic-deterministic gap < 30%": gap < 0.30,
            "all_destroyed rate < 10%": all_destroyed_share < 0.10,
            "ep_len_mean > 20": avg_steps > 20.0,
            "DeliveryOnlyCallback rate > 30%": float(delivery_only_rate) > 0.30,
            "No single termination path > 50%": top_termination_share <= 0.50,
        }

        print("  > Phase 3 Baseline Gates:")
        for label, passed in gate_checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"    - {status}: {label}")

        if not all(gate_checks.values()):
            print("  ! PHASE 3 NOT SIGNED OFF: one or more hard gates failed.")

        run_phase3_tier_gate(model, config, seed=seed)

    eval_env.close()
    return {
        "delivery_rate": float(delivery_rate),
        "deterministic_delivery_rate": float(delivery_by_mode.get("deterministic", 0.0)),
        "stochastic_delivery_rate": float(delivery_by_mode.get("stochastic", 0.0)),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-refrigeration-reaction", action="store_true")
    parser.add_argument("--disable-bc-warmstart", action="store_true")
    parser.add_argument("--disable-tier-aligned-validation", action="store_true")
    parser.add_argument("--enable-phase4-alignment", action="store_true")
    args = parser.parse_args()
    
    phases = {
        1: 1000,
        2: 100000,
        3: 500000
    }
    
    run_training_phase(
        args.phase,
        phases[args.phase],
        args.resume,
        args.seed,
        enable_refrigeration_reaction=bool(args.enable_refrigeration_reaction),
        enable_bc_warmstart=not bool(args.disable_bc_warmstart),
        enable_tier_aligned_validation=not bool(args.disable_tier_aligned_validation),
        enable_phase4_alignment=bool(args.enable_phase4_alignment),
    )
