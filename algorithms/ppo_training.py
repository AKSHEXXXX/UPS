import os
import argparse
import sys
from pathlib import Path
from collections import Counter
from dataclasses import replace
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.env import ColdChainEnv, CurriculumWrapper
from core.config import ColdChainConfig
from evaluation.eval_contract import build_eval_env, run_eval_episode
from core.graders import run_full_evaluation

def get_mask(env):
    return env.unwrapped.action_masks()

class EntropyAnnealingCallback(BaseCallback):
    """Hold high entropy until delivery competence is reached, then anneal."""
    def __init__(self, start_ent=0.05, end_ent=0.01, min_delivery_rate=0.5, decay_steps=120000, min_entropy_floor=0.01, verbose=0):
        super().__init__(verbose)
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.min_entropy_floor = min_entropy_floor
        self.min_delivery_rate = min_delivery_rate
        self.decay_steps = max(decay_steps, 1)
        self._decay_start_timestep = None

    def _on_step(self) -> bool:
        recent_infos = list(self.model.ep_info_buffer)
        if recent_infos:
            delivery_rate = float(np.mean([float(info.get("delivery_success", 0.0)) for info in recent_infos]))
        else:
            delivery_rate = 0.0

        if self._decay_start_timestep is None and delivery_rate >= self.min_delivery_rate:
            self._decay_start_timestep = int(self.num_timesteps)

        if self._decay_start_timestep is None:
            current_ent = self.start_ent
        else:
            progress = min((self.num_timesteps - self._decay_start_timestep) / self.decay_steps, 1.0)
            current_ent = self.start_ent + (self.end_ent - self.start_ent) * progress

        # Keep exploration alive throughout training.
        self.model.ent_coef = max(current_ent, self.min_entropy_floor)
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

def run_training_phase(phase, steps, resume_path=None, seed=42):
    print(f"\n--- [Phase {phase}] Training for {steps} steps ---")

    curriculum_difficulty = 1 if phase == 1 else (2 if phase == 2 else 3)

    config_kwargs = dict(
        n_vehicles=1,
        n_nodes=10,
        n_shipments=1,
        max_shipments=1,
        max_steps=200,
        penalty_anneal_steps=30000,
        penalty_initial_scale=0.10,
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
            breakdown_probability=0.01,
            refrigeration_degradation_prob=0.012,
            penalty_anneal_steps=60000,
            penalty_initial_scale=0.15,
        )

    if resume_path is None:
        if phase >= 2:
            config_kwargs["max_steps"] = 240
        if phase >= 3:
            config_kwargs["max_steps"] = 300

    config = ColdChainConfig(**config_kwargs)
    
    # Run exploit-proof gate on stable base difficulty so it measures reward-path
    # exploitability rather than random destruction noise at hard difficulty.
    pre_gate_difficulty = 1
    if not pre_training_gate(config, seed=seed, n_steps=500, curriculum_difficulty=pre_gate_difficulty):
        raise RuntimeError("Pre-training gate failed. Reward architecture still exploitable.")

    env = ColdChainEnv(config=config)
    env = CurriculumWrapper(env)
    env.difficulty = int(curriculum_difficulty)
    # Phase 3 benefits from replaying easier regimes so the policy does not
    # overfit the hardest rollout shape and forget the stable behaviors.
    if phase == 3:
        env.successes_to_advance = 50
        env.replay_prob = 0.15
        env.replay_min_difficulty = 1
        env.max_difficulty = 3
    else:
        # Keep the earlier phases anchored so the phase-specific evaluation
        # remains predictable and quick.
        env.successes_to_advance = 10**9
        env.replay_prob = 0.0
    env = ActionMasker(env, get_mask) # Add masking wrapper
    env = Monitor(env, info_keywords=("delivery_success",))
    env = gym.wrappers.FlattenObservation(env)
    
    anneal_callback = AnnealingCallback(steps_to_full=30000)
    ent_callback = EntropyAnnealingCallback(start_ent=0.05, end_ent=0.01, min_delivery_rate=0.5, decay_steps=120000, min_entropy_floor=0.01)
    check_every = 500 if phase == 1 else (3000 if phase == 2 else 5000)
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
    callbacks = [anneal_callback, ent_callback, illegal_action_callback, delivery_only_callback, PhasedMetricsCallback()]
    if phase >= 2:
        callbacks.insert(3, ExploitDetectorCallback(check_every=2000, destroy_threshold=0.5))
    
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        model = MaskablePPO.load(resume_path, env=env)
        # Update model parameters if needed
    else:
        if phase == 1:
            learning_rate = 1e-4
            n_steps = 256
            batch_size = 64
            n_epochs = 4
        elif phase == 2:
            learning_rate = 7e-5
            n_steps = 512
            batch_size = 128
            n_epochs = 6
        else:
            learning_rate = 4e-5
            n_steps = 512
            batch_size = 128
            n_epochs = 6
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=0.05,
            learning_rate=learning_rate,
            gamma=0.998 if phase >= 3 else 0.995,
            verbose=1,
            seed=seed
        )
    
    model.learn(total_timesteps=steps, callback=callbacks)
    
    save_path = f"models/ppo_phase{phase}.zip"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"✓ Phase {phase} complete. Model saved to {save_path}")
    
    # Validation
    validate_model(model, config, seed, curriculum_difficulty=curriculum_difficulty, delivery_only_rate=delivery_only_callback.latest_true_delivery_rate)
    if phase == 3:
        run_termination_probe(model, config, seed=seed, n_episodes=10, curriculum_difficulty=curriculum_difficulty)
        eval_step = max(int(model.num_timesteps), int(config.penalty_anneal_steps))
        sweep = run_acceptance_seed_sweep(
            model,
            seeds=[42, 101, 202, 303, 404],
            evaluation_training_step=eval_step,
            deterministic=True,
        )
        if sweep["hard_mean"] >= 0.60 and sweep["extreme_mean"] >= 0.45 and sweep["robust_score_mean"] >= 0.58:
            robust_path = "models/ppo_phase3_robust_ready.zip"
            model.save(robust_path)
            print(f"✓ Robustness gate passed. Snapshot saved to {robust_path}")
        else:
            print("! Robustness gate not met. Keep training on phase 3 with harder seeds.")
    return model

def validate_model(model, config, seed, curriculum_difficulty=3, delivery_only_rate=0.0):
    print("\n[Validation Check]")
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
        for offset in range(5):
            obs, _ = gap_env.reset(seed=seed + 500 + offset, options={"curriculum_difficulty": curriculum_difficulty})
            done = False
            while not done:
                mask = gap_env.unwrapped.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
                obs, _, terminated, truncated, info = gap_env.step(action)
                done = bool(terminated or truncated)
            delivered += int(bool(info.get("delivery_success", False)))
        label = "deterministic" if deterministic else "stochastic"
        mode_rate = delivered / 5
        delivery_by_mode[label] = mode_rate
        print(f"  > {label:>14} delivery rate: {mode_rate:.2%}")

    if curriculum_difficulty == 3:
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

        print("  > Phase 3 Hard Gates:")
        for label, passed in gate_checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"    - {status}: {label}")

        if not all(gate_checks.values()):
            print("  ! PHASE 3 NOT SIGNED OFF: one or more hard gates failed.")

        tier_report = run_full_evaluation(
            model,
            seed=seed,
            deterministic=True,
            evaluation_training_step=eval_training_step,
            trace_every=0,
        )
        rows = tier_report.get("rows", [])
        hard_row = _row_by_tier(rows, "hard")
        extreme_row = _row_by_tier(rows, "extreme")
        print("  > Tier diagnostics (deterministic):")
        if hard_row:
            print(
                f"    - hard: score={float(hard_row.get('score', 0.0)):.4f}, "
                f"delivery={float(hard_row.get('delivery_ratio', 0.0)):.4f}, "
                f"speed={float(hard_row.get('speed_ratio', 0.0)):.4f}, "
                f"eff={float(hard_row.get('efficiency_ratio', 0.0)):.4f}, "
                f"triage={float(hard_row.get('triage_score', 0.0)):.4f}"
            )
        if extreme_row:
            print(
                f"    - extreme: score={float(extreme_row.get('score', 0.0)):.4f}, "
                f"delivery={float(extreme_row.get('delivery_ratio', 0.0)):.4f}, "
                f"speed={float(extreme_row.get('speed_ratio', 0.0)):.4f}, "
                f"eff={float(extreme_row.get('efficiency_ratio', 0.0)):.4f}, "
                f"triage={float(extreme_row.get('triage_score', 0.0)):.4f}"
            )

    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    phases = {
        1: 1000,
        2: 10000,
        3: 150000
    }
    
    run_training_phase(args.phase, phases[args.phase], args.resume, args.seed)
