"""
PPO Training Agent for ColdChain-Gym

Trains a Proximal Policy Optimization (PPO) agent on ColdChainEnv with:
- Monitor wrapper for trajectory logging
- 100K timesteps training
- Deterministic evaluation over 20 episodes
- CompositeGrader scoring
- Reward breakdown tracking

Usage:
    python examples/ppo_training_example.py

    Or in Google Colab:
    !python examples/ppo_training_example.py
"""

import argparse
import os
import numpy as np
import gymnasium as gym

# Stable-Baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import get_linear_fn
    HAS_SB3 = True
except ImportError:
    PPO = None
    BaseCallback = object
    Monitor = None
    HAS_SB3 = False

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    HAS_MASKABLE = True
except ImportError:
    MaskablePPO = None
    ActionMasker = None
    HAS_MASKABLE = False

# ColdChain-Gym imports
from server.env import ColdChainEnv
from core.config import ColdChainConfig
from graders import CompositeGrader


def _flat_to_action(flat_index, config):
    """Convert flat action index into [vehicle_idx, action_type, target_idx]."""
    vehicle_idx = flat_index // (6 * config.n_nodes)
    rem = flat_index % (6 * config.n_nodes)
    action_type = rem // config.n_nodes
    target_idx = rem % config.n_nodes
    return np.array([vehicle_idx, action_type, target_idx], dtype=np.int64)


def _mask_fn(env):
    """ActionMasker callback to expose masks from the base ColdChain environment."""
    return env.unwrapped.action_masks().astype(bool)


class FlatDiscreteActionWrapper(gym.ActionWrapper):
    """Expose the environment as Discrete actions while keeping base flat action mask semantics."""

    def __init__(self, env):
        super().__init__(env)
        cfg = env.config
        self._cfg = cfg
        self.action_space = gym.spaces.Discrete((cfg.n_vehicles + 1) * 6 * cfg.n_nodes)

    def action(self, action):
        flat_index = int(action)
        return _flat_to_action(flat_index, self._cfg)

    def action_masks(self):
        return self.env.action_masks()


def make_maskable_flat_env(config, log_dir=None, seed=None):
    """Create a flattened-observation environment compatible with MaskablePPO."""
    env = ColdChainEnv(config=config)
    env = FlatDiscreteActionWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = ActionMasker(env, _mask_fn)
    if log_dir is not None:
        env = Monitor(env, log_dir)
    if seed is not None:
        env.reset(seed=seed)
    return env


class TrendMetricsCallback(BaseCallback):
    """Collect rollout/train metrics to evaluate short-run learning trends."""

    def __init__(self):
        super().__init__()
        self.history = {
            "ep_rew_mean": [],
            "ep_len_mean": [],
            "entropy_loss": [],
        }

    def _on_step(self) -> bool:
        return True

    def _capture(self) -> None:
        # rollout means are derived from the episode info buffer populated by Monitor.
        try:
            ep_infos = list(getattr(self.model, "ep_info_buffer", []))
            if ep_infos:
                rewards = [float(item["r"]) for item in ep_infos if "r" in item]
                lengths = [float(item["l"]) for item in ep_infos if "l" in item]
                if rewards:
                    self.history["ep_rew_mean"].append(float(np.mean(rewards)))
                if lengths:
                    self.history["ep_len_mean"].append(float(np.mean(lengths)))
        except Exception:
            pass

        # train entropy metric comes from logger.
        values = getattr(self.model.logger, "name_to_value", {})
        entropy_value = values.get("train/entropy_loss")
        if entropy_value is not None:
            try:
                self.history["entropy_loss"].append(float(entropy_value))
            except Exception:
                pass

    def _on_rollout_end(self) -> None:
        self._capture()

    def _on_training_end(self) -> None:
        self._capture()


def run_micro_training_sanity(timesteps=960, n_steps=64):
    """
    CPU-friendly micro training sanity pass.

    Tracks SB3 metrics:
    - ep_rew_mean (expected upward)
    - ep_len_mean (expected downward)
    - entropy_loss magnitude (expected to decrease gradually, not collapse)
    """
    if not HAS_SB3:
        raise RuntimeError("stable-baselines3 is required. Install with: pip install stable-baselines3")
    if not HAS_MASKABLE:
        raise RuntimeError("sb3-contrib is required for MaskablePPO. Install with: pip install sb3-contrib")

    if timesteps < 500 or timesteps > 1000:
        raise ValueError(f"micro timesteps must be in [500, 1000], got {timesteps}")

    cfg = ColdChainConfig(max_steps=150, debug_step_trace=False)
    log_dir = "./logs/micro_sanity"
    os.makedirs(log_dir, exist_ok=True)

    env = make_maskable_flat_env(cfg, log_dir=log_dir, seed=42)
    callback = TrendMetricsCallback()

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        verbose=1,
    )

    print("\n" + "=" * 70)
    print("MICRO TRAINING SANITY CHECK")
    print("=" * 70)
    print(f"timesteps={timesteps} n_steps={n_steps} (CPU-friendly)")
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    env.close()

    metrics = callback.history

    def _first_last(values):
        if len(values) < 2:
            return None, None
        return values[0], values[-1]

    rew_first, rew_last = _first_last(metrics["ep_rew_mean"])
    len_first, len_last = _first_last(metrics["ep_len_mean"])
    ent_first, ent_last = _first_last(metrics["entropy_loss"])

    print("\nTracked SB3 metrics:")
    print(f"  ep_rew_mean points: {len(metrics['ep_rew_mean'])}")
    print(f"  ep_len_mean points: {len(metrics['ep_len_mean'])}")
    print(f"  entropy_loss points: {len(metrics['entropy_loss'])}")

    if rew_first is None or len_first is None or ent_first is None:
        raise RuntimeError("Insufficient metric points collected. Increase micro timesteps slightly.")

    reward_up = rew_last > rew_first
    length_down = len_last < len_first

    # entropy_loss in SB3 is typically negative; use magnitude for decay checks.
    ent_mag_first = abs(ent_first)
    ent_mag_last = abs(ent_last)
    entropy_decreasing = ent_mag_last < ent_mag_first
    entropy_not_crashed = ent_mag_last > 1e-3 and ent_mag_last > 0.05 * ent_mag_first

    print(f"\n  ep_rew_mean:  start={rew_first:.4f} end={rew_last:.4f} delta={rew_last - rew_first:.4f}")
    print(f"  ep_len_mean:  start={len_first:.4f} end={len_last:.4f} delta={len_last - len_first:.4f}")
    print(
        f"  entropy_loss: start={ent_first:.6f} end={ent_last:.6f} "
        f"|abs| start={ent_mag_first:.6f} end={ent_mag_last:.6f}"
    )

    print("\nTrend checks:")
    print(f"  ep_rew_mean trending up: {'PASS' if reward_up else 'FAIL'}")
    print(f"  ep_len_mean trending down: {'PASS' if length_down else 'FAIL'}")
    print(f"  entropy decreasing slowly (not zero-crash): {'PASS' if (entropy_decreasing and entropy_not_crashed) else 'FAIL'}")
    print("=" * 70 + "\n")

    return {
        "reward_up": reward_up,
        "length_down": length_down,
        "entropy_ok": entropy_decreasing and entropy_not_crashed,
        "metrics": metrics,
    }


def run_pretraining_sanity_checks(config=None, random_steps=200, episodes=3):
    """
    Execute pre-training environment checks from the PPO debug guide.

    Hard checks:
    - Observation shape/type is valid
    - Reward becomes non-zero at least sometimes under random play
    - Environment state changes (movement happens)
    - Episodes terminate within max_steps
    - Action mask always provides at least one legal action
    """
    cfg = config or ColdChainConfig()
    env = ColdChainEnv(config=cfg)

    stats = {
        "steps": 0,
        "non_zero_reward_steps": 0,
        "movement_steps": 0,
        "terminations": 0,
        "truncations": 0,
        "invalid_masks": 0,
        "obs_space_violations": 0,
        "revisit_steps": 0,
    }

    print("\n" + "=" * 70)
    print("PRE-TRAINING SANITY CHECKS")
    print("=" * 70)
    print(f"Episodes: {episodes} | Random steps cap per episode: {random_steps}")

    for ep in range(episodes):
        obs, info = env.reset(seed=100 + ep)
        if not env.observation_space.contains(obs):
            stats["obs_space_violations"] += 1

        done = False
        step = 0
        visited_locations = set()

        while not done and step < min(random_steps, cfg.max_steps + 1):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                stats["invalid_masks"] += 1
                action = np.array([cfg.n_vehicles, 0, 0], dtype=np.int64)
            else:
                sampled_flat = int(env.np_random.choice(valid))
                action = _flat_to_action(sampled_flat, cfg)

            prev_locations = tuple(v.location for v in env.vehicles)
            obs, reward, terminated, truncated, info = env.step(action)
            curr_locations = tuple(v.location for v in env.vehicles)

            if not env.observation_space.contains(obs):
                stats["obs_space_violations"] += 1

            if abs(float(reward)) > 1e-9:
                stats["non_zero_reward_steps"] += 1
            if curr_locations != prev_locations:
                stats["movement_steps"] += 1

            lead_loc = env.vehicles[0].location if env.vehicles else -1
            if lead_loc in visited_locations:
                stats["revisit_steps"] += 1
            visited_locations.add(lead_loc)

            stats["steps"] += 1
            step += 1
            done = bool(terminated or truncated)

            if terminated:
                stats["terminations"] += 1
            if truncated:
                stats["truncations"] += 1

        print(
            f"Episode {ep + 1}: steps={step:3d} | "
            f"terminated={done and not bool(truncated)} | truncated={bool(truncated)}"
        )

    env.close()

    print("-" * 70)
    print(f"Total steps:                 {stats['steps']}")
    print(f"Non-zero reward steps:       {stats['non_zero_reward_steps']}")
    print(f"Movement steps:              {stats['movement_steps']}")
    print(f"Terminations:                {stats['terminations']}")
    print(f"Truncations:                 {stats['truncations']}")
    print(f"Invalid action masks:        {stats['invalid_masks']}")
    print(f"Observation violations:      {stats['obs_space_violations']}")
    print(f"Revisit steps (diagnostic):  {stats['revisit_steps']}")

    failures = []
    termination_check_active = random_steps >= cfg.max_steps
    if stats["steps"] == 0:
        failures.append("No rollout steps executed")
    if stats["non_zero_reward_steps"] == 0:
        failures.append("Rewards stayed exactly zero in random rollouts")
    if stats["movement_steps"] == 0:
        failures.append("Environment state never changed (agent appears stationary)")
    if termination_check_active and (stats["terminations"] + stats["truncations"]) == 0:
        failures.append("Episodes never ended within configured max_steps")
    if stats["invalid_masks"] > 0:
        failures.append("At least one state had no legal action in mask")
    if stats["obs_space_violations"] > 0:
        failures.append("Observation space violations detected")

    if failures:
        print("\n✗ PRE-CHECK FAILED")
        for issue in failures:
            print(f"  - {issue}")
        raise RuntimeError("Pre-training sanity checks failed. Fix environment issues before PPO training.")

    print("\n✓ PRE-CHECK PASSED")
    if not termination_check_active:
        print(
            f"  Note: termination check skipped because precheck_steps ({random_steps}) < max_steps ({cfg.max_steps})."
        )
    print("  Environment is ready for PPO training.")
    print("=" * 70 + "\n")
    return stats


def setup_environment():
    """Create and wrap environment with Monitor."""
    if not HAS_SB3:
        raise RuntimeError("stable-baselines3 not installed. Install with: pip install stable-baselines3")

    # Create logs directory
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    config = ColdChainConfig(
        n_vehicles=4,
        n_nodes=20,
        max_steps=500,
        breakdown_probability=0.001,
    )
    env = ColdChainEnv(config=config)
    
    # Wrap with Monitor for tracking
    env = Monitor(env, log_dir, info_keywords=("reward_breakdown",))
    
    return env, log_dir


def train_ppo(env, log_dir, total_timesteps=100000):
    """
    Train PPO agent on the environment with edge case handling.
    
    Args:
        env: Gymnasium environment (wrapped with Monitor)
        log_dir: Directory for logs
        total_timesteps: Total training timesteps (default: 100K)
    
    Returns:
        Trained PPO model
    
    Raises:
        RuntimeError: If training fails
        ValueError: If parameters are invalid
    """
    # Edge case: Validate parameters
    if total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be > 0, got {total_timesteps}")
    if total_timesteps < 1000:
        print(f"⚠ WARNING: total_timesteps={total_timesteps} is very small (recommended: ≥10,000)")
    
    print("\n" + "="*70)
    print("COLDCHAIN-GYM PPO TRAINING")
    print("="*70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Log directory: {log_dir}")
    print("="*70 + "\n")
    
    try:
        # Initialize PPO with error handling
        try:
            ent_coef_schedule = get_linear_fn(0.05, 0.01, 0.5)
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,  # Print training progress
                learning_rate=3e-4,
                batch_size=64,
                n_steps=2048,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=ent_coef_schedule,
            )
            print("✓ PPO model initialized\n")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PPO: {e}")
        
        # Train with error handling
        print("Starting training...\n")
        try:
            model.learn(total_timesteps=total_timesteps)
            print("\n✓ Training completed successfully")
        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")
            # Save what we have so far
            print("Saving partial model...")
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
        
        # Save model with error handling
        try:
            model_path = os.path.join(log_dir, "ppo_coldchain")
            model.save(model_path)
            
            # Verify file exists
            if not os.path.exists(f"{model_path}.zip"):
                raise RuntimeError("Model file not created after save()")
            
            file_size = os.path.getsize(f"{model_path}.zip")
            print(f"✓ Model saved to {model_path}.zip ({file_size/1024/1024:.1f} MB)")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
        
        return model
    
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        raise


def validate_model(model, num_episodes=20):
    """
    Run validation loop with deterministic policy.
    Includes comprehensive edge case handling.
    
    Args:
        model: Trained PPO model
        num_episodes: Number of validation episodes (default: 20)
    
    Returns:
        Dict with validation results
    
    Raises:
        RuntimeError: If validation fails critically
    """
    print("\n" + "="*70)
    print(f"VALIDATION: {num_episodes} Episodes (Deterministic)")
    print("="*70 + "\n")
    
    # Edge case: invalid num_episodes
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be > 0, got {num_episodes}")
    
    # Create fresh environment for validation
    try:
        config = ColdChainConfig()
        env = ColdChainEnv(config=config)
    except Exception as e:
        raise RuntimeError(f"Failed to create environment: {e}")
    
    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "grader_scores": [],
        "reward_breakdowns": {
            "r_temp": [],
            "r_progress": [],
            "r_cost": [],
            "r_idle": [],
        },
        "failed_episodes": 0,
    }
    
    episodes_completed = 0
    
    for episode in range(num_episodes):
        try:
            # Edge case: Reset with different seeds
            try:
                obs, info = env.reset(seed=42 + episode)
                if obs is None:
                    print(f"Episode {episode+1:2d}  |  ✗ FAILED: Reset returned None")
                    results["failed_episodes"] += 1
                    continue
            except Exception as e:
                print(f"Episode {episode+1:2d}  |  ✗ FAILED: Reset error: {e}")
                results["failed_episodes"] += 1
                continue
            
            trajectory = []
            episode_reward = 0.0
            episode_length = 0
            max_steps = config.max_steps + 10  # Safety margin
            
            done = False
            while not done and episode_length < max_steps:
                try:
                    # Edge case: Predict could fail
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Edge case: Invalid action
                    if action is None:
                        print(f"Episode {episode+1:2d}  |  ✗ FAILED: Model returned None action")
                        results["failed_episodes"] += 1
                        break
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Edge case: NaN reward
                    if not np.isfinite(reward):
                        print(f"Episode {episode+1:2d}  |  ✗ FAILED: Non-finite reward {reward}")
                        results["failed_episodes"] += 1
                        break
                    
                    trajectory.append((obs, action, reward, info))
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                    
                except Exception as e:
                    print(f"Episode {episode+1:2d}  |  ✗ FAILED: Step error: {e}")
                    results["failed_episodes"] += 1
                    break
            
            # Edge case: Empty trajectory
            if not trajectory:
                print(f"Episode {episode+1:2d}  |  ✗ FAILED: Empty trajectory")
                results["failed_episodes"] += 1
                continue
            
            # Edge case: Episode too short
            if episode_length < 10:
                print(f"Episode {episode+1:2d}  |  ⚠ WARNING: Very short episode ({episode_length} steps)")
            
            # Compute grader score with error handling
            try:
                grader_score = CompositeGrader(trajectory).score()
                
                # Edge case: Invalid grader score
                if not (0.0 <= grader_score <= 1.0):
                    print(f"Episode {episode+1:2d}  |  ✗ FAILED: Invalid grader score {grader_score}")
                    results["failed_episodes"] += 1
                    continue
                
                # Edge case: NaN grader score
                if not np.isfinite(grader_score):
                    print(f"Episode {episode+1:2d}  |  ✗ FAILED: Non-finite grader score {grader_score}")
                    results["failed_episodes"] += 1
                    continue
                    
            except Exception as e:
                print(f"Episode {episode+1:2d}  |  ✗ FAILED: Grader error: {e}")
                results["failed_episodes"] += 1
                continue
            
            results["episode_rewards"].append(episode_reward)
            results["episode_lengths"].append(episode_length)
            results["grader_scores"].append(grader_score)
            
            # Extract reward breakdown from first step's info
            if trajectory and len(trajectory) > 0:
                first_info = trajectory[0][3]
                rb = first_info.get("reward_breakdown", {})
                if rb:
                    for key in results["reward_breakdowns"]:
                        val = rb.get(key, 0.0)
                        if np.isfinite(val):
                            results["reward_breakdowns"][key].append(val)
            
            # Print episode result
            print(f"Episode {episode+1:2d}  |  "
                  f"Reward: {episode_reward:7.4f}  |  "
                  f"Length: {episode_length:3d}  |  "
                  f"Grader Score: {grader_score:.4f}")
            
            episodes_completed += 1
            
        except KeyboardInterrupt:
            print(f"\n✗ Validation interrupted by user at episode {episode+1}")
            break
        except Exception as e:
            print(f"Episode {episode+1:2d}  |  ✗ FAILED: Unexpected error: {e}")
            results["failed_episodes"] += 1
            continue
    
    env.close()
    
    # Edge case: All episodes failed
    if episodes_completed == 0:
        raise RuntimeError("All validation episodes failed! Check environment and model.")
    
    print(f"\nCompleted {episodes_completed}/{num_episodes} episodes ({results['failed_episodes']} failed)")
    
    return results


def print_validation_summary(results):
    """Print summary statistics from validation with error handling."""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    # Edge case: Check if we have any valid results
    if not results["episode_rewards"]:
        print("\n✗ ERROR: No valid episodes completed!")
        print(f"  Failed episodes: {results.get('failed_episodes', '?')}")
        print("  Check environment logs above for details.")
        return
    
    # Averages with validation
    episode_rewards = np.array(results["episode_rewards"])
    episode_lengths = np.array(results["episode_lengths"])
    grader_scores = np.array(results["grader_scores"])
    
    # Edge case: Sanity check arrays
    if not (np.all(np.isfinite(episode_rewards)) and 
            np.all(np.isfinite(episode_lengths)) and 
            np.all(np.isfinite(grader_scores))):
        print("\n✗ ERROR: Invalid values in results!")
        print(f"  Finite rewards: {np.sum(np.isfinite(episode_rewards))}/{len(episode_rewards)}")
        print(f"  Finite lengths: {np.sum(np.isfinite(episode_lengths))}/{len(episode_lengths)}")
        print(f"  Finite scores: {np.sum(np.isfinite(grader_scores))}/{len(grader_scores)}")
        return
    
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_grader_score = np.mean(grader_scores)
    
    std_reward = np.std(episode_rewards)
    std_grader_score = np.std(grader_scores)
    
    print(f"\nCompleted Episodes: {len(episode_rewards)}")
    if results.get("failed_episodes", 0) > 0:
        print(f"Failed Episodes: {results['failed_episodes']}")
    
    print(f"\nAverage Episode Reward:        {avg_reward:.4f} (±{std_reward:.4f})")
    print(f"Average Episode Length:        {avg_length:.1f} steps")
    print(f"Average CompositeGrader Score: {avg_grader_score:.4f} (±{std_grader_score:.4f})")
    
    # Ranges with validation
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_score = np.min(grader_scores)
    max_score = np.max(grader_scores)
    
    print(f"\nReward Range:     [{min_reward:.4f}, {max_reward:.4f}]")
    print(f"Grader Range:     [{min_score:.4f}, {max_score:.4f}]")
    
    # Edge case: Check score bounds
    if not (np.all(grader_scores >= 0.0) and np.all(grader_scores <= 1.0)):
        print("\n⚠ WARNING: Some grader scores outside [0, 1] range!")
    
    # Reward breakdown
    print("\n" + "-"*70)
    print("REWARD BREAKDOWN (First Step Averages)")
    print("-"*70)
    
    has_breakdown = False
    for key in results["reward_breakdowns"]:
        values = results["reward_breakdowns"][key]
        if values and len(values) > 0:
            has_breakdown = True
            avg_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"  {key:15s}: {avg_val:7.4f} (±{std_val:.4f}) [{min_val:.4f}, {max_val:.4f}]")
    
    if not has_breakdown:
        print("  (No reward breakdown data available)")
    
    # Learning success check
    print("\n" + "-"*70)
    print("LEARNING SUCCESS CHECK")
    print("-"*70)
    
    # Edge case: Check if score is valid for comparison
    if not np.isfinite(avg_grader_score):
        print(f"✗ ERROR: Invalid average grader score: {avg_grader_score}")
        return
    
    # Success criteria
    success_threshold = 0.5
    if avg_grader_score > success_threshold:
        print(f"✓ CompositeGrader score ({avg_grader_score:.4f}) > {success_threshold}")
        print("  LEARNING SUCCESSFUL!")
    else:
        print(f"✗ CompositeGrader score ({avg_grader_score:.4f}) <= {success_threshold}")
        print("  WARNING: Model may not be learning effectively")
        print("  Recommendations:")
        print("    - Train for longer (increase total_timesteps)")
        print("    - Check reward function design")
        print("    - Verify action masking is working")
        print("    - Increase learning rate slightly")
    
    # Edge case: Very low min score
    if min_score < 0.1:
        print(f"\n⚠ WARNING: Minimum grader score very low ({min_score:.4f})")
        print("  Some episodes performed very poorly")
    
    # Edge case: High variance
    if std_grader_score > 0.2:
        print(f"\n⚠ WARNING: High variance in grader scores (std: {std_grader_score:.4f})")
        print("  Performance is unstable - may need more training")
    
    print("="*70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on ColdChain-Gym with pre-training safety checks")
    parser.add_argument("--precheck-only", action="store_true", help="Run pre-training checks only and exit")
    parser.add_argument("--micro-sanity", action="store_true", help="Run 500-1000 step MaskablePPO sanity run and exit")
    parser.add_argument("--micro-timesteps", type=int, default=960, help="Timesteps for micro sanity run (500-1000)")
    parser.add_argument("--micro-n-steps", type=int, default=64, help="n_steps for micro sanity run")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total PPO training timesteps")
    parser.add_argument("--precheck-episodes", type=int, default=3, help="Number of random episodes for precheck")
    parser.add_argument("--precheck-steps", type=int, default=200, help="Max random steps per precheck episode")
    return parser.parse_args()


def main():
    """Main training and validation pipeline with comprehensive error handling."""
    args = parse_args()
    try:
        # Mandatory pre-training checks
        print("\n[PHASE 0/4] Running pre-training sanity checks...")
        run_pretraining_sanity_checks(
            config=ColdChainConfig(),
            random_steps=args.precheck_steps,
            episodes=args.precheck_episodes,
        )
        print("✓ Pre-training checks complete")

        if args.precheck_only:
            print("\nPrecheck-only mode enabled. Exiting without training.")
            return True

        if args.micro_sanity:
            print("\n[PHASE 0.5/4] Running micro training sanity checks...")
            summary = run_micro_training_sanity(timesteps=args.micro_timesteps, n_steps=args.micro_n_steps)
            all_ok = summary["reward_up"] and summary["length_down"] and summary["entropy_ok"]
            if not all_ok:
                print("✗ Micro sanity trend checks failed. Reward/env/policy pipeline likely needs fixes before full training.")
                return False
            print("✓ Micro sanity trend checks passed.")
            print("Micro-sanity mode enabled. Exiting without full training.")
            return True

        # Setup phase with error handling
        print("\n[PHASE 1/4] Setting up environment...")
        try:
            env, log_dir = setup_environment()
            print("✓ Environment setup complete")
        except Exception as e:
            print(f"✗ Environment setup failed: {e}")
            raise
        
        # Training phase with error handling
        print("\n[PHASE 2/4] Training PPO model...")
        try:
            model = train_ppo(env, log_dir, total_timesteps=args.timesteps)
            print("✓ Training complete")
        except Exception as e:
            print(f"✗ Training failed: {e}")
            env.close()
            raise
        
        env.close()
        
        # Validation phase with error handling
        print("\n[PHASE 3/4] Running validation...")
        try:
            results = validate_model(model, num_episodes=20)
            print("✓ Validation complete")
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            raise
        
        # Summary phase with error handling
        print("\n[PHASE 4/4] Generating summary...")
        try:
            print_validation_summary(results)
            print("✓ Summary generated")
        except Exception as e:
            print(f"✗ Summary generation failed: {e}")
            raise
        
        # Final status
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Model saved to: ./logs/ppo_coldchain.zip")
        print(f"Training logs in: ./logs/")
        print("="*70 + "\n")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user (Ctrl+C)")
        return False
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
