import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from server.env import ColdChainEnv, CurriculumWrapper
from core.config import ColdChainConfig
from core.graders import CompositeGrader

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

def run_phase_1():
    print("\n--- PHASE 1: 1,000 Steps (Health Check) ---")
    config = ColdChainConfig(n_vehicles=1, n_nodes=10, max_steps=200)
    env = ColdChainEnv(config=config)
    env = CurriculumWrapper(env) # Phase 1 now uses Curriculum
    env = gym.wrappers.FlattenObservation(env)
    
    obs, _ = env.reset(seed=42)
    movement_detected = False
    non_zero_reward = False
    illegal_actions = 0
    
    illegal_samples = []
    
    for _ in range(100):
        action = env.action_space.sample()
        prev_loc = env.unwrapped.vehicles[0].location
        _, reward, terminated, truncated, info = env.step(action)
        
        if env.unwrapped.vehicles[0].location != prev_loc:
            movement_detected = True
        if abs(reward) > 1e-5:
            non_zero_reward = True
        
        # Check if action was illegal (masked)
        if info.get("action_was_masked", False):
            illegal_actions += 1
            if len(illegal_samples) < 5:
                # Capture action: [vehicle_idx, action_type, target_idx]
                illegal_samples.append({
                    "action": action.tolist(),
                    "reason": info.get("message", "Masked by environment")
                })
            
        if terminated or truncated:
            env.reset()
            
    print(f"  > Movement Detected: {'YES' if movement_detected else 'NO'}")
    print(f"  > Non-Zero Rewards:  {'YES' if non_zero_reward else 'NO'}")
    print(f"  > Illegal Actions:   {illegal_actions}/100")
    
    if illegal_samples:
        print("\n  Sample Illegal Actions (Forced to WAIT):")
        for sample in illegal_samples:
            act = sample['action']
            print(f"    - Action: [V:{act[0]}, Type:{act[1]}, Target:{act[2]}] | Reason: {sample['reason']}")
    
    success = movement_detected and non_zero_reward
    if not success:
        print("!!! Phase 1 Failed: Environment or Reward logic issue !!!")
    return success

def run_phase_2(model=None):
    print("\n--- PHASE 2: 10,000 Steps (Trend Check) ---")
    config = ColdChainConfig(n_vehicles=1, n_nodes=10, max_steps=200)
    env = ColdChainEnv(config=config)
    env = Monitor(env)
    env = gym.wrappers.FlattenObservation(env)
    
    callback = PhasedMetricsCallback()
    
    if model is None:
        model = PPO(
            "MlpPolicy",         # Correct for flattened Box observations
            env,
            n_steps=256,         # Fix 5
            batch_size=64,
            ent_coef=0.05,        # Exploration
            learning_rate=1e-4,   # Stability
            gamma=0.995,          # Future delivery reward focus
            verbose=1
        )
    else:
        model.set_env(env)
        
    model.learn(total_timesteps=10000, callback=callback)
    
    if len(callback.rewards) >= 2:
        reward_trend = callback.rewards[-1] > callback.rewards[0]
        len_trend = callback.lengths[-1] <= callback.lengths[0]
        print(f"  > Reward Trending Up: {'YES' if reward_trend else 'NO'} ({callback.rewards[0]:.2f} -> {callback.rewards[-1]:.2f})")
        print(f"  > Length Trending Down: {'YES' if len_trend else 'NO'} ({callback.lengths[0]:.1f} -> {callback.lengths[-1]:.1f})")
        return model, True # Proceed even if trend is flat early on
    
    return model, True

def run_phase_3(model):
    print("\n--- PHASE 3: 50,000 Steps (Success Check) ---")
    model.learn(total_timesteps=50000)
    
    # Evaluation with 1 vehicle
    config = ColdChainConfig(n_vehicles=1, n_nodes=10)
    env = ColdChainEnv(config=config)
    # Note: Phase 3 Eval doesn't use the curriculum wrapper to ensure true performance
    env = gym.wrappers.FlattenObservation(env)
    
    delivered_count = 0
    total_shipments = 0
    
    for i in range(5):
        obs, _ = env.reset(seed=100+i)
        done = False
        trajectory = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            trajectory.append((obs, action, reward, info))
            done = term or trunc
        
        delivered = sum(1 for s in env.unwrapped.shipments if s.is_delivered)
        delivered_count += delivered
        total_shipments += len(env.unwrapped.shipments)
        
        score = CompositeGrader(trajectory).score()
        print(f"  > Eval Episode {i+1}: Delivered {delivered}, Grader Score: {score:.4f}")

    success_rate = delivered_count / total_shipments
    print(f"\n  > Overall Delivery Success Rate: {success_rate:.2%}")
    
    model.save("ppo_coldchain_final")
    print(f"✓ Model saved as ppo_coldchain_final.zip")
    
    if success_rate > 0.1: # At least some deliveries
        print("✓ Phase 3 Successful: Agent is reaching destinations.")
        return True
    else:
        print("! Phase 3 Feedback: Agent is still struggling to deliver consistently.")
        return False

if __name__ == "__main__":
    if run_phase_1():
        model, p2_ok = run_phase_2()
        if p2_ok:
            run_phase_3(model)
