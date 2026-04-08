import os
import argparse
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from server.env import ColdChainEnv, CurriculumWrapper
from core.config import ColdChainConfig

def get_mask(env):
    return env.unwrapped.action_masks()
from core.graders import CompositeGrader

class EntropyAnnealingCallback(BaseCallback):
    """Anneals entropy bonus over time."""
    def __init__(self, start_ent=0.1, end_ent=0.01, steps_to_reach=20000, verbose=0):
        super().__init__(verbose)
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.steps_to_reach = steps_to_reach

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.steps_to_reach, 1.0)
        current_ent = self.start_ent + (self.end_ent - self.start_ent) * progress
        # Correctly set ent_coef in the model's policy
        self.model.ent_coef = current_ent
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

def run_training_phase(phase, steps, resume_path=None, seed=42):
    print(f"\n--- [Phase {phase}] Training for {steps} steps ---")
    
    config = ColdChainConfig(
        n_vehicles=1,
        n_nodes=10,
        max_steps=200,
        penalty_anneal_steps=30000
    )
    
    env = ColdChainEnv(config=config)
    env = CurriculumWrapper(env)
    env = ActionMasker(env, get_mask) # Add masking wrapper
    env = Monitor(env)
    env = gym.wrappers.FlattenObservation(env)
    
    anneal_callback = AnnealingCallback(steps_to_full=30000)
    ent_callback = EntropyAnnealingCallback(start_ent=0.2, end_ent=0.01, steps_to_reach=100000)
    callbacks = [anneal_callback, ent_callback, PhasedMetricsCallback()]
    
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        model = MaskablePPO.load(resume_path, env=env)
        # Update model parameters if needed
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=2048,
            batch_size=128,
            ent_coef=0.2, # Higher initial entropy 
            learning_rate=2e-4,
            gamma=0.99,
            verbose=1,
            seed=seed
        )
    
    model.learn(total_timesteps=steps, callback=callbacks)
    
    save_path = f"models/ppo_phase{phase}.zip"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"✓ Phase {phase} complete. Model saved to {save_path}")
    
    # Validation
    validate_model(model, config, seed)
    return model

def validate_model(model, config, seed):
    print("\n[Validation Check]")
    env = ColdChainEnv(config=config)
    # Validate on difficulty 3 to see true progress
    env = gym.wrappers.FlattenObservation(env)
    
    obs, _ = env.reset(seed=seed+100)
    done = False
    movement = False
    rewards = 0
    steps = 0
    start_loc = env.unwrapped.vehicles[0].location
    
    while not done and steps < 100:
        action, _states = model.predict(obs, action_masks=env.unwrapped.action_masks(), deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        rewards += reward
        steps += 1
        if env.unwrapped.vehicles[0].location != start_loc:
            movement = True
        done = term or trunc
        
    print(f"  > Steps: {steps}, Moved: {movement}, Total Reward: {rewards:.2f}")
    if not movement:
        print("  ! WARNING: Agent is stationary.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    phases = {
        1: 50000,
        2: 50000,
        3: 200000
    }
    
    run_training_phase(args.phase, phases[args.phase], args.resume, args.seed)
