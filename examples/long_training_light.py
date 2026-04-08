import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from server.env import ColdChainEnv, CurriculumWrapper
from core.config import ColdChainConfig

def main():
    print("🚀 Starting LIGHTWEIGHT LONG TRAINING (250,000 steps)")
    print("Optimization: Low n_epochs and batch_size to save CPU.")
    
    # 1. Environment Setup
    config = ColdChainConfig(
        n_vehicles=1, 
        n_nodes=10, 
        max_steps=200,
        penalty_anneal_steps=50000 # Stretch annealing for long run
    )
    
    env = ColdChainEnv(config=config)
    env = CurriculumWrapper(env)
    env = Monitor(env)
    env = gym.wrappers.FlattenObservation(env)
    
    # 2. Lightweight PPO Configuration
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=32,       # Smaller batch = lighter on CPU/RAM
        n_epochs=4,          # Fewer passes per update to save CPU
        ent_coef=0.05,
        learning_rate=1e-4,
        gamma=0.995,
        target_kl=0.015,     # Early stopping for stability
        verbose=1
    )
    
    # 3. Training Loop
    try:
        model.learn(total_timesteps=250000, log_interval=10)
        model.save("ppo_coldchain_light_250k")
        print("\n✓ Training COMPLETE. Model saved as ppo_coldchain_light_250k.zip")
    except KeyboardInterrupt:
        print("\nSaving partial progress...")
        model.save("ppo_coldchain_light_interrupted")

if __name__ == "__main__":
    main()
