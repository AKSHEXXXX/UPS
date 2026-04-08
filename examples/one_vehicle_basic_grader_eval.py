import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from server.env import ColdChainEnv
from core.config import ColdChainConfig
from core.graders.basic_grader import BasicGrader
from core.graders.delivery_success import DeliverySuccessGrader
from core.graders.thermal_integrity import ThermalIntegrityGrader

def main():
    print("\n" + "="*70)
    print("ONE-VEHICLE BASIC GRADER EVALUATION")
    print("="*70)

    # 1. Load the model
    model_path = "ppo_coldchain_light_250k.zip"
    if not os.path.exists(model_path):
        print(f"!!! Error: Model file {model_path} not found. Run phased_training.py first.")
        return

    try:
        model = PPO.load(model_path)
        print(f"✓ Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"!!! Error loading model: {e}")
        return

    # 2. Setup Environment (1 vehicle, 10 nodes to match training)
    config = ColdChainConfig(n_vehicles=1, n_nodes=10, max_steps=300)
    env = ColdChainEnv(config=config)
    env = gym.wrappers.FlattenObservation(env)

    # 3. Run Evaluation Episode
    obs, _ = env.reset(seed=42)
    trajectory = []
    done = False
    
    print("\nRunning evaluation episode...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log status
        vehicle = env.unwrapped.vehicles[0]
        shipments = env.unwrapped.shipments
        temp = shipments[0].cargo_temp if shipments else 0.0
        print(f"Step {env.unwrapped.steps_elapsed}: Loc={vehicle.location}, Temp={temp:.2f}, Reward={reward:.4f}, Action={action}")
        
        trajectory.append((obs, action, reward, info))
        done = terminated or truncated

    # 4. Grading
    print("\n" + "-"*40)
    print("BASIC GRADER RESULTS")
    print("-"*40)
    
    grader = BasicGrader(trajectory)
    composite = grader.score()
    
    delivery_score = DeliverySuccessGrader(trajectory).score()
    thermal_score = ThermalIntegrityGrader(trajectory).score()
    
    print(f"  Delivery Success    : {delivery_score:.4f}")
    print(f"  Thermal Integrity   : {thermal_score:.4f}")
    
    print("-"*40)
    print(f"  FINAL COMPOSITE SCORE: {composite:.4f}")
    print("="*70 + "\n")

    env.close()

if __name__ == "__main__":
    import os
    main()
