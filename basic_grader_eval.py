import os
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from server.env import ColdChainEnv
from core.config import ColdChainConfig
from core.graders.basic_grader import BasicGrader
from core.graders.delivery_success import DeliverySuccessGrader
from core.graders.thermal_integrity import ThermalIntegrityGrader

def get_mask(env):
    return env.unwrapped.action_masks()

def main():
    print("\n" + "="*70)
    print("STABILIZED MODEL BASIC GRADER EVALUATION")
    print("="*70)

    # 1. Load the model
    model_path = "models/ppo_phase3.zip"
    if not os.path.exists(model_path):
        print(f"!!! Error: Model file {model_path} not found. Run ppo_training.py phase 3 first.")
        # Try finding any phase model
        for p in ["models/ppo_phase2.zip", "models/ppo_phase1.zip"]:
             if os.path.exists(p):
                 model_path = p
                 print(f"Using fallback model: {model_path}")
                 break
        else:
            return

    try:
        model = MaskablePPO.load(model_path)
        print(f"✓ Successfully loaded MaskablePPO model: {model_path}")
    except Exception as e:
        print(f"!!! Error loading model: {e}")
        return

    # 2. Setup Environment (1 vehicle, 10 nodes to match training)
    config = ColdChainConfig(n_vehicles=1, n_nodes=10, max_steps=300)
    env = ColdChainEnv(config=config)
    env = ActionMasker(env, get_mask)
    env = gym.wrappers.FlattenObservation(env)

    # 3. Run Evaluation Episode
    obs, _ = env.reset(seed=42)
    trajectory = []
    done = False
    
    print("\nRunning evaluation episode...")
    while not done:
        # Use stochastic prediction for better exploration during eval if not converged
        action, _ = model.predict(obs, action_masks=env.unwrapped.action_masks(), deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log status
        vehicle = env.unwrapped.vehicles[0]
        shipments = env.unwrapped.shipments
        temp = shipments[0].cargo_temp if shipments else 0.0
        
        # Decode action for printing
        n_nodes = config.n_nodes
        v_idx = action // (6 * n_nodes)
        a_type = (action % (6 * n_nodes)) // n_nodes
        t_idx = action % n_nodes
        
        if env.unwrapped.steps_elapsed % 10 == 0 or terminated or truncated:
            print(f"Step {env.unwrapped.steps_elapsed:3}: Loc={vehicle.location:2}, Temp={temp:5.2f}, Reward={reward:7.2f}, Action=[V{v_idx} A{a_type} T{t_idx}]")
        
        # Grading expects (obs, action, reward, info)
        # Note: reward here is the step reward
        trajectory.append((obs, action, reward, info))
        done = terminated or truncated

    # 4. Grading
    print("\n" + "-"*40)
    print("GRADER RESULTS")
    print("-"*40)
    
    # Simple check if there is data
    if not trajectory:
        print("No trajectory data collected.")
        return

    try:
        # Grader might need specific formatting of trajectory
        grader = BasicGrader(trajectory)
        composite = grader.score()
        
        delivery_score = DeliverySuccessGrader(trajectory).score()
        thermal_score = ThermalIntegrityGrader(trajectory).score()
        
        print(f"  Delivery Success    : {delivery_score:.4f}")
        print(f"  Thermal Integrity   : {thermal_score:.4f}")
        print("-"*40)
        print(f"  FINAL COMPOSITE SCORE: {composite:.4f}")
    except Exception as e:
        print(f"Error during grading: {e}")
        # Print summary anyway
        print(f"  Steps taken: {len(trajectory)}")
        print(f"  Final Reward: {sum(t[2] for t in trajectory):.2f}")

    print("="*70 + "\n")
    env.close()

if __name__ == "__main__":
    main()
