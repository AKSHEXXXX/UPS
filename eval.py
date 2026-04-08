import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from server.env import ColdChainEnv, CurriculumWrapper
from core.config import ColdChainConfig
from core.graders import CompositeGrader

def get_mask(env):
    return env.unwrapped.action_masks()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_difficulties", nargs="+", type=int, default=None)
    args = parser.parse_args()

    if args.test_difficulties:
        for diff in args.test_difficulties:
            print(f"\nEvaluating Difficulty {diff}")
            evaluate(args.checkpoint, args.episodes, diff, args.seed)
    else:
        evaluate(args.checkpoint, args.episodes, args.difficulty, args.seed)

def evaluate(checkpoint, episodes, difficulty, seed):
    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint {checkpoint} not found.")
        return

    # Setup env
    config = ColdChainConfig(n_vehicles=1, n_nodes=10) # Match training config
    env = ColdChainEnv(config=config)
    env = CurriculumWrapper(env)
    env = ActionMasker(env, get_mask) # Add masking wrapper
    env = gym.wrappers.FlattenObservation(env)
    
    print(f"Loading checkpoint: {checkpoint}")
    model = MaskablePPO.load(checkpoint)
    
    # Force curriculum difficulty on the wrapper
    curr_wrapper = env
    while hasattr(curr_wrapper, "env"):
        if isinstance(curr_wrapper, CurriculumWrapper):
            break
        curr_wrapper = curr_wrapper.env
    
    if isinstance(curr_wrapper, CurriculumWrapper):
        curr_wrapper.difficulty = difficulty
    else:
        # Fallback if wrapper not found
        env.unwrapped.difficulty = difficulty

    model = MaskablePPO.load(checkpoint)
    
    success_count = 0
    total_rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0
        nodes_visited = [env.unwrapped.vehicles[0].location]
        milestones = {"pickup": False, "depart": False, "halfway": False, "delivery": False}
        
        print(f"\nEpisode {ep + 1}:")
        while not done:
            action, _ = model.predict(obs, action_masks=env.unwrapped.action_masks(), deterministic=False)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            
            # Tracking
            loc = env.unwrapped.vehicles[0].location
            if loc != nodes_visited[-1]:
                nodes_visited.append(loc)
                
            # Check for milestones in info if available, or manually
            if info.get("message") == "Shipment delivered":
                milestones["delivery"] = True
            
            done = term or trunc

        # Manual milestone check for logging
        shipment = env.unwrapped.shipments[0]
        if shipment.is_delivered:
            milestones["delivery"] = True
            success_count += 1
        if shipment.current_vehicle_id != -1:
            milestones["pickup"] = True
        
        # Check node path for "depart" (left node 0)
        if len(nodes_visited) > 1 and nodes_visited[0] == 0:
            milestones["depart"] = True

        print(f"  Nodes visited: {nodes_visited}")
        print(f"  Milestones: {', '.join([k for k, v in milestones.items() if v])}")
        print(f"  Total Reward: {ep_reward:.2f}")
        print(f"  Status: {'SUCCESS' if milestones['delivery'] else 'FAILED'}")
        total_rewards.append(ep_reward)

    print(f"\nAverage Reward: {np.mean(total_rewards):.2f}")
    print(f"Success Rate: {success_count/episodes:.2%}")

if __name__ == "__main__":
    main()
