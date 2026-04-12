import numpy as np
from server.env import ColdChainEnv
from graders import CompositeGrader


def main():
    """
    Random Agent Example for ColdChain-Gym (Phase 12.1)
    
    Demonstrates:
    - Environment instantiation
    - Episode loop with reset/step
    - Grader scoring at end
    - Episode summary printing
    """
    env = ColdChainEnv()
    obs, info = env.reset(seed=42)
    trajectory = []
    
    print("=" * 60)
    print("ColdChain-Gym Random Agent Example")
    print("=" * 60)
    print(f"Config: {env.config.n_vehicles} vehicles, {env.config.max_shipments} max shipments")
    print(f"Max steps per episode: {env.config.max_steps}")
    print("-" * 60)

    steps = 0
    while True:
        # Sample random action from action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((obs, action, reward, info))
        steps += 1
        
        if steps % 50 == 0:
            print(f"Step {steps}: reward={reward:.4f}")
        
        if terminated or truncated:
            break

    # Compute grader scores
    composite_score = CompositeGrader(trajectory).score()
    
    print("-" * 60)
    print(f"Episode completed in {steps} steps")
    print(f"Composite Grader Score: {composite_score:.4f}")
    
    if "grader_scores" in info:
        print("\nGrader Breakdown:")
        for grader_name, score in info["grader_scores"].items():
            print(f"  {grader_name}: {score:.4f}")
    
    if "episode_summary" in info:
        summary = info["episode_summary"]
        print("\nEpisode Summary:")
        print(f"  Shipments delivered: {summary.get('shipments_delivered', 0)}")
        print(f"  Shipments destroyed: {summary.get('shipments_destroyed', 0)}")
        print(f"  Total reward: {summary.get('total_reward', 0):.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
