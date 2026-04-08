"""
Debug Test Suite for ColdChain-Gym
Tests reward functions, action masking, and observations systematically.
"""
import numpy as np
from server.env import ColdChainEnv
from core.reward import (
    action_cost_reward,
    catastrophe_event_reward,
    delivery_event_reward,
    idle_penalty_reward,
    progress_shaping_reward,
    temp_shaping_reward,
)
from core.graders import CompositeGrader


class TestRewardFunctions:
    """Test individual reward components."""

    def test_temp_shaping_reward(self):
        """Verify temperature shaping reward calculation."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        temp_reward = temp_shaping_reward(env.shipments, env.config)
        assert isinstance(temp_reward, float), f"temp_reward should be float, got {type(temp_reward)}"
        assert -10.0 <= temp_reward <= 10.0, f"temp_reward out of bounds: {temp_reward}"
        print(f"✓ temp_shaping_reward: {temp_reward:.4f}")

    def test_idle_penalty_reward(self):
        """Verify idle penalty calculation."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        idle_reward = idle_penalty_reward(env.vehicles, env.shipments, env.config)
        assert isinstance(idle_reward, float), f"idle_reward should be float, got {type(idle_reward)}"
        # All vehicles start IDLE, so should be negative
        assert idle_reward <= 0.0, f"idle_reward should be negative when all idle: {idle_reward}"
        print(f"✓ idle_penalty_reward: {idle_reward:.4f}")

    def test_action_cost_reward(self):
        """Test action cost for each action type."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        vehicle = env.vehicles[0]
        
        costs = {}
        for action_type in range(6):
            action = np.array([0, action_type, 0])
            cost = action_cost_reward(action, vehicle, env.config)
            assert isinstance(cost, float), f"action_cost should be float, got {type(cost)}"
            assert -10.0 <= cost <= 10.0, f"action_cost out of bounds: {cost}"
            costs[action_type] = cost
        
        # WAIT (0) should be free
        assert costs[0] == 0.0, f"WAIT should be free, got {costs[0]}"
        # Other actions should have negative cost (penalty)
        assert costs[1] < 0.0, f"REROUTE should cost negative, got {costs[1]}"
        assert costs[5] < 0.0, f"ABORT should cost negative, got {costs[5]}"
        
        print(f"✓ action_cost_reward: {costs}")

    def test_progress_shaping_reward(self):
        """Test progress reward when vehicles move toward destinations."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        prev_distances = {}
        progress = progress_shaping_reward(env.vehicles, env.shipments, env.graph, env.config, prev_distances)
        assert isinstance(progress, float), f"progress should be float, got {type(progress)}"
        print(f"✓ progress_shaping_reward: {progress:.4f}")

    def test_delivery_event_reward(self):
        """Test delivery bonus calculation."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        shipment = env.shipments[0]
        
        # Not delivered yet
        reward = delivery_event_reward(shipment, 0, env.config)
        assert reward == 0.0, f"Undelivered shipment should give 0 reward, got {reward}"
        
        # Mark as delivered on time with no excursions
        shipment.is_delivered = True
        shipment.excursion_count = 0
        reward = delivery_event_reward(shipment, 100, env.config)
        assert reward > 0.0, f"On-time delivery should be positive, got {reward}"
        
        print(f"✓ delivery_event_reward: {reward:.4f}")

    def test_catastrophe_event_reward(self):
        """Test destruction penalty."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        shipment = env.shipments[0]
        
        # Not destroyed
        reward = catastrophe_event_reward(shipment, env.config)
        assert reward == 0.0, f"Unharmed shipment should give 0 penalty, got {reward}"
        
        # Organ destroyed with excursion
        shipment.is_destroyed = True
        shipment.cargo_type = "organ"
        shipment.excursion_count = 1
        reward = catastrophe_event_reward(shipment, env.config)
        assert reward < -10.0, f"Organ destruction should be severe penalty, got {reward}"
        
        print(f"✓ catastrophe_event_reward: {reward:.4f}")


class TestObservationStructure:
    """Test observation space and values."""

    def test_observation_shape(self):
        """Verify observation has correct shape."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        assert "global" in obs, "Missing 'global' key"
        assert "vehicles" in obs, "Missing 'vehicles' key"
        assert "shipments" in obs, "Missing 'shipments' key"
        
        # Check shapes
        assert obs["vehicles"].shape == (env.config.n_vehicles, 9 + env.config.max_cargo_per_vehicle), \
            f"vehicle shape wrong: {obs['vehicles'].shape}"
        assert obs["shipments"].shape == (env.config.max_shipments, 13), \
            f"shipment shape wrong: {obs['shipments'].shape}"
        
        print(f"✓ observation_shape: vehicles={obs['vehicles'].shape}, shipments={obs['shipments'].shape}")

    def test_observation_dtypes(self):
        """Verify observations have correct dtypes (float32 or int32)."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        assert obs["vehicles"].dtype == np.float32, f"vehicles dtype wrong: {obs['vehicles'].dtype}"
        assert obs["shipments"].dtype == np.float32, f"shipments dtype wrong: {obs['shipments'].dtype}"
        
        # Global is now a flat float32 array (7 elements)
        assert obs["global"].dtype == np.float32, \
            f"global obs should be float32, got {obs['global'].dtype}"
        assert obs["global"].shape == (7,), \
            f"global obs should have shape (7,), got {obs['global'].shape}"
        
        print(f"✓ observation_dtypes: correct (float32)")

    def test_observation_bounds(self):
        """Verify observations are within valid bounds."""
        env = ColdChainEnv()
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            
            # Global flat array: [ambient_temp(0), time_of_day(1), traffic(2), weather(3), hub_temp(4), steps_elapsed(5), steps_remaining(6)]
            assert -40 <= obs["global"][0] <= 60, "ambient_temperature out of bounds"
            assert -10 <= obs["global"][4] <= 30, "hub_cold_storage_temp out of bounds"
            assert 0   <= obs["global"][1] <= 1,  "time_of_day out of bounds"
            assert 0   <= obs["global"][5] <= env.config.max_steps, "steps_elapsed out of bounds"
            assert 0   <= obs["global"][6] <= env.config.max_steps, "steps_remaining out of bounds"
            
            # Vehicle / shipment observations: no NaN
            assert not np.isnan(obs["vehicles"]).any(),  "NaN in vehicle observations"
            assert not np.isnan(obs["shipments"]).any(), "NaN in shipment observations"
        
        print(f"✓ observation_bounds: all within valid ranges")

    def test_observation_consistency_across_steps(self):
        """Verify observation shape stays consistent during episode."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        initial_v_shape = obs["vehicles"].shape
        initial_s_shape = obs["shipments"].shape
        
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert obs["vehicles"].shape == initial_v_shape, \
                f"vehicle shape changed at step {step}"
            assert obs["shipments"].shape == initial_s_shape, \
                f"shipment shape changed at step {step}"
            
            if terminated or truncated:
                break
        
        print(f"✓ observation_consistency: shape preserved across steps")


class TestActionMasking:
    """Test action masking system."""

    def test_action_mask_shape(self):
        """Verify action mask has correct shape."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        mask = env.action_masks()
        
        expected_size = (env.config.n_vehicles + 1) * 6 * env.config.n_nodes
        assert mask.shape == (expected_size,), f"mask shape wrong: {mask.shape}"
        assert mask.dtype == np.int8, f"mask dtype wrong: {mask.dtype}"
        
        print(f"✓ action_mask_shape: {mask.shape}")

    def test_action_mask_legal_actions_exist(self):
        """Verify at least one legal action per vehicle."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        mask = env.action_masks()
        
        for vehicle_idx in range(env.config.n_vehicles + 1):
            base = vehicle_idx * 6 * env.config.n_nodes
            end = base + 6 * env.config.n_nodes
            vehicle_mask = mask[base:end]
            
            assert vehicle_mask.sum() > 0, f"No legal actions for vehicle {vehicle_idx}"
        
        print(f"✓ action_mask_legal_actions: all vehicles have legal actions")

    def test_action_mask_wait_always_legal(self):
        """Verify WAIT action (0) is always legal."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        for step in range(10):
            mask = env.action_masks()
            
            # WAIT is action 0 for each vehicle
            for vehicle_idx in range(env.config.n_vehicles):
                base = vehicle_idx * 6 * env.config.n_nodes
                wait_idx = base  # action 0, target 0
                assert mask[wait_idx] == 1, f"WAIT not legal for vehicle {vehicle_idx}"
            
            # Global no-op always legal
            global_base = env.config.n_vehicles * 6 * env.config.n_nodes
            assert mask[global_base] == 1, "Global no-op not legal"
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print(f"✓ action_mask_wait_always_legal: WAIT consistently legal")

    def test_illegal_action_handling(self):
        """Verify illegal actions are forced to WAIT."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        # Try to create an illegal action
        mask = env.action_masks()
        illegal_idx = np.argmin(mask)  # Find a masked action
        
        if mask[illegal_idx] == 0:  # If we found an illegal action
            illegal_action = np.array([
                illegal_idx // (6 * env.config.n_nodes),
                (illegal_idx % (6 * env.config.n_nodes)) // env.config.n_nodes,
                illegal_idx % env.config.n_nodes
            ])
            
            obs, reward, terminated, truncated, info = env.step(illegal_action)
            
            assert info["action_was_masked"] == True, "Should marked illegal action"
            assert env._illegal_action_count > 0, "Should increment illegal count"
        
        print(f"✓ action_mask_illegal_handling: forced to WAIT")


class TestTrajectoryAndGraders:
    """Test trajectory structure and grader integration."""

    def test_trajectory_structure(self):
        """Verify trajectory has correct tuple structure."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        trajectory = env._trajectory
        assert len(trajectory) > 0, "Trajectory is empty"
        
        for obs, action, reward, info in trajectory:
            assert isinstance(obs, dict), "obs should be dict"
            assert isinstance(action, np.ndarray), "action should be ndarray"
            assert isinstance(reward, float), "reward should be float"
            assert isinstance(info, dict), "info should be dict"
        
        print(f"✓ trajectory_structure: {len(trajectory)} steps, valid tuples")

    def test_grader_scoring(self):
        """Verify graders work on trajectory."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        trajectory = env._trajectory
        score = CompositeGrader(trajectory).score()
        
        assert isinstance(score, float), f"Score should be float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"
        
        print(f"✓ grader_scoring: {score:.4f}")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_episode_termination(self):
        """Verify episode terminates correctly."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if terminated or truncated:
                break
            
            assert steps <= env.config.max_steps + 1, "Episode exceeded max_steps"
        
        assert steps > 0, "Episode ended immediately"
        print(f"✓ episode_termination: ended at step {steps}")

    def test_reward_clipping(self):
        """Verify rewards are clipped to [-10, 10]."""
        env = ColdChainEnv()
        obs, _ = env.reset(seed=0)
        
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert -10.0 <= reward <= 10.0, f"Reward out of bounds: {reward}"
            
            if terminated or truncated:
                break
        
        print(f"✓ reward_clipping: within [-10, 10]")

    def test_reproducibility_with_seed(self):
        """Verify same seed produces same state/observations (actions are sampled randomly)."""
        def get_initial_obs(seed):
            env = ColdChainEnv()
            obs, _ = env.reset(seed=seed)
            return obs["vehicles"].copy(), obs["shipments"].copy()
        
        obs1_v, obs1_s = get_initial_obs(42)
        obs2_v, obs2_s = get_initial_obs(42)
        
        # Check that same seed produces same initial observations
        assert np.allclose(obs1_v, obs2_v), "Initial vehicle obs should match"
        assert np.allclose(obs1_s, obs2_s), "Initial shipment obs should match"
        
        print(f"✓ reproducibility_with_seed: consistent resets")


if __name__ == "__main__":
    print("=" * 60)
    print("ColdChain-Gym Debug Test Suite")
    print("=" * 60)
    
    # Test reward functions
    print("\n[Reward Functions]")
    test_rewards = TestRewardFunctions()
    test_rewards.test_temp_shaping_reward()
    test_rewards.test_idle_penalty_reward()
    test_rewards.test_action_cost_reward()
    test_rewards.test_progress_shaping_reward()
    test_rewards.test_delivery_event_reward()
    test_rewards.test_catastrophe_event_reward()
    
    # Test observations
    print("\n[Observation Structure]")
    test_obs = TestObservationStructure()
    test_obs.test_observation_shape()
    test_obs.test_observation_dtypes()
    test_obs.test_observation_bounds()
    test_obs.test_observation_consistency_across_steps()
    
    # Test action masking
    print("\n[Action Masking]")
    test_mask = TestActionMasking()
    test_mask.test_action_mask_shape()
    test_mask.test_action_mask_legal_actions_exist()
    test_mask.test_action_mask_wait_always_legal()
    test_mask.test_illegal_action_handling()
    
    # Test trajectory and graders
    print("\n[Trajectory & Graders]")
    test_traj = TestTrajectoryAndGraders()
    test_traj.test_trajectory_structure()
    test_traj.test_grader_scoring()
    
    # Test edge cases
    print("\n[Edge Cases]")
    test_edge = TestEdgeCases()
    test_edge.test_episode_termination()
    test_edge.test_reward_clipping()
    test_edge.test_reproducibility_with_seed()
    
    print("\n" + "=" * 60)
    print("✓ All debug tests passed!")
    print("=" * 60)
