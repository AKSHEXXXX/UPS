from core.models import ColdChainAction
from server.environment import ColdChainEnvironment


def test_openenv_reset_and_step():
    env = ColdChainEnvironment()
    obs = env.reset(seed=0)
    assert obs.done is False
    assert obs.reward == 0.0
    assert len(obs.vehicles) == env.config.n_vehicles
    assert len(obs.shipments) == env.config.n_shipments

    action = ColdChainAction(vehicle_index=0, action_type=0, target_index=0)
    next_obs = env.step(action)
    assert next_obs.reward is not None
    assert env.state.step_count == 1
