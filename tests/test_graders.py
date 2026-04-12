from server.env import ColdChainEnv
from graders import (
    BasicGrader,
    CarrierThermalLoadGrader,
    CompositeGrader,
    DeliverySuccessGrader,
    EfficiencyGrader,
    HardGrader,
    ModerateGrader,
    ThermalIntegrityGrader,
)


def _first_legal_action(env: ColdChainEnv):
    mask = env.action_masks()
    flat_index = int(mask.argmax())
    vehicle = flat_index // (6 * env.config.n_nodes)
    remainder = flat_index % (6 * env.config.n_nodes)
    action_type = remainder // env.config.n_nodes
    target = remainder % env.config.n_nodes
    return [vehicle, action_type, target]


def test_grader_range():
    env = ColdChainEnv()
    obs, info = env.reset(seed=42)
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(_first_legal_action(env))
        if terminated or truncated:
            break

    trajectory = env._trajectory
    for grader_cls in [
        DeliverySuccessGrader,
        ThermalIntegrityGrader,
        EfficiencyGrader,
        CarrierThermalLoadGrader,
        BasicGrader,
        ModerateGrader,
        HardGrader,
        CompositeGrader,
    ]:
        score = grader_cls(trajectory).score()
        assert 0.0 <= score <= 1.0
