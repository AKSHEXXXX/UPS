import pytest
import numpy as np

from core.config import ColdChainConfig

try:
    from evaluation.eval_contract import build_eval_env, run_eval_episode
    HAS_EVAL_CONTRACT = True
except Exception:
    HAS_EVAL_CONTRACT = False


class FirstLegalModel:
    def predict(self, obs, action_masks=None, deterministic=True):
        legal = np.flatnonzero(action_masks)
        return int(legal[0]), None


@pytest.mark.skipif(not HAS_EVAL_CONTRACT, reason="eval contract dependencies not available")
def test_run_eval_episode_contract_fields():
    config = ColdChainConfig(n_vehicles=1, n_nodes=10, n_shipments=1, max_steps=40)
    env = build_eval_env(config)
    model = FirstLegalModel()

    result = run_eval_episode(
        model,
        env,
        seed=123,
        deterministic=True,
        curriculum_difficulty=3,
        evaluation_training_step=50000,
        collect_trajectory=True,
    )

    assert result["seed"] == 123
    assert result["difficulty"] == 3
    assert result["evaluation_training_step"] == 50000
    assert result["deterministic"] is True
    assert result["steps"] > 0
    assert result["initial_legal_actions"] > 0
    assert len(result["trajectory"]) == result["steps"]
    assert sum(result["action_type_counts"].values()) == result["steps"]

    env.close()


@pytest.mark.skipif(not HAS_EVAL_CONTRACT, reason="eval contract dependencies not available")
def test_run_eval_episode_respects_curriculum_difficulty():
    config = ColdChainConfig(n_vehicles=1, n_nodes=10, n_shipments=1, max_steps=25)
    env = build_eval_env(config)
    model = FirstLegalModel()

    result_easy = run_eval_episode(
        model,
        env,
        seed=11,
        deterministic=True,
        curriculum_difficulty=1,
        evaluation_training_step=30000,
    )
    result_hard = run_eval_episode(
        model,
        env,
        seed=11,
        deterministic=True,
        curriculum_difficulty=3,
        evaluation_training_step=30000,
    )

    assert result_easy["difficulty"] == 1
    assert result_hard["difficulty"] == 3
    assert result_easy["initial_legal_actions"] <= result_hard["initial_legal_actions"]

    env.close()