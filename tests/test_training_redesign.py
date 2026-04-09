from collections import Counter

import numpy as np
import pytest

from algorithms.ppo_training import (
    _base_training_space_config,
    _classify_failure,
    _configure_curriculum_wrapper,
    _load_imitation_dataset,
    _phase_difficulty_weights,
    _select_harvest_candidates,
    _training_scenario_library,
)
from core.reward import partial_delivery_terminal_reward
from server.env import ColdChainEnv, CurriculumWrapper


def test_curriculum_can_reach_all_five_difficulties():
    env = CurriculumWrapper(ColdChainEnv())
    env.max_difficulty = 5
    env.difficulty_weights = {level: 1.0 for level in range(1, 6)}

    seen = set()
    try:
        for seed in range(200):
            _, info = env.reset(seed=seed)
            seen.add(int(info["curriculum_difficulty"]))
    finally:
        env.close()

    assert seen == {1, 2, 3, 4, 5}


def test_phase3_curriculum_weights_favor_hard_and_extreme():
    weights = _phase_difficulty_weights(3)
    assert weights[4] == 0.35
    assert weights[5] == 0.50
    assert weights[4] + weights[5] == pytest.approx(0.85)
    assert weights[1] + weights[2] + weights[3] == pytest.approx(0.15)


def test_phase3_curriculum_sampling_matches_configured_bias():
    env = CurriculumWrapper(ColdChainEnv(config=_base_training_space_config(3)))
    env = _configure_curriculum_wrapper(env, phase=3)

    counts = Counter()
    try:
        for seed in range(400):
            _, info = env.reset(seed=seed)
            counts[int(info["curriculum_difficulty"])] += 1
    finally:
        env.close()

    assert counts[5] > counts[4] > counts[3]
    assert counts[4] + counts[5] > counts[1] + counts[2] + counts[3]


def test_training_scenario_library_matches_target_scales():
    library = _training_scenario_library()

    medium_like = library[3][0]
    hard_like = library[4][0]
    extreme_like = library[5][0]

    assert medium_like["active_vehicle_count"] == 2
    assert medium_like["active_shipment_count"] == 3
    assert hard_like["active_vehicle_count"] == 3
    assert hard_like["active_shipment_count"] == 5
    assert extreme_like["active_vehicle_count"] == 5
    assert extreme_like["active_shipment_count"] == 8


def test_adversarial_variants_are_confined_to_hard_and_extreme():
    library = _training_scenario_library()

    for difficulty in (1, 2, 3):
        for scenario in library[difficulty]:
            assert "forced_breakdowns" not in scenario

    assert any("forced_breakdowns" in scenario for scenario in library[4])
    assert any("forced_breakdowns" in scenario for scenario in library[5])


def test_partial_delivery_terminal_reward_orders_outcomes():
    env = ColdChainEnv()
    env.reset(seed=0, options={"active_shipment_count": 4})
    shipments = env.shipments

    for shipment in shipments:
        shipment.is_delivered = False
        shipment.is_destroyed = False

    zero_delivery_reward = partial_delivery_terminal_reward(shipments)
    assert zero_delivery_reward < 0.0

    shipments[0].is_delivered = True
    shipments[1].is_delivered = True
    partial_reward = partial_delivery_terminal_reward(shipments)

    for shipment in shipments[:4]:
        shipment.is_delivered = True
    full_reward = partial_delivery_terminal_reward(shipments)

    env.close()

    assert partial_reward > zero_delivery_reward
    assert full_reward > partial_reward


def test_partial_delivery_terminal_reward_ignores_single_shipment_cases():
    env = ColdChainEnv()
    env.reset(seed=0, options={"active_shipment_count": 1})
    reward = partial_delivery_terminal_reward(env.shipments)
    env.close()
    assert reward == 0.0


def test_select_harvest_candidates_keeps_successes_and_percentile(monkeypatch):
    # Force deterministic ranking for this test so selection logic is isolated.
    score_map = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}

    def fake_quality(result, difficulty):
        _ = difficulty
        return score_map[result["id"]]

    monkeypatch.setattr("algorithms.ppo_training._normalized_episode_quality", fake_quality)

    tier_results = [
        {"id": "a", "delivery_success": False},
        {"id": "b", "delivery_success": True},
        {"id": "c", "delivery_success": False},
        {"id": "d", "delivery_success": True},
        {"id": "e", "delivery_success": False},
    ]

    selected = _select_harvest_candidates(tier_results, difficulty=5, top_percentile=0.4)
    selected_ids = [row["id"] for row in selected]

    # Always keep successful episodes.
    assert "b" in selected_ids
    assert "d" in selected_ids
    # Also keep the top percentile of scored episodes.
    assert "a" in selected_ids
    assert "c" not in selected_ids


def test_classify_failure_timeout_and_masked_and_refrigeration_paths():
    timeout_many = {
        "termination_reason": "max_steps",
        "final_info": {
            "per_shipment_status": {
                "0": {"is_active": True, "is_delivered": True},
                "1": {"is_active": True, "is_delivered": True},
                "2": {"is_active": True, "is_delivered": False},
                "3": {"is_active": True, "is_delivered": False},
            }
        },
    }
    timeout_few = {
        "termination_reason": "max_steps",
        "final_info": {
            "per_shipment_status": {
                "0": {"is_active": True, "is_delivered": True},
                "1": {"is_active": True, "is_delivered": False},
                "2": {"is_active": True, "is_delivered": False},
                "3": {"is_active": True, "is_delivered": False},
            }
        },
    }
    masked_heavy = {
        "termination_reason": "terminated",
        "steps": 10,
        "final_info": {"illegal_action_count": 2},
    }
    refrig_mishandling = {
        "termination_reason": "terminated",
        "steps": 10,
        "final_info": {
            "illegal_action_count": 0,
            "per_vehicle_status": {
                "0": {"refrig_status": 2, "shipments_onboard": [1]},
            },
        },
    }
    all_destroyed = {"termination_reason": "all_destroyed"}

    assert _classify_failure(all_destroyed) == "all_destroyed"
    assert _classify_failure(timeout_many) == "timeout_many_delivered"
    assert _classify_failure(timeout_few) == "timeout_few_delivered"
    assert _classify_failure(masked_heavy) == "illegal_masked_action_heavy"
    assert _classify_failure(refrig_mishandling) == "refrigeration_mishandling"


def test_load_imitation_dataset_preserves_shapes_and_dtypes(tmp_path):
    dataset_path = tmp_path / "imitation_dataset.npz"
    np.savez_compressed(
        dataset_path,
        obs=np.zeros((3, 8), dtype=np.float64),
        actions=np.array([1, 2, 3], dtype=np.int32),
        masks=np.ones((3, 12), dtype=np.int16),
        weights=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        tiers=np.array([4, 5, 4], dtype=np.int32),
        families=np.array(["hard_clean", "extreme_clean", "hard_heatwave"], dtype=object),
        action_ranks=np.array([1, 2, 3], dtype=np.int32),
        action_probs=np.array([0.6, 0.3, 0.1], dtype=np.float64),
    )

    loaded = _load_imitation_dataset(str(dataset_path))

    assert loaded["obs"].dtype == np.float32
    assert loaded["actions"].dtype == np.int64
    assert loaded["masks"].dtype == np.int8
    assert loaded["weights"].dtype == np.float32
    assert loaded["tiers"].dtype == np.int64
    assert loaded["families"].dtype == object
    assert loaded["action_ranks"].dtype == np.int64
    assert loaded["action_probs"].dtype == np.float32
    assert loaded["obs"].shape == (3, 8)
    assert loaded["masks"].shape == (3, 12)


def test_legacy_promotion_disabled_when_difficulty_weights_set(monkeypatch):
    wrapped = CurriculumWrapper(ColdChainEnv())
    wrapped.difficulty = 2
    wrapped.success_count = 0
    wrapped.successes_to_advance = 1
    wrapped.max_difficulty = 5
    wrapped.difficulty_weights = {1: 0.4, 2: 0.6}

    monkeypatch.setattr(
        wrapped.env,
        "step",
        lambda action: ({}, 0.0, True, False, {"delivery_success": True}),
    )

    _, _, terminated, _, info = wrapped.step(0)
    wrapped.close()

    assert terminated is True
    assert info["delivery_success"] is True
    # Weighted curriculum should disable legacy success-based promotion path.
    assert wrapped.success_count == 0
    assert wrapped.difficulty == 2


def test_legacy_promotion_active_without_difficulty_weights(monkeypatch):
    wrapped = CurriculumWrapper(ColdChainEnv())
    wrapped.difficulty = 2
    wrapped.success_count = 0
    wrapped.successes_to_advance = 1
    wrapped.max_difficulty = 5
    wrapped.difficulty_weights = None

    monkeypatch.setattr(
        wrapped.env,
        "step",
        lambda action: ({}, 0.0, True, False, {"delivery_success": True}),
    )

    wrapped.step(0)
    wrapped.close()

    assert wrapped.success_count == 0
    assert wrapped.difficulty == 3
