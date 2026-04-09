from core.models import ColdChainAction
from core.client import ColdChainEnv


def _sample_observation_payload() -> dict:
    return {
        "done": True,
        "reward": 1.5,
        "global_state": {
            "ambient_temperature": 21.5,
            "time_of_day": 0.25,
            "traffic_multiplier": 1.0,
            "weather_event": 0,
            "hub_cold_storage_temp": 4.0,
            "steps_elapsed": 3,
            "steps_remaining": 7,
            "difficulty_level_norm": 0.8,
            "active_shipments_norm": 0.625,
            "active_vehicles_norm": 0.6,
        },
        "vehicles": [
            {
                "id": 0,
                "location": 1,
                "status": 0,
                "refrigeration_status": 0,
                "fuel_level": 0.9,
                "steps_until_next_waypoint": 0,
                "shipments_onboard": [0, -1],
                "nearest_cold_depot_node": 1,
                "steps_to_cold_depot": 2,
                "steps_to_destination": 5,
                "detour_cost_to_depot": 1,
                "hold_temp": 4.0,
                "dock_steps_remaining": 0,
            }
        ],
        "shipments": [
            {
                "id": 0,
                "cargo_temp": 4.5,
                "temp_lower_bound": 2.0,
                "temp_upper_bound": 8.0,
                "time_to_deadline": 9,
                "current_vehicle": 0,
                "destination_node": 3,
                "priority": 1,
                "cargo_type": 0,
                "excursion_count": 0,
                "excursion_duration": 0,
                "is_destroyed": 0,
                "is_delivered": 0,
                "steps_since_last_reading": 0,
            }
        ],
        "action_mask": [1, 0, 0, 0, 0, 0],
        "action_was_masked": False,
        "illegal_action_count": 2,
        "last_action_error": None,
        "reward_breakdown": {"r_temp": 0.2, "r_progress": 0.4, "r_cost": -0.1, "r_idle": 0.0},
        "episode_summary": {"steps_elapsed": 3, "shipments_delivered": 0, "shipments_destroyed": 0},
        "grader_scores": {"delivery": 0.0, "thermal": 1.0, "efficiency": 0.8, "composite": 0.44},
        "message": "step complete",
    }


def test_openenv_client_payload_roundtrip():
    env = ColdChainEnv.__new__(ColdChainEnv)

    action = ColdChainAction(vehicle_index=2, action_type=4, target_index=7)
    assert env._step_payload(action) == {
        "vehicle_index": 2,
        "action_type": 4,
        "target_index": 7,
    }

    result = env._parse_result({"observation": _sample_observation_payload(), "reward": 1.5, "done": True})
    assert result.done is True
    assert result.reward == 1.5
    assert result.observation.done is True
    assert result.observation.reward == 1.5
    assert result.observation.last_action_error is None
    assert result.observation.global_state.steps_elapsed == 3
    assert result.observation.global_state.difficulty_level_norm == 0.8
    assert result.observation.vehicles[0].shipments_onboard == [0, -1]
    assert result.observation.shipments[0].current_vehicle == 0

    state = env._parse_state(
        {
            "state": {
                "episode_id": "episode-1",
                "step_count": 3,
                "steps_remaining": 7,
                "illegal_action_count": 2,
                "terminated": True,
                "truncated": False,
            }
        }
    )
    assert state.step_count == 3
    assert state.steps_remaining == 7
    assert state.illegal_action_count == 2
    assert state.terminated is True
    assert state.truncated is False
