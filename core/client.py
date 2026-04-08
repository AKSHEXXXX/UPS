# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import ColdChainAction, ColdChainObservation, ColdChainState


class ColdChainEnv(EnvClient[ColdChainAction, ColdChainObservation, ColdChainState]):
    def _step_payload(self, action: ColdChainAction) -> Dict:
        return {
            "vehicle_index": action.vehicle_index,
            "action_type": action.action_type,
            "target_index": action.target_index,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ColdChainObservation]:
        obs_data = payload.get("observation", {})
        observation = ColdChainObservation(**obs_data)
        observation.done = payload.get("done", False)
        observation.reward = payload.get("reward")

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ColdChainState:
        data = payload.get("state", payload)
        return ColdChainState(**data)

