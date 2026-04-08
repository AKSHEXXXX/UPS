from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .config import ColdChainConfig


class WeatherEvent(IntEnum):
    NONE = 0
    HEATWAVE = 1
    STORM = 2
    POWER_OUTAGE = 3


@dataclass
class WeatherSystem:
    current_event: WeatherEvent = WeatherEvent.NONE
    event_duration_remaining: int = 0
    base_outdoor_temp: float = 22.0

    def step(self, rng: np.random.Generator, config: ColdChainConfig) -> None:
        if self.event_duration_remaining > 0:
            self.event_duration_remaining -= 1
            if self.event_duration_remaining == 0:
                self.current_event = WeatherEvent.NONE
            return

        if config.weather_events_enabled and rng.random() < 0.005:
            self.current_event = WeatherEvent(int(rng.integers(1, 4)))
            self.event_duration_remaining = int(rng.integers(8, 25))

    @property
    def outdoor_temperature(self) -> float:
        if self.current_event == WeatherEvent.HEATWAVE:
            return self.base_outdoor_temp + 12.0
        if self.current_event == WeatherEvent.STORM:
            return self.base_outdoor_temp - 5.0
        return self.base_outdoor_temp

    @property
    def traffic_multiplier(self) -> float:
        if self.current_event == WeatherEvent.STORM:
            return 2.5
        return 1.0
