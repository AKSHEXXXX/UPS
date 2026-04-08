from dataclasses import dataclass, field


@dataclass
class ColdChainConfig:
    # Fleet
    n_vehicles: int = 3
    max_cargo_per_vehicle: int = 2

    # Shipments
    n_shipments: int = 5
    max_shipments: int = 20

    # City graph
    n_nodes: int = 20
    n_cold_depots: int = 3
    graph_seed: int = 42

    # Episode
    max_steps: int = 500
    step_duration_hours: float = 0.25

    # Physics
    thermal_k: dict = field(
        default_factory=lambda: {
            "vaccine": 0.08,
            "insulin": 0.10,
            "blood": 0.12,
            "organ": 0.15,
        }
    )

    # Events
    weather_events_enabled: bool = True
    breakdown_probability: float = 0.002
    refrigeration_degradation_prob: float = 0.005

    # Reward tuning
    temp_shaping_in_range: float = 0.0
    temp_shaping_per_degree: float = 0.10
    temp_integrity_hard_margin: float = 1.0
    temp_integrity_borderline_margin: float = 0.5
    progress_discount: float = 0.99
    revisit_penalty_scale: float = 0.01
    step_time_penalty: float = 0.03

    # Debug controls
    debug_step_trace: bool = False

    # Training Progress (For Annealing/Curriculum)
    current_training_step: int = 0
    penalty_anneal_steps: int = 30000
    penalty_initial_scale: float = 0.1


    def __post_init__(self):
        assert self.n_cold_depots >= 1, "Must have at least one cold depot"
        assert self.n_nodes > self.n_cold_depots + 1, "City too small for depots + hub + destinations"
        assert self.n_vehicles >= 1
        assert self.max_steps > 0
        assert 0.0 <= self.breakdown_probability <= 1.0
        assert 0.0 <= self.refrigeration_degradation_prob <= 1.0