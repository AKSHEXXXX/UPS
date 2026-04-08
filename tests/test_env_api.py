import warnings

from gymnasium.utils.env_checker import check_env

from server.env import ColdChainEnv


def test_gymnasium_compliance():
    env = ColdChainEnv()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        check_env(env, skip_render_check=True)


def test_obs_shape_consistent():
    env = ColdChainEnv()
    obs, _ = env.reset(seed=0)
    assert obs["vehicles"].shape == (env.config.n_vehicles, 9 + env.config.max_cargo_per_vehicle)
    assert obs["shipments"].shape == (env.config.max_shipments, 13)
