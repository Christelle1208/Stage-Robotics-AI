"""Tests: environment can be created and reset without errors."""

from __future__ import annotations

import numpy as np
import pytest

import so100_mujoco_rl.envs  # noqa: F401 — registers envs


@pytest.fixture
def reach_env():
    import gymnasium as gym
    env = gym.make("SO100Reach-v0", render_mode=None)
    yield env
    env.close()


@pytest.fixture
def pick_place_env():
    import gymnasium as gym
    env = gym.make("SO100PickPlace-v0", render_mode=None)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Reach env
# ---------------------------------------------------------------------------

def test_reach_env_creates(reach_env):
    assert reach_env is not None


def test_reach_env_spaces(reach_env):
    obs_space = reach_env.observation_space
    act_space = reach_env.action_space
    assert hasattr(obs_space, "shape")
    assert hasattr(act_space, "shape")
    # At minimum: 5 qpos + 5 qvel + 3 ee + 3 target = 16
    assert obs_space.shape[0] >= 16
    # At minimum 5 arm actuators
    assert act_space.shape[0] >= 5


def test_reach_env_reset_returns_obs(reach_env):
    obs, info = reach_env.reset(seed=42)
    obs = np.asarray(obs)
    assert obs.shape == reach_env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_reach_env_reset_seed_determinism(reach_env):
    obs1, _ = reach_env.reset(seed=0)
    obs2, _ = reach_env.reset(seed=0)
    np.testing.assert_array_equal(obs1, obs2)


def test_reach_env_reset_different_seeds_differ(reach_env):
    obs1, _ = reach_env.reset(seed=0)
    obs2, _ = reach_env.reset(seed=99)
    # Target position is randomised, so at least some obs values differ.
    assert not np.allclose(obs1, obs2)


# ---------------------------------------------------------------------------
# Pick-and-place env
# ---------------------------------------------------------------------------

def test_pick_place_env_creates(pick_place_env):
    assert pick_place_env is not None


def test_pick_place_env_spaces(pick_place_env):
    obs_space = pick_place_env.observation_space
    act_space = pick_place_env.action_space
    # Minimum: 5 arm qpos + 5 arm qvel + 1 grip qpos + 1 grip qvel
    #          + 3 ee + 3 obj + 3 tgt + 3 + 3 = 28
    assert obs_space.shape[0] >= 27
    # Action: 5 arm + 1 gripper = 6
    assert act_space.shape[0] >= 5


def test_pick_place_env_reset_returns_obs(pick_place_env):
    obs, info = pick_place_env.reset(seed=0)
    obs = np.asarray(obs)
    assert obs.shape == pick_place_env.observation_space.shape
    assert obs.dtype == np.float32


def test_pick_place_action_space_is_continuous(pick_place_env):
    """SAC requires a continuous Box action space."""
    from gymnasium.spaces import Box
    assert isinstance(pick_place_env.action_space, Box)
    assert pick_place_env.action_space.dtype == np.float32
