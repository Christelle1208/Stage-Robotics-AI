"""Tests: environment step, reward, and randomisation."""

from __future__ import annotations

import numpy as np
import pytest

import so100_mujoco_rl.envs  # noqa: F401


@pytest.fixture
def reach_env():
    import gymnasium as gym
    env = gym.make("SO100Reach-v0", render_mode=None)
    env.reset(seed=7)
    yield env
    env.close()


@pytest.fixture
def pp_env():
    import gymnasium as gym
    env = gym.make("SO100PickPlace-v0", render_mode=None)
    env.reset(seed=7)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Basic step checks
# ---------------------------------------------------------------------------

def test_reach_zero_action_step(reach_env):
    obs, reward, terminated, truncated, info = reach_env.step(
        np.zeros(reach_env.action_space.shape, dtype=np.float32)
    )
    obs = np.asarray(obs)
    assert obs.shape == reach_env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reach_random_steps_do_not_crash(reach_env):
    for _ in range(20):
        action = reach_env.action_space.sample()
        obs, reward, terminated, truncated, info = reach_env.step(action)
        if terminated or truncated:
            reach_env.reset()


def test_pick_place_zero_action_step(pp_env):
    obs, reward, terminated, truncated, info = pp_env.step(
        np.zeros(pp_env.action_space.shape, dtype=np.float32)
    )
    assert np.asarray(obs).shape == pp_env.observation_space.shape
    assert isinstance(reward, float)


def test_pick_place_random_steps_do_not_crash(pp_env):
    for _ in range(20):
        action = pp_env.action_space.sample()
        obs, reward, terminated, truncated, info = pp_env.step(action)
        if terminated or truncated:
            pp_env.reset()


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------

def test_reach_reward_has_correct_keys(reach_env):
    _, _, _, _, info = reach_env.step(reach_env.action_space.sample())
    assert "dist_to_target" in info
    assert "reach_reward" in info
    assert "success_bonus" in info


def test_pick_place_reward_has_correct_keys(pp_env):
    _, _, _, _, info = pp_env.step(pp_env.action_space.sample())
    assert "dist_ee_to_obj" in info
    assert "dist_obj_to_target" in info
    assert "reach_reward" in info
    assert "lift_reward" in info
    assert "place_reward" in info
    assert "success_bonus" in info


def test_pick_place_success_bonus_on_trivial_success():
    """Success bonus fires when object is within threshold of target."""
    import gymnasium as gym
    from so100_mujoco_rl.utils.config import load_config
    import mujoco

    env = gym.make("SO100PickPlace-v0", render_mode=None)
    env.reset(seed=0)

    # Manually move object to target position.
    inner = env.unwrapped  # type: ignore[attr-defined]
    task = inner._task
    data = inner.data
    model = inner.model

    target_pos = task.get_target_pos(data)
    adr = task._cube_qpos_adr
    data.qpos[adr : adr + 3] = target_pos
    data.qpos[adr + 3] = 1.0
    data.qpos[adr + 4 : adr + 7] = 0.0
    mujoco.mj_forward(model, data)

    assert task.is_success(data), "Object placed at target should trigger success"

    reward, info = task.compute_reward(data, np.zeros(inner.action_space.shape))
    assert info["success_bonus"] > 0.0

    env.close()


# ---------------------------------------------------------------------------
# Randomisation
# ---------------------------------------------------------------------------

def test_pick_place_object_randomised_across_resets():
    import gymnasium as gym
    env = gym.make("SO100PickPlace-v0", render_mode=None)

    positions = []
    for seed in range(5):
        env.reset(seed=seed)
        pos = env.unwrapped._task.get_object_pos(env.unwrapped.data)  # type: ignore[attr-defined]
        positions.append(pos.copy())
    env.close()

    # At least two different positions expected.
    unique = [p for i, p in enumerate(positions) if all(not np.allclose(p, q) for q in positions[:i])]
    assert len(unique) >= 2, "Object should be randomised across resets"


def test_pick_place_target_randomised_across_resets():
    import gymnasium as gym
    env = gym.make("SO100PickPlace-v0", render_mode=None)

    positions = []
    for seed in range(5):
        env.reset(seed=seed)
        pos = env.unwrapped._task.get_target_pos(env.unwrapped.data)  # type: ignore[attr-defined]
        positions.append(pos.copy())
    env.close()

    unique = [p for i, p in enumerate(positions) if all(not np.allclose(p, q) for q in positions[:i])]
    assert len(unique) >= 2, "Target should be randomised across resets"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_episode_truncates_at_max_steps():
    import gymnasium as gym
    env = gym.make("SO100PickPlace-v0", render_mode=None)
    env.reset(seed=0)
    max_steps = env.spec.max_episode_steps if env.spec else 200  # type: ignore[union-attr]

    truncated = False
    for _ in range(max_steps + 5):
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        if terminated or truncated:
            break

    env.close()
    assert truncated or terminated, "Episode should terminate or truncate"
