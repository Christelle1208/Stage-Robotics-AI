"""Stable-Baselines3 integration helpers.

SB3 requires:
* Environments wrapped in a VecEnv.
* Observations as flat numpy arrays (or dicts for dict-obs policies).
* float32 dtype throughout.

This module provides a factory that:
1. Imports the so100_mujoco_rl env registry.
2. Builds a ``DummyVecEnv`` (single process) or ``SubprocVecEnv`` (multi).
3. Applies ``VecNormalize`` if requested.
4. Returns a ready-to-use SB3 VecEnv.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Importing the envs package registers the Gymnasium environments.
import so100_mujoco_rl.envs  # noqa: F401


def make_vec_env(
    env_id: str,
    n_envs: int = 1,
    seed: int = 0,
    env_kwargs: dict | None = None,
    use_subproc: bool = False,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
) -> DummyVecEnv | SubprocVecEnv | VecNormalize:
    """Create a vectorised SB3 environment.

    Parameters
    ----------
    env_id:
        Registered Gymnasium env id (e.g. ``"SO100PickPlace-v0"``).
    n_envs:
        Number of parallel environments.
    seed:
        Base random seed; each env gets seed + rank.
    env_kwargs:
        Extra kwargs forwarded to the env constructor.
    use_subproc:
        Use ``SubprocVecEnv`` instead of ``DummyVecEnv``.  Only useful for
        envs without a GUI renderer.  MuJoCo rendering is single-process, so
        default is ``DummyVecEnv``.
    normalize_obs:
        Wrap with ``VecNormalize`` (running mean/std on observations).
    normalize_reward:
        Also normalise rewards inside ``VecNormalize``.

    Returns
    -------
    VecEnv ready for SB3 training.
    """
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv

    vec_env = sb3_make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs or {},
        vec_env_cls=vec_cls,
    )

    if normalize_obs or normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=10.0,
        )

    return vec_env


def make_env_fn(env_id: str, rank: int, seed: int, env_kwargs: dict) -> Callable:
    """Return a thunk that creates one environment instance.

    Useful when building a custom VecEnv manually.
    """
    def _init() -> gym.Env:
        env = gym.make(env_id, **env_kwargs)
        env.reset(seed=seed + rank)
        return env
    return _init
