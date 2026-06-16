"""Generic SB3 training entry-point.

Supports both PPO and SAC, selected via the ``algo`` key in the training
YAML config.  Algorithm-specific scripts in ``scripts/`` simply call
``train(config_path=...)`` with their respective config.

Usage
-----
Python API::

    from so100_mujoco_rl.train.train_sb3 import train
    train("configs/train/ppo_so100_pick_place.yaml")

CLI (via scripts)::

    python scripts/train_ppo_so100_pick_place.py
    python scripts/train_sac_so100_pick_place.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

import so100_mujoco_rl.envs  # noqa: F401 — registers envs with Gymnasium
from so100_mujoco_rl.utils.config import load_config, project_root
from so100_mujoco_rl.wrappers.sb3 import make_vec_env

_ALGO_REGISTRY: dict[str, type] = {
    "ppo": PPO,
    "sac": SAC,
}


def _build_ppo_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "ent_coef",
        "vf_coef", "max_grad_norm",
    ]
    return {k: cfg[k] for k in keys if k in cfg}


def _build_sac_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "learning_rate", "buffer_size", "learning_starts", "batch_size",
        "tau", "gamma", "train_freq", "gradient_steps", "ent_coef",
        "target_update_interval",
    ]
    return {k: cfg[k] for k in keys if k in cfg}


def train(config_path: str | Path, overrides: dict[str, Any] | None = None) -> None:
    """Train an RL policy from a YAML config file.

    Parameters
    ----------
    config_path:
        Path to a training YAML (absolute or relative to project root).
    """
    cfg = load_config(config_path)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    root = project_root()

    algo_name: str = cfg.get("algo", "ppo").lower()
    if algo_name not in _ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Supported: {list(_ALGO_REGISTRY.keys())}"
        )
    AlgoClass = _ALGO_REGISTRY[algo_name]

    env_id: str = cfg["env_id"]
    env_config = cfg.get("env_config")
    resume_from = cfg.get("resume_from")
    total_timesteps: int = int(cfg.get("total_timesteps", 500_000))
    seed: int = int(cfg.get("seed", 42))
    device: str = cfg.get("device", "auto")
    policy: str = cfg.get("policy", "MlpPolicy")

    log_dir = root / cfg.get("log_dir", f"outputs/logs/{algo_name}_{env_id.lower()}")
    model_dir = root / cfg.get("model_dir", "outputs/models")
    tensorboard_log = root / cfg.get("tensorboard_log", "outputs/logs/tensorboard")
    checkpoint_freq: int = int(cfg.get("checkpoint_freq", 50_000))
    verbose: int = int(cfg.get("verbose", 1))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    env_kwargs: dict[str, Any] = {"render_mode": None}
    if env_config is not None:
        env_kwargs["env_config"] = env_config

    # Build training and evaluation envs.
    # Training env uses render_mode=None to avoid opening a window.
    train_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    eval_env = Monitor(
        gym.make(env_id, **env_kwargs),
        filename=str(log_dir / "eval_monitor"),
    )

    # Build algorithm-specific kwargs.
    if algo_name == "ppo":
        algo_kwargs = _build_ppo_kwargs(cfg)
    else:
        algo_kwargs = _build_sac_kwargs(cfg)

    if resume_from:
        resume_path = root / resume_from
        model = AlgoClass.load(str(resume_path), env=train_env, device=device)
        model.verbose = verbose
        model.tensorboard_log = str(tensorboard_log)
    else:
        model = AlgoClass(
            policy=policy,
            env=train_env,
            seed=seed,
            device=device,
            verbose=verbose,
            tensorboard_log=str(tensorboard_log),
            **algo_kwargs,
        )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Algorithm : {algo_name.upper()}")
        print(f"  Env ID    : {env_id}")
        if resume_from:
            print(f"  Resume    : {root / resume_from}")
        print(f"  Timesteps : {total_timesteps:,}")
        print(f"  Log dir   : {log_dir}")
        print(f"  Model dir : {model_dir}")
        print(f"{'='*60}\n")

    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(log_dir / "checkpoints"),
            name_prefix=f"{algo_name}_{env_id}",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best_model"),
            log_path=str(log_dir),
            eval_freq=max(checkpoint_freq // 2, 1000),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not bool(resume_from),
    )

    # Save final model.
    final_path = model_dir / f"{algo_name}_{env_id}_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}.zip")

    train_env.close()
    eval_env.close()
