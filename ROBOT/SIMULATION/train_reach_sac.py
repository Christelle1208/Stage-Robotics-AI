"""
train_reach_sac.py
-------------------
Train a SAC agent (stable-baselines3) on the SO-ARM100 with TEST_SCENE.xml.

Supported tasks:
  reach      -- end-effector reaches the green cube   (21-dim obs)
  pick_place -- reach → grasp → lift → transport → place on a goal (29-dim obs)

Usage (from SIMULATION/):
    # Phase 1 — reach only (fast, ~500k steps)
    python train_reach_sac.py --task reach

    # Phase 2 — pick-and-place (train from scratch, ~2M steps recommended)
    python train_reach_sac.py --task pick_place --timesteps 2_000_000

Outputs:
    outputs/reach_sac/        or    outputs/pick_place_sac/
    ├── best_model.zip
    ├── final_model.zip
    ├── vecnorm.pkl
    └── logs/          (TensorBoard)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC for SO-ARM100.")
    p.add_argument("--task",         type=str,   default="reach",
                   choices=["reach", "pick_place"],
                   help="Task to train (default: reach).")
    p.add_argument("--timesteps",    type=int,   default=500_000,
                   help="Total env steps (default: 500_000).")
    p.add_argument("--max_ep_steps", type=int,   default=500,
                   help="Max steps per episode (default: 500).")    
    p.add_argument("--frame_skip",   type=int,   default=8)
    p.add_argument("--use_yolo",     action="store_true",
                   help="Use YOLO for cube observation (need trained model).")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--n_envs",       type=int,   default=1,
                   help="Parallel envs via SubprocVecEnv (default: 1).")
    p.add_argument("--output_dir",   type=str,   default="",
                   help="Output directory (default: outputs/<task>_sac).")
    p.add_argument("--eval_freq",    type=int,   default=20_000,
                   help="Evaluate & checkpoint every N steps.")
    p.add_argument("--eval_episodes",type=int,   default=10)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--buffer_size",  type=int,   default=500_000)
    p.add_argument("--load_model",   type=str,   default="",
                   help="Path to a previous .zip to resume training from.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import (
            EvalCallback,
            CheckpointCallback,
        )
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        sys.exit(
            f"stable-baselines3 not found: {e}\n"
            "Install with:  pip install stable-baselines3[extra]"
        )

    from reach_cube_env import ReachCubeEnv

    output_dir = pathlib.Path(
        args.output_dir if args.output_dir else f"outputs/{args.task}_sac"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir    = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------
    def _make_env(rank: int = 0):
        def _init():
            env = ReachCubeEnv(
                use_yolo          = args.use_yolo,
                task              = args.task,
                render_mode       = None,
                max_episode_steps = args.max_ep_steps,
                frame_skip        = args.frame_skip,
                random_cube       = True,
                seed              = args.seed + rank,
            )
            env = Monitor(env)
            return env
        return _init

    vec_cls = SubprocVecEnv if args.n_envs > 1 else None

    if args.n_envs > 1:
        train_env = SubprocVecEnv([_make_env(i) for i in range(args.n_envs)])
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv([_make_env(0)])

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )

    # Separate eval env (no VecNormalize reward normalisation during eval)
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env_raw = DummyVecEnv([_make_env(args.seed + 99)])
    eval_env     = VecNormalize(
        eval_env_raw,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = str(output_dir),
        log_path             = str(log_dir),
        eval_freq            = max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes      = args.eval_episodes,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )
    ckpt_callback = CheckpointCallback(
        save_freq   = max(args.eval_freq // args.n_envs, 1),
        save_path   = str(output_dir / "checkpoints"),
        name_prefix = "sac_reach",
        verbose     = 1,
    )

    # ------------------------------------------------------------------
    # SAC model
    # ------------------------------------------------------------------
    policy_kwargs = dict(net_arch=[256, 256])

    if args.load_model:
        print(f"Resuming from {args.load_model}")
        model = SAC.load(
            args.load_model,
            env            = train_env,
            device         = "auto",
            learning_rate  = args.lr,
            batch_size     = args.batch_size,
        )
    else:
        model = SAC(
            policy          = "MlpPolicy",
            env             = train_env,
            learning_rate   = args.lr,
            buffer_size     = args.buffer_size,
            batch_size      = args.batch_size,
            learning_starts = 5_000,
            train_freq      = 1,
            gradient_steps  = 1,
            ent_coef        = "auto",
            target_update_interval = 1,
            gamma           = 0.99,
            tau             = 0.005,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = str(log_dir),
            verbose         = 1,
            seed            = args.seed,
            device          = "auto",
        )

    print("\n" + "=" * 60)
    print(f"Training SAC — SO-ARM100  task={args.task}")
    print(f"  Timesteps    : {args.timesteps:,}")
    print(f"  Max ep steps : {args.max_ep_steps}")
    print(f"  Use YOLO     : {args.use_yolo}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 60 + "\n")

    model.learn(
        total_timesteps    = args.timesteps,
        callback           = [eval_callback, ckpt_callback],
        reset_num_timesteps= not bool(args.load_model),
    )

    # ------------------------------------------------------------------
    # Save final model & normalisation stats
    # ------------------------------------------------------------------
    model.save(str(output_dir / "final_model"))
    train_env.save(str(output_dir / "vecnorm.pkl"))
    print(f"\nTraining complete. Models saved to {output_dir}/")
    print(f"  best_model.zip   ← best eval score")
    print(f"  final_model.zip  ← last checkpoint")
    print(f"  vecnorm.pkl      ← observation normalisation stats")
    print("\nEvaluate with:")
    print(f"  mjpython eval_reach_sac.py --task {args.task} "
          f"--model {output_dir}/best_model.zip "
          f"--vecnorm {output_dir}/vecnorm.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
