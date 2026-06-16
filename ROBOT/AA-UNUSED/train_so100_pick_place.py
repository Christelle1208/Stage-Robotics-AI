from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from so100_rl import SO100PickPlaceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a policy for SO100 pick-and-place tasks.")
    parser.add_argument("--task", type=str, default="pick_place", choices=["reach", "pick", "pick_place"])
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--total_timesteps", type=int, default=50_000)
    parser.add_argument("--n_envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max_episode_steps", type=int, default=1500)
    parser.add_argument("--fixed_reset_pose", action="store_true", help="Disable random reset pose.")
    parser.add_argument("--assist_grasp", action="store_true", help="Enable grasp assistance.")
    parser.add_argument(
        "--no_assist_grasp",
        action="store_true",
        help="Disable grasp assistance (harder, more realistic).",
    )
    parser.add_argument("--normalize_obs", action="store_true", help="Use VecNormalize on observations.")
    parser.add_argument("--log_dir", type=str, default="outputs/so100_rl")
    return parser.parse_args()


def make_env(
    task: str,
    max_episode_steps: int,
    assist_grasp: bool,
    monitor_cls: Any,
    random_reset_pose: bool,
):
    def _factory():
        env = SO100PickPlaceEnv(
            task=task,
            max_episode_steps=max_episode_steps,
            assist_grasp=assist_grasp,
            random_reset_pose=random_reset_pose,
        )
        return monitor_cls(env)

    return _factory


def load_sb3_symbols() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        sb3 = importlib.import_module("stable_baselines3")
        callbacks = importlib.import_module("stable_baselines3.common.callbacks")
        env_util = importlib.import_module("stable_baselines3.common.env_util")
        monitor_mod = importlib.import_module("stable_baselines3.common.monitor")
        vec_env_mod = importlib.import_module("stable_baselines3.common.vec_env")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is missing. Install dependencies with: pip install -r requirements_so100_rl.txt"
        ) from exc

    return (
        sb3.PPO,
        sb3.SAC,
        callbacks.EvalCallback,
        env_util.make_vec_env,
        vec_env_mod.VecNormalize,
        monitor_mod.Monitor,
    )


def tensorboard_is_available() -> bool:
    return importlib.util.find_spec("tensorboard") is not None


def main() -> None:
    args = parse_args()
    PPO, SAC, EvalCallback, make_vec_env, VecNormalize, Monitor = load_sb3_symbols()

    assist_grasp = args.assist_grasp and not args.no_assist_grasp
    if not args.assist_grasp and not args.no_assist_grasp:
        assist_grasp = True
    random_reset_pose = not args.fixed_reset_pose

    run_name = f"{args.algo}_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log = str(run_dir / "tb") if tensorboard_is_available() else None
    if tensorboard_log is None:
        print("TensorBoard is not installed: continuing without tensorboard logs.")

    vec_env = make_vec_env(
        make_env(args.task, args.max_episode_steps, assist_grasp, Monitor, random_reset_pose),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    eval_env = make_vec_env(
        make_env(args.task, args.max_episode_steps, assist_grasp, Monitor, random_reset_pose),
        n_envs=1,
        seed=args.seed + 1,
    )

    if args.normalize_obs:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_log,
            learning_rate=1e-4,
            n_steps=2048 // max(args.n_envs, 1),
            batch_size=256,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            clip_range=0.2,
        )
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_log,
            learning_rate=1e-4,
            batch_size=256,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            learning_starts=5_000,
            buffer_size=200_000,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(5_000 // max(args.n_envs, 1), 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, progress_bar=True)

    model_path = run_dir / "final_model"
    model.save(str(model_path))

    if args.normalize_obs and isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vecnormalize.pkl"))

    print(f"Training finished. Run dir: {run_dir}")
    print(f"Final model: {model_path}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
