"""
Script d'entraînement avec les VRAIES limites physiques du SO100.
"""
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

from so100_physical_limits_env import SO100PhysicalLimitsEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SO100 with PHYSICAL limits")
    parser.add_argument("--task", type=str, default="pick_place", choices=["reach", "pick", "pick_place"])
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_steps", type=int, default=800)
    parser.add_argument("--fixed_reset_pose", action="store_true")
    parser.add_argument("--log_dir", type=str, default="outputs/so100_physical")
    return parser.parse_args()


def make_env(task: str, max_episode_steps: int, monitor_cls: Any, random_reset_pose: bool):
    def _factory():
        env = SO100PhysicalLimitsEnv(
            task=task,
            max_episode_steps=max_episode_steps,
            assist_grasp=True,
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
            "stable-baselines3 missing. Install with: pip install stable-baselines3[extra]"
        ) from exc

    return (
        sb3.PPO,
        sb3.SAC,
        callbacks.EvalCallback,
        env_util.make_vec_env,
        vec_env_mod.VecNormalize,
        monitor_mod.Monitor,
    )


def main() -> None:
    args = parse_args()
    PPO, SAC, EvalCallback, make_vec_env, VecNormalize, Monitor = load_sb3_symbols()

    random_reset_pose = not args.fixed_reset_pose

    run_name = f"{args.algo}_{args.task}_physical_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SO100 TRAINING - PHYSICAL LIMITS")
    print("="*70)
    print("\nLIMITES PHYSIQUES RÉELLES:")
    print("  Motor 0 (Rotation):  -1.92 à +1.92")
    print("  Motor 1 (Pitch):     -3.32 à +0.174  ← CRITIQUE: max=+0.174!")
    print("  Motor 2 (Elbow):     -0.174 à +3.14")
    print("  Motor 3 (Wrist_P):   -1.66 à +1.66")
    print("  Motor 4 (Wrist_R):   -2.79 à +2.79")
    print("  Motor 5 (Jaw):       -0.174 à +1.75")
    print("\nLIMITES SÛRES UTILISÉES:")
    print("  Pitch: -1.2 à +0.1 rad")
    print("  Elbow: 0.3 à 2.8 rad")
    print("\nPOSE SÛRE PAR DÉFAUT:")
    print("  [0.0, -0.8, 1.0, 0.5, -0.5, 0.0]")
    print("  (Pitch=-0.8 = penché vers l'avant)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Task: {args.task}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Random reset: {random_reset_pose}")
    print(f"  Output: {run_dir}")
    print("="*70 + "\n")

    vec_env = make_vec_env(
        make_env(args.task, args.max_episode_steps, Monitor, random_reset_pose),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    eval_env = make_vec_env(
        make_env(args.task, args.max_episode_steps, Monitor, random_reset_pose),
        n_envs=1,
        seed=args.seed + 1,
    )

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            n_steps=2048 // max(args.n_envs, 1),
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.02,
            clip_range=0.2,
            n_epochs=10,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            ),
        )
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            learning_starts=10_000,
            buffer_size=300_000,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
            ),
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(5_000 // max(args.n_envs, 1), 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    print("Début de l'entraînement...\n")
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, progress_bar=True)

    model_path = run_dir / "final_model"
    model.save(str(model_path))
    vec_env.save(str(run_dir / "vecnormalize.pkl"))

    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ!")
    print("="*70)
    print(f"Dossier: {run_dir}")
    print(f"Modèle final: {model_path}.zip")
    print(f"Meilleur modèle: {run_dir / 'best_model' / 'best_model.zip'}")
    print(f"VecNormalize: {run_dir / 'vecnormalize.pkl'}")
    print("\nÉvaluation:")
    print(f"python eval_so100_pick_place.py \\")
    print(f"    --model {run_dir / 'best_model' / 'best_model.zip'} \\")
    print(f"    --vecnorm {run_dir / 'vecnormalize.pkl'} \\")
    print(f"    --episodes 5 --render_human")
    print("="*70 + "\n")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
