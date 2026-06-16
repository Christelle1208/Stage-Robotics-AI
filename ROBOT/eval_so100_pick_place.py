from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from so100_rl import SO100PickPlaceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SO100 policy.")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model produced by SB3.")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--task", type=str, default="pick_place", choices=["reach", "pick", "pick_place"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_episode_steps", type=int, default=900)
    parser.add_argument("--fixed_reset_pose", action="store_true", help="Disable random reset pose.")
    parser.add_argument("--assist_grasp", action="store_true")
    parser.add_argument("--no_assist_grasp", action="store_true")
    parser.add_argument("--vecnorm", type=str, default="", help="Optional path to vecnormalize.pkl")
    parser.add_argument("--save_gif", type=str, default="", help="Optional output gif path.")
    parser.add_argument("--render_human", action="store_true", help="Open a MuJoCo viewer window.")
    return parser.parse_args()


def load_sb3_symbols() -> tuple[Any, Any, Any, Any]:
    try:
        sb3 = importlib.import_module("stable_baselines3")
        vec_env_mod = importlib.import_module("stable_baselines3.common.vec_env")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is missing. Install dependencies with: pip install -r requirements_so100_rl.txt"
        ) from exc

    return sb3.PPO, sb3.SAC, vec_env_mod.DummyVecEnv, vec_env_mod.VecNormalize


def main() -> None:
    args = parse_args()
    PPO, SAC, DummyVecEnv, VecNormalize = load_sb3_symbols()

    assist_grasp = args.assist_grasp and not args.no_assist_grasp
    if not args.assist_grasp and not args.no_assist_grasp:
        assist_grasp = True
    random_reset_pose = not args.fixed_reset_pose

    if args.render_human and args.save_gif:
        raise SystemExit("Choose either --render_human or --save_gif, not both.")

    render_mode = "human" if args.render_human else ("rgb_array" if args.save_gif else None)

    env = SO100PickPlaceEnv(
        task=args.task,
        max_episode_steps=args.max_episode_steps,
        assist_grasp=assist_grasp,
        random_reset_pose=random_reset_pose,
        render_mode=render_mode,
    )
    vec_env = DummyVecEnv([lambda: env])

    if args.vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model_cls = PPO if args.algo == "ppo" else SAC
    model = model_cls.load(args.model)

    rewards = []
    successes = []
    frames = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_success = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)
            done = bool(done_arr[0])
            ep_reward += float(reward[0])
            if info and isinstance(info, list):
                ep_success = max(ep_success, float(info[0].get("success", 0.0)))

            if args.save_gif:
                frame = env.render()
                frames.append(frame)

        rewards.append(ep_reward)
        successes.append(ep_success)
        print(f"Episode {ep + 1:02d} | reward={ep_reward:.3f} | success={ep_success:.0f}")

    mean_reward = sum(rewards) / len(rewards)
    success_rate = sum(successes) / len(successes)

    print("\nEvaluation summary")
    print(f"Mean reward : {mean_reward:.3f}")
    print(f"Success rate: {success_rate:.3f}")

    if args.save_gif:
        gif_path = Path(args.save_gif)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"Saved gif: {gif_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
