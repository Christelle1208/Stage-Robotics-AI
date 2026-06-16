from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from so100_rl import SO100PickPlaceEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick random-policy smoke test for SO100 env.")
    parser.add_argument("--task", type=str, default="pick_place", choices=["reach", "pick", "pick_place"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fixed_reset_pose", action="store_true", help="Disable random reset pose.")
    parser.add_argument("--assist_grasp", action="store_true")
    parser.add_argument("--no_assist_grasp", action="store_true")
    parser.add_argument("--render_human", action="store_true", help="Open a MuJoCo viewer window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assist_grasp = args.assist_grasp and not args.no_assist_grasp
    if not args.assist_grasp and not args.no_assist_grasp:
        assist_grasp = True
    random_reset_pose = not args.fixed_reset_pose

    render_mode = "human" if args.render_human else None
    env = SO100PickPlaceEnv(
        task=args.task,
        max_episode_steps=args.steps,
        assist_grasp=assist_grasp,
        random_reset_pose=random_reset_pose,
        render_mode=render_mode,
    )

    for ep in range(args.episodes):
        _, info = env.reset(seed=ep)
        total_reward = 0.0
        success = 0.0
        for _ in range(args.steps):
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            success = max(success, float(info.get("success", 0.0)))
            if terminated or truncated:
                break

        print(f"Episode {ep + 1} | reward={total_reward:.3f} | success={success:.0f} | task={info.get('task', args.task)}")

    env.close()


if __name__ == "__main__":
    main()
