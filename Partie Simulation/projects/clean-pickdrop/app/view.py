"""Look at the SO-100 pick-and-drop env to sanity-check the scene and reward.

This intentionally does NOT contain a hand-written controller -- writing a
real IK-based reach/grasp/place sequence for a 5-DOF arm is its own project,
and a fake one that doesn't actually grasp the cube would be misleading. What
you actually want to verify before spending compute on training is:

  1. The scene looks right (table, cube spawning across the work area, bin
     placement, end-effector marker) -- check with `--mode video`.
  2. The reward decomposition behaves sensibly -- r_pos shrinks as the
     end-effector nears the goal, r_energy stays small, the phase flips
     reach -> place on grasp, and the success bonus fires on drop -- check
     with `--mode rewards`.

Modes
-----
  video    Roll out random-action episodes and save them as an mp4. No GUI
           needed.
               python -m clean_pickdrop.view --mode video --episodes 3

  rewards  Roll out random-action episodes and print the reward decomposition
           every few steps -- r_pos / r_energy / phase / bonuses.
               python -m clean_pickdrop.view --mode rewards

  interactive   Open MuJoCo's interactive viewer stepping the env live with
                random actions. On macOS this MUST run with `mjpython` (the
                viewer needs the main thread):
               mjpython -m clean_pickdrop.view --mode interactive
"""

from __future__ import annotations

import argparse

import numpy as np

from clean_pickdrop.env import SO100PickDropEnv


def record_video(out_path: str, episodes: int, max_steps: int) -> None:
    import imageio

    env = SO100PickDropEnv(render_mode="rgb_array", max_episode_steps=max_steps)
    frames = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        for t in range(max_steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            ep_reward += reward
            frames.append(env.render())
            if terminated or truncated:
                break
        print(
            f"episode {ep}: {t + 1} steps, return={ep_reward:.2f}, "
            f"final phase={info['phase']}, success={info['is_success']}, "
            f"cube spawn -> final = {info['cube_pos']}"
        )

    env.close()
    imageio.mimsave(out_path, frames, fps=env.metadata["render_fps"])
    print(f"Saved {len(frames)} frames to {out_path}")


def print_rewards(episodes: int, max_steps: int, every: int) -> None:
    env = SO100PickDropEnv(max_episode_steps=max_steps)
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        print(f"\n--- episode {ep} (cube spawned at {info_pos(env)}) ---")
        print(f"{'step':>5} {'phase':<6} {'r_pos':>9} {'r_energy':>9} "
              f"{'grasp_bonus':>12} {'success_bonus':>14} {'ee->goal':>10}")
        for t in range(max_steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if t % every == 0 or terminated or truncated:
                terms = info["reward_terms"]
                dist = float(np.linalg.norm(info["cube_pos"] - info["ee_pos"]))
                print(
                    f"{t:>5} {info['phase']:<6} {terms['r_pos']:>9.4f} "
                    f"{terms['r_energy']:>9.5f} {terms['grasp_bonus']:>12.1f} "
                    f"{terms['success_bonus']:>14.1f} {dist:>10.4f}"
                )
            if terminated or truncated:
                print(f"  -> ended: success={info['is_success']}")
                break
    env.close()


def info_pos(env: SO100PickDropEnv) -> np.ndarray:
    return np.round(env._cube_pos(), 3)


def run_interactive(max_steps: int) -> None:
    import mujoco
    import mujoco.viewer

    env = SO100PickDropEnv(max_episode_steps=max_steps)
    obs, _ = env.reset(seed=0)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            viewer.sync()
            if terminated or truncated:
                print(f"episode ended: phase={info['phase']}, success={info['is_success']}")
                obs, _ = env.reset()
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["video", "rewards", "interactive"], default="video")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--every", type=int, default=20, help="rewards mode: print every N steps")
    parser.add_argument("--out", default="clean_pickdrop/rollout.mp4", help="video mode only")
    args = parser.parse_args()

    if args.mode == "video":
        record_video(args.out, args.episodes, args.max_steps)
    elif args.mode == "rewards":
        print_rewards(args.episodes, args.max_steps, args.every)
    else:
        run_interactive(args.max_steps)


if __name__ == "__main__":
    main()
