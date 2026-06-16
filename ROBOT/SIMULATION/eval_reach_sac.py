"""
eval_reach_sac.py
------------------
Evaluate a trained SAC model on the SO-ARM100 reach-cube task.

Features:
  - Human viewer (interactive MuJoCo window)  OR  save to .gif
  - Optionally enable YOLO detector and overlay detection bbox on the side panel
  - Displays distance, reward, and success info per step

Usage (from SIMULATION/):
    # Interactive viewer
    mjpython eval_reach_sac.py --model outputs/reach_sac/best_model.zip \\
                               --vecnorm outputs/reach_sac/vecnorm.pkl

    # Save to gif
    python eval_reach_sac.py --model outputs/reach_sac/best_model.zip \\
                             --vecnorm outputs/reach_sac/vecnorm.pkl \\
                             --save_gif outputs/reach_sac/eval.gif

    # With YOLO (needs trained cube_yolov10.pt)
    python eval_reach_sac.py --model outputs/reach_sac/best_model.zip \\
                             --use_yolo
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

_THIS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SAC model.")
    p.add_argument("--model",       required=True, help="Path to .zip model.")
    p.add_argument("--vecnorm",     default="",    help="Path to vecnorm.pkl.")
    p.add_argument("--task",        type=str,   default="reach",
                   choices=["reach", "pick_place"],
                   help="Task the model was trained on (default: reach).")
    p.add_argument("--episodes",    type=int,   default=15)
    p.add_argument("--max_steps",   type=int,   default=500)
    p.add_argument("--use_yolo",    action="store_true",
                   help="Use YOLO cube detector instead of GT position.")
    p.add_argument("--save_gif",    default="",
                   help="Save rollout to this .gif path (disables human viewer).")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--seed",        type=int,   default=3)
    p.add_argument("--close_only",  action="store_true",
                   help="Restrict cube placement to x < 0.4 (near side of table, close to robot).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as e:
        sys.exit(f"stable-baselines3 not found: {e}")

    from reach_cube_env import ReachCubeEnv

    render_mode = "rgb_array" if args.save_gif else "human"

    # ------------------------------------------------------------------
    # Build env
    # ------------------------------------------------------------------
    # Cube placement range: near side of table when --close_only (x in [0.26, 0.39])
    cube_x_range = (0.26, 0.39) if args.close_only else None

    env = ReachCubeEnv(
        use_yolo          = args.use_yolo,
        task              = args.task,
        render_mode       = render_mode,
        max_episode_steps = args.max_steps,
        frame_skip        = 8,
        random_cube       = True,
        seed              = args.seed,
        cube_x_range      = cube_x_range,
    )

    vec_env = DummyVecEnv([lambda: env])
    if args.vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training   = False
        vec_env.norm_reward = False

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = SAC.load(args.model)

    # ------------------------------------------------------------------
    # Run episodes
    # ------------------------------------------------------------------
    all_rewards: list[float]  = []
    all_dists:   list[float]  = []
    all_success: list[bool]   = []
    frames: list[np.ndarray]  = []

    for ep in range(args.episodes):
        obs    = vec_env.reset()
        done   = False
        ep_reward  = 0.0
        ep_success = False
        step_count = 0
        min_dist   = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done_arr, info = vec_env.step(action)
            done = bool(done_arr[0])

            ep_reward  += float(reward[0])
            step_count += 1
            if info and isinstance(info, list):
                dist = float(info[0].get("distance", 0.0))
                min_dist   = min(min_dist, dist)
                ep_success = ep_success or bool(info[0].get("success", False))

            if args.save_gif:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            
            if render_mode == "human":
                time.sleep(0.004)

        all_rewards.append(ep_reward)
        all_dists.append(min_dist)
        all_success.append(ep_success)

        print(
            f"Episode {ep+1:2d}/{args.episodes}  |  "
            f"reward={ep_reward:8.2f}  |  "
            f"min_dist={min_dist:.4f} m  |  "
            f"success={ep_success}  |  "
            f"steps={step_count}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("SUMMARY")
    print(f"  Mean reward  : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean min dist: {np.mean(all_dists):.4f} m")
    print(f"  Success rate : {np.mean(all_success) * 100:.0f}%  "
          f"({sum(all_success)}/{args.episodes})")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Save gif
    # ------------------------------------------------------------------
    if args.save_gif and frames:
        try:
            import imageio.v2 as imageio
            gif_path = pathlib.Path(args.save_gif)
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(gif_path), frames, fps=25)
            print(f"\nGIF saved → {gif_path}")
        except ImportError:
            print("[warn] imageio not installed — GIF not saved.  pip install imageio")

    vec_env.close()


if __name__ == "__main__":
    main()
