"""
Collect demonstration episodes from the SO100 MuJoCo simulation using a
trained SB3 policy (SAC or PPO from train_so100_pick_place.py).

Each episode is saved as a compressed .npz file containing:
  - images               : (T, 256, 256, 3) uint8  — rendered camera frame
  - states               : (T, 6) float32           — joint positions (qpos)
  - actions              : (T, 7) float32           — joint deltas applied [-1, 1]
  - rewards              : (T,) float32
  - language_instruction : str

Usage
-----
conda activate octo

python collect_demos_so100.py \\
    --model outputs/so100_rl/sac_reach_.../final_model.zip \\
    --vecnorm outputs/so100_rl/sac_reach_.../vecnormalize.pkl \\
    --algo sac --task reach \\
    --num_episodes 200 \\
    --output_dir outputs/demos/reach
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from so100_rl import SO100PickPlaceEnv
from so100_rl.octo_wrapper import TASK_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize h×w×3 uint8 image to size×size."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    try:
        import cv2  # type: ignore
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    except ImportError:
        sy = np.linspace(0, img.shape[0] - 1, size, dtype=int)
        sx = np.linspace(0, img.shape[1] - 1, size, dtype=int)
        return img[np.ix_(sy, sx)]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect SO100 simulation demos using a trained SB3 policy."
    )
    p.add_argument("--model", required=True,
                   help="Path to .zip model from train_so100_pick_place.py.")
    p.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    p.add_argument("--task", default="reach", choices=["reach", "pick", "pick_place"])
    p.add_argument("--vecnorm", default="",
                   help="Optional path to vecnormalize.pkl produced during training.")
    p.add_argument("--num_episodes", type=int, default=200,
                   help="Number of episodes to collect (default: 200).")
    p.add_argument("--max_episode_steps", type=int, default=300,
                   help="Max steps per episode (default: 300).")
    p.add_argument("--image_size", type=int, default=256,
                   help="Square image size to save (default: 256).")
    p.add_argument("--output_dir", default="",
                   help="Directory where .npz files will be saved. "
                        "Defaults to outputs/demos/<task>/.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_episode_len", type=int, default=10,
                   help="Discard episodes shorter than this (default: 10).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Load SB3 -------------------------------------------------------
    try:
        sb3 = importlib.import_module("stable_baselines3")
        vec_env_mod = importlib.import_module("stable_baselines3.common.vec_env")
    except ModuleNotFoundError as e:
        raise SystemExit("stable-baselines3 not found. Install it first.") from e

    PPO, SAC = sb3.PPO, sb3.SAC
    DummyVecEnv, VecNormalize = vec_env_mod.DummyVecEnv, vec_env_mod.VecNormalize

    # ---- Output directory -----------------------------------------------
    output_dir = Path(args.output_dir or f"outputs/demos/{args.task}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Environment ----------------------------------------------------
    env = SO100PickPlaceEnv(
        task=args.task,
        max_episode_steps=args.max_episode_steps,
        render_mode="rgb_array",   # needed to capture frames
        assist_grasp=True,
    )
    vec_env = DummyVecEnv([lambda: env])

    if args.vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # ---- Load policy ----------------------------------------------------
    model_cls = PPO if args.algo == "ppo" else SAC
    model = model_cls.load(args.model)
    print(f"Loaded {args.algo.upper()} policy from {args.model}")

    language = TASK_INSTRUCTIONS[args.task]
    saved = 0
    discarded = 0
    attempts = 0

    # ---- Collection loop ------------------------------------------------
    # Keep running until we have `num_episodes` SUCCESSFUL episodes.
    while saved < args.num_episodes:
        attempts += 1
        flat_obs = vec_env.reset()
        done = False
        images, states, actions, rewards = [], [], [], []
        ep_success = False

        while not done:
            # Render the current frame
            frame = env.render()
            frame = _resize_image(frame, args.image_size)
            images.append(frame)

            # Record proprio from flat observation (first 6 dims = qpos)
            states.append(flat_obs[0, :6].copy().astype(np.float32))

            # Query policy
            action, _ = model.predict(flat_obs, deterministic=True)
            actions.append(action[0].copy().astype(np.float32))

            flat_obs, reward, done_arr, info_arr = vec_env.step(action)
            rewards.append(float(reward[0]))
            done = bool(done_arr[0])

            # Track success from env info
            if info_arr and isinstance(info_arr, list):
                ep_success = ep_success or bool(info_arr[0].get("success", False))

        ep_len = len(actions)
        ep_return = sum(rewards)

        if not ep_success:
            discarded += 1
            print(f"  [skip] attempt {attempts:04d} | len={ep_len:4d} | r={ep_return:7.1f} | FAILED")
            continue

        if ep_len < args.min_episode_len:
            discarded += 1
            print(f"  [skip] attempt {attempts:04d} | len={ep_len} (too short) | r={ep_return:.1f}")
            continue

        npz_path = output_dir / f"episode_{saved:05d}.npz"
        np.savez_compressed(
            npz_path,
            images=np.array(images, dtype=np.uint8),
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            language_instruction=np.array(language),
        )
        saved += 1
        success_rate = saved / attempts * 100
        print(f"  [{saved:04d}/{args.num_episodes}] attempt {attempts:04d} | "
              f"len={ep_len:4d} | r={ep_return:7.1f} | "
              f"success_rate={success_rate:.0f}% | saved → {npz_path.name}")

    vec_env.close()

    print(f"\n{'─'*50}")
    print(f"Saved    : {saved} successful episodes")
    print(f"Attempts : {attempts} total ({discarded} discarded — failed or too short)")
    print(f"Success rate: {saved / attempts * 100:.1f}%")
    print(f"Output   : {output_dir.resolve()}")
    print(f"\nNext step — fine-tune Octo:")
    print(f"  conda activate octo")
    print(f"  python finetune_octo_so100.py \\")
    print(f"      --demos_dir {output_dir} \\")
    print(f"      --task {args.task} \\")
    print(f"      --save_dir outputs/octo_finetuned/{args.task}/")


if __name__ == "__main__":
    main()
