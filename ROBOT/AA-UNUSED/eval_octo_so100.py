"""
Evaluate a pre-trained Octo model inside the SO100 MuJoCo simulation.

Usage
-----
# Zero-shot with the public octo-small-1.5 checkpoint (default):
    python eval_octo_so100.py --task reach --episodes 5

# With human viewer (needs mjpython):
    mjpython eval_octo_so100.py --task reach --episodes 3 --render_human

# Save a gif:
    python eval_octo_so100.py --task reach --save_gif outputs/octo_reach.gif

# Use a local / fine-tuned checkpoint:
    python eval_octo_so100.py --task pick --model_path outputs/octo_finetuned/

Architecture note
-----------------
Octo (pretrained on bridge_dataset) outputs 7-D EEF-delta actions while SO100
uses 7-D joint-delta actions. The dimensions match, but the semantics differ.
For a zero-shot run Octo's outputs are clipped to [-1, 1] and passed directly
through — this is useful to verify the plumbing before fine-tuning.

To get useful robot behaviour, collect demonstrations and fine-tune the Octo
action head on SO100 data (see octo/examples/02_finetune_new_observation_action.py).
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from functools import partial
from pathlib import Path

import jax
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OCTO_DIR = ROOT_DIR / "octo"
if str(OCTO_DIR) not in sys.path:
    sys.path.insert(0, str(OCTO_DIR))

from so100_rl import SO100PickPlaceEnv
from so100_rl.octo_wrapper import OctoSO100Wrapper, TASK_INSTRUCTIONS

try:
    from octo.model.octo_model import OctoModel
    from octo.utils.train_callbacks import supply_rng
except ImportError as e:
    raise SystemExit(
        "Octo is not installed. From the workspace root run:\n"
        "  pip install -e octo/\n"
        "Then install JAX for your platform (CPU or GPU/Metal)."
    ) from e


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an Octo policy inside the SO100 MuJoCo sim."
    )
    p.add_argument(
        "--model_path",
        default="hf://rail-berkeley/octo-small-1.5",
        help="Path to an Octo checkpoint directory or a HuggingFace repo path "
             "(default: hf://rail-berkeley/octo-small-1.5).",
    )
    p.add_argument(
        "--task", default="reach", choices=["reach", "pick", "pick_place"],
        help="SO100 task to run (default: reach).",
    )
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of episodes to run (default: 3).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_episode_steps", type=int, default=300,
                   help="Maximum steps per episode (default: 300).")

    # Octo inference settings
    p.add_argument(
        "--history_horizon", type=int, default=5,
        help="Number of past obs frames fed to Octo as context (default: 5).",
    )
    p.add_argument(
        "--exec_horizon", type=int, default=4,
        help="How many actions from each predicted chunk to execute (default: 4).",
    )
    p.add_argument(
        "--stat_key", default="",
        help="Key into model.dataset_statistics for un-normalisation. "
             "Leave empty to auto-detect (bridge_dataset or first key).",
    )

    # Observation settings
    p.add_argument("--image_size", type=int, default=256,
                   help="Square image size for Octo (default: 256).")
    p.add_argument("--no_proprio", action="store_true",
                   help="Disable proprioception in the observation dict.")
    p.add_argument(
        "--language", default="",
        help="Override the language instruction (default: task-specific sentence).",
    )

    # Output / rendering
    p.add_argument("--render_human", action="store_true",
                   help="Show a live MuJoCo viewer window (use mjpython).")
    p.add_argument("--save_gif", default="",
                   help="Path to save a gif of the evaluation (optional).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _build_obs_batch(history: deque, num_obs: int, horizon: int) -> dict:
    """
    Stack a sliding window of obs dicts into a single batch dict.
    Adds a 'timestep_pad_mask' to indicate which timesteps are padding.
    Shape after stacking: each value has shape (horizon, ...).
    After adding batch dim: (1, horizon, ...).
    """
    horizon = len(history)
    stacked = {k: np.stack([h[k] for h in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    pad_mask = np.ones(horizon, dtype=np.float32)
    pad_mask[:pad_length] = 0.0
    stacked["timestep_pad_mask"] = pad_mask
    # Add batch dimension (Octo expects [batch, horizon, ...])
    return jax.tree_map(lambda x: x[None], stacked)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.render_human and args.save_gif:
        raise SystemExit("Choose --render_human OR --save_gif, not both.")

    # -----------------------------------------------------------------------
    # 1.  Build the wrapped environment
    # -----------------------------------------------------------------------
    base_env = SO100PickPlaceEnv(
        task=args.task,
        max_episode_steps=args.max_episode_steps,
        render_mode="rgb_array",  # mandatory: Octo needs rendered images
    )
    language = args.language or TASK_INSTRUCTIONS[args.task]
    env = OctoSO100Wrapper(
        base_env,
        image_size=args.image_size,
        include_proprio=not args.no_proprio,
        language_instruction=language,
    )

    # Optional: launch a passive human viewer (separate from the rgb_array renderer)
    viewer = None
    if args.render_human:
        try:
            import mujoco.viewer as _mv
            viewer = _mv.launch_passive(base_env.model, base_env.data)
            print("Human viewer launched. Close the window to stop.")
        except Exception as exc:
            print(f"Warning: could not launch human viewer — {exc}")

    # -----------------------------------------------------------------------
    # 2.  Load Octo
    # -----------------------------------------------------------------------
    print(f"\nLoading Octo from: {args.model_path}")
    model = OctoModel.load_pretrained(args.model_path)
    print(f"Octo loaded.  Available dataset keys: {list(model.dataset_statistics.keys())}")

    # Pick the correct un-normalisation statistics.
    # Pre-trained model: stats are nested  {"bridge_dataset": {"action": ...}}
    # Fine-tuned model:  stats are flat    {"action": ..., "proprio": ...}
    ds_stats = model.dataset_statistics
    first_val = next(iter(ds_stats.values()))
    if isinstance(first_val, dict) and "mean" in first_val:
        # Flat layout (fine-tuned with finetune_octo_so100.py)
        unnorm_stats = ds_stats["action"]
        print("Using flat dataset statistics (fine-tuned model).")
    else:
        # Nested layout (pre-trained)
        if args.stat_key and args.stat_key in ds_stats:
            stat_key = args.stat_key
        elif "bridge_dataset" in ds_stats:
            stat_key = "bridge_dataset"
        else:
            stat_key = next(iter(ds_stats))
        print(f"Using dataset statistics key: '{stat_key}'")
        unnorm_stats = ds_stats[stat_key]["action"]

    # Wrap policy to automatically supply a new JAX RNG key each call
    policy_fn = supply_rng(
        partial(model.sample_actions, unnormalization_statistics=unnorm_stats)
    )

    # Create the language-conditioned task spec
    task = model.create_tasks(texts=[language])
    print(f"Language instruction: '{language}'\n")

    # -----------------------------------------------------------------------
    # 3.  Rollout loop
    # -----------------------------------------------------------------------
    horizon = args.history_horizon
    frames: list[np.ndarray] = []
    episode_rewards: list[float] = []
    episode_successes: list[float] = []

    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + ep)

        # Initialise observation history (pad with first obs)
        history: deque = deque(maxlen=horizon)
        history.extend([obs] * horizon)
        num_obs = 1

        ep_reward = 0.0
        ep_success = 0.0
        done = False
        step_count = 0

        while not done and step_count < args.max_episode_steps:
            # ---- build context window --------------------------------
            obs_batch = _build_obs_batch(history, num_obs, horizon)

            # ---- Octo inference --------------------------------------
            # actions: (1, pred_horizon, action_dim)  →  (pred_horizon, action_dim)
            actions = np.array(policy_fn(obs_batch, task)[0])

            # ---- execute exec_horizon actions from the chunk ---------
            for i in range(min(args.exec_horizon, len(actions))):
                if done or step_count >= args.max_episode_steps:
                    break

                # SO100 action space is [-1, 1]; clip just in case
                action = np.clip(actions[i], -1.0, 1.0)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)

                # Success metric (info may be aggregated by RHC wrappers)
                if isinstance(info, dict):
                    raw_success = info.get("success", 0)
                    ep_success = max(
                        ep_success,
                        float(raw_success[0])
                        if hasattr(raw_success, "__len__")
                        else float(raw_success),
                    )

                history.append(obs)
                num_obs += 1
                step_count += 1
                done = bool(terminated or truncated)

                # Optional: sync human viewer
                if viewer is not None and viewer.is_running():
                    viewer.sync()

                # Optional: collect frames for gif
                if args.save_gif:
                    frame = base_env.render()
                    if frame is not None:
                        frames.append(frame)

        episode_rewards.append(ep_reward)
        episode_successes.append(ep_success)
        print(
            f"Episode {ep + 1:02d}/{args.episodes} | "
            f"steps={step_count:4d} | "
            f"reward={ep_reward:7.2f} | "
            f"success={ep_success:.0f}"
        )

    # -----------------------------------------------------------------------
    # 4.  Summary
    # -----------------------------------------------------------------------
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    success_rate = sum(episode_successes) / len(episode_successes)
    print(f"\n{'─'*45}")
    print(f"Mean reward : {mean_reward:.3f}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"{'─'*45}")

    if args.save_gif and frames:
        import imageio.v2 as imageio  # type: ignore
        gif_path = Path(args.save_gif)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=15)
        print(f"Saved gif: {gif_path}")

    if viewer is not None:
        viewer.close()
    env.close()


if __name__ == "__main__":
    main()
