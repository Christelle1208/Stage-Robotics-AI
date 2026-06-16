"""
eval_octo_so100_sim.py
======================
Zero-shot evaluation of the pretrained Octo model on the SO-100 robot arm
in a MuJoCo pick-and-place simulation, with live matplotlib visualisation
and optional GIF export.

Usage
-----
  # Basic run — pops a visualisation window at the end
  python eval_octo_so100_sim.py

  # Save a GIF of all episodes (no display needed)
  python eval_octo_so100_sim.py --save_gif outputs/octo_so100/eval.gif

  # Open the live MuJoCo viewer (requires a desktop session)
  python eval_octo_so100_sim.py --render_human

  # More episodes, different task
  python eval_octo_so100_sim.py --task pick --episodes 5

Notes
-----
Octo (pretrained on bridge_dataset) outputs 7-D EEF-delta actions
[dx, dy, dz, dyaw, dpitch, droll, gripper].
SO-100 uses 7-D joint-delta actions — the dimensions match by coincidence.
For zero-shot runs we rescale Octo's output to the env's action range.
Fine-tuning Octo on SO-100 data would produce physically meaningful actions.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — make so100_rl and octo importable from within this repo
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
AA_UNUSED = ROOT / "AA-UNUSED"
OCTO_DIR = ROOT / "octo"
SCENE_XML  = ROOT / "SIMULATION" / "mujoco_menagerie" / "trs_so_arm100" / "scene_pick_place.xml"
GOAL_IMAGE = ROOT / "outputs" / "octo_so100" / "goal_picture.png"

for p in (str(AA_UNUSED), str(OCTO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Imports (deferred after path setup)
# ---------------------------------------------------------------------------
import logging
import numpy as np
import jax

from octo.model.octo_model import OctoModel

# ---------------------------------------------------------------------------
# Suppress non-fatal Octo warnings about optional missing keys
# (wrist camera, timestep, pad_mask_dict) that don't affect single-cam eval.
# ---------------------------------------------------------------------------
class _OctoOptionalKeyFilter(logging.Filter):
    _SKIP = (
        "missing items compared to example_batch",
        "No pad_mask_dict found",
        "Skipping observation tokenizer",
    )
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._SKIP)

logging.getLogger().addFilter(_OctoOptionalKeyFilter())
from so100_rl.so100_pick_place_env import SO100PickPlaceEnv
from so100_rl.octo_wrapper import OctoSO100Wrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 2          # Octo temporal context (matched to its training setup)
DEFAULT_MODEL = "hf://rail-berkeley/octo-small-1.5"
DEFAULT_TASK = "pick_place"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Octo zero-shot eval on SO-100 MuJoCo pick-and-place"
    )
    p.add_argument(
        "--task", default=DEFAULT_TASK, choices=["reach", "pick", "pick_place"],
        help="Pick-and-place sub-task (default: pick_place)."
    )
    p.add_argument("--episodes", type=int, default=3, help="Number of episodes.")
    p.add_argument("--max_steps", type=int, default=300, help="Max env steps per episode.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=256, help="Square image size for Octo.")
    p.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL,
        help="Octo checkpoint path or HuggingFace repo id."
    )
    p.add_argument(
        "--action_scale", type=float, default=3,
        help="Global scale applied to Octo actions before feeding to the env."
    )
    p.add_argument(
        "--save_gif", type=str, default="",
        help="Path to save a GIF of all episodes (requires imageio)."
    )
    p.add_argument(
        "--render_human", action="store_true",
        help="Open a live MuJoCo viewer window (desktop session required)."
    )
    p.add_argument(
        "--no_assist_grasp", action="store_true",
        help="Disable kinematic grasp assist (harder task)."
    )
    p.add_argument(
        "--no_random_reset", action="store_true",
        help="Use a fixed reset pose instead of random."
    )
    p.add_argument(
        "--goal_conditioned", action="store_true",
        help="Use goal-image conditioning instead of language. "
             "A frame of the arm grasping the cube is rendered and passed as goal."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Goal image generation
# ---------------------------------------------------------------------------

# Joint config [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw] that
# positions the gripper near the default cube spawn (0, -0.31, 0.018) and
# closes the jaw.  Tuned empirically for the scene_pick_place.xml geometry.
# _GRAB_QPOS = np.array([0.0, -2.45, 2.10, 0.60, 0.0, 1.50], dtype=np.float32)
_GRAB_QPOS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

def generate_goal_image(
    base_env: "SO100PickPlaceEnv",
    image_size: int = 256,
) -> np.ndarray:
    """
    Render a goal image with the arm in a grasping pose over the cube.

    The function:
      1. Saves the full MuJoCo state.
      2. Moves the arm to a pre-defined grabbing config and closes the jaw.
      3. Teleports the cube to the end-effector so it looks held.
      4. Renders one frame from the tracking camera.
      5. Restores the original state before returning.

    Returns
    -------
    np.ndarray  shape (image_size, image_size, 3) uint8
    """
    import mujoco
    from so100_rl.octo_wrapper import _resize_image

    # ---- save state ----
    saved_qpos = base_env.data.qpos.copy()
    saved_qvel = base_env.data.qvel.copy()
    saved_ctrl = base_env.data.ctrl.copy()
    saved_site_pos = base_env.model.site_pos[base_env.goal_site_id].copy()

    try:
        # ---- move arm to grab pose ----
        grab = np.clip(_GRAB_QPOS, base_env.safe_ctrl_low, base_env.safe_ctrl_high)
        base_env.data.qpos[:6] = grab
        base_env.data.qvel[:] = 0.0
        base_env.data.ctrl[:] = grab
        mujoco.mj_forward(base_env.model, base_env.data)

        # Let the arm settle into position
        for _ in range(20):
            base_env.data.ctrl[:] = grab
            mujoco.mj_step(base_env.model, base_env.data)

        # ---- teleport cube to EE so it looks grasped ----
        ee = base_env._ee_pos()
        base_env._set_cube_pose(ee, zero_cube_vel=True)

        # ---- render ----
        assert base_env.renderer is not None, "renderer must not be None"
        base_env.renderer.update_scene(base_env.data, camera="track_cam")
        frame = base_env.renderer.render()          # (H, W, 3) uint8
        goal_img = _resize_image(frame, image_size)

    finally:
        # ---- restore state unconditionally ----
        base_env.data.qpos[:] = saved_qpos
        base_env.data.qvel[:] = saved_qvel
        base_env.data.ctrl[:] = saved_ctrl
        base_env.model.site_pos[base_env.goal_site_id] = saved_site_pos
        mujoco.mj_forward(base_env.model, base_env.data)

    return goal_img

# ---------------------------------------------------------------------------
# Octo observation builder
# ---------------------------------------------------------------------------
def build_octo_obs(img_window: list[np.ndarray]) -> dict[str, np.ndarray]:
    """
    Build the minimal Octo observation dict from a sliding image window.

    Only keys with known-correct shapes are included.  Optional keys that
    Octo complains about (image_wrist, timestep, task_completed, pad_mask_dict)
    are intentionally omitted — the non-fatal warnings they generate are
    suppressed by _OctoOptionalKeyFilter above.

    Parameters
    ----------
    img_window : list of (H, W, 3) uint8 arrays, length == WINDOW_SIZE

    Returns
    -------
    dict with:
      "image_primary"     : (1, T, H, W, 3) uint8
      "timestep_pad_mask" : (1, T) bool
    """
    imgs = np.stack(img_window, axis=0)[np.newaxis]   # (1, T, H, W, 3)
    pad_mask = np.ones((1, len(img_window)), dtype=bool)
    return {"image_primary": imgs, "timestep_pad_mask": pad_mask}


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------
def run_episode(
    env: OctoSO100Wrapper,
    model: OctoModel,
    max_steps: int,
    seed: int,
    rng_key,
    action_scale: float,
    collect_frames: bool,
    passive_viewer: Any = None,
    goal_image: np.ndarray | None = None,
) -> tuple[float, bool, list[np.ndarray]]:
    """
    Run one evaluation episode.

    Parameters
    ----------
    goal_image : (H, W, 3) uint8 array, optional
        When provided, Octo is run in goal-conditioned mode.
        When None, language-conditioned mode is used.

    Returns
    -------
    (total_reward, success, frames)
      frames is populated only when collect_frames=True.
    """
    obs_dict, _ = env.reset(seed=seed)

    if goal_image is not None:
        print("  Mode: goal-conditioned  (arm-grasping-cube image)")
        # Octo expects (batch, H, W, 3)
        octo_task = model.create_tasks(goals={"image_primary": goal_image[np.newaxis]})
    else:
        task_dict = env.get_task()
        lang = task_dict["language_instruction"]
        print(f"  Mode: language-conditioned  → \"{lang}\"")
        octo_task = model.create_tasks(texts=[lang])

    img_window: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    total_reward = 0.0
    success = False

    for step in range(max_steps):
        img = obs_dict["image_primary"]          # (H, W, 3) uint8
        img_window.append(img)
        if len(img_window) > WINDOW_SIZE:
            img_window.pop(0)

        # Pad the front of the window if we haven't seen enough frames yet
        padded = [img_window[0]] * (WINDOW_SIZE - len(img_window)) + img_window

        octo_obs = build_octo_obs(padded)

        rng_key, subkey = jax.random.split(rng_key)
        actions = model.sample_actions(
            octo_obs,
            octo_task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
            rng=subkey,
        )
        # actions: (1, action_horizon, action_dim)  — take first horizon step
        raw_action = np.array(actions[0, 0], dtype=np.float32)   # (7,)

        # Rescale into the env's sane range and clip
        action = np.clip(raw_action * action_scale, -1.0, 1.0)

        obs_dict, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        success = success or bool(info.get("success", False))

        if passive_viewer is not None:
            passive_viewer.sync()

        if collect_frames:
            frames.append(obs_dict["image_primary"].copy())

        if terminated or truncated:
            break

    return total_reward, success, frames


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def _make_strip(
    frames: list[np.ndarray], n_cols: int = 8
) -> np.ndarray:
    """Pick n_cols evenly-spaced frames and concatenate them horizontally."""
    if not frames:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    idxs = np.linspace(0, len(frames) - 1, min(n_cols, len(frames)), dtype=int)
    return np.concatenate([frames[i] for i in idxs], axis=1)


def save_summary_plot(
    all_rewards: list[float],
    all_successes: list[bool],
    all_frames: list[list[np.ndarray]],
    task: str,
    out_dir: Path,
) -> None:
    """Save a matplotlib figure with a per-episode image strip + result badge."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_ep = len(all_rewards)
    n_cols = 8
    fig = plt.figure(figsize=(2 * n_cols + 3, 3.5 * n_ep), constrained_layout=True)
    fig.suptitle(
        f"Octo zero-shot → SO-100 MuJoCo   |   task: {task}",
        fontsize=13, weight="bold"
    )

    outer = gridspec.GridSpec(n_ep, 1, figure=fig, hspace=0.35)

    for ep_idx in range(n_ep):
        frames = all_frames[ep_idx]
        r = all_rewards[ep_idx]
        s = all_successes[ep_idx]

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[ep_idx],
            width_ratios=[n_cols, 1], wspace=0.04
        )

        # --- image strip ---
        ax_strip = fig.add_subplot(inner[0])
        strip = _make_strip(frames, n_cols=n_cols)
        ax_strip.imshow(strip)
        ax_strip.axis("off")
        ax_strip.set_title(
            f"Episode {ep_idx + 1}   |   reward = {r:.2f}   |   "
            f"steps = {len(frames)}",
            fontsize=9, loc="left", pad=4
        )

        # --- result badge ---
        ax_badge = fig.add_subplot(inner[1])
        ax_badge.set_facecolor("#d4edda" if s else "#f8d7da")
        ax_badge.axis("off")
        ax_badge.text(
            0.5, 0.5,
            "SUCCESS\n✓" if s else "FAIL\n✗",
            ha="center", va="center",
            fontsize=11, weight="bold",
            color="#155724" if s else "#721c24",
            transform=ax_badge.transAxes,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "eval_summary.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"[vis] Summary plot → {plot_path}")

    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)


def save_gif(
    all_frames: list[list[np.ndarray]],
    gif_path: str,
    fps: int = 15,
) -> None:
    """Concatenate all episode frames and save as an animated GIF."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio  # type: ignore
        except ImportError:
            print("[vis] imageio not found — skipping GIF export. Install with: pip install imageio")
            return

    flat = [f for ep in all_frames for f in ep]
    if not flat:
        print("[vis] No frames collected — skipping GIF.")
        return

    out = Path(gif_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), flat, fps=fps, loop=0)
    print(f"[vis] GIF → {out}  ({len(flat)} frames @ {fps} fps)")


def save_metrics_json(
    all_rewards: list[float],
    all_successes: list[bool],
    task: str,
    model_path: str,
    out_dir: Path,
) -> None:
    """Write a simple JSON summary of the evaluation run."""
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "task": task,
        "model": model_path,
        "n_episodes": len(all_rewards),
        "mean_reward": float(np.mean(all_rewards)),
        "success_rate": float(np.mean(all_successes)),
        "per_episode": [
            {"reward": float(r), "success": bool(s)}
            for r, s in zip(all_rewards, all_successes)
        ],
    }
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] JSON → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load Octo model
    # ------------------------------------------------------------------
    print(f"\n[octo] Loading checkpoint: {args.model_path}")
    model = OctoModel.load_pretrained(args.model_path)
    print("[octo] Model ready.\n")

    # ------------------------------------------------------------------
    # 2. Build environment
    # ------------------------------------------------------------------
    if not SCENE_XML.exists():
        raise FileNotFoundError(
            f"MuJoCo scene XML not found: {SCENE_XML}\n"
            "Expected SIMULATION/mujoco_menagerie/trs_so_arm100/scene_pick_place.xml"
        )

    # OctoSO100Wrapper requires rgb_array (it calls env.render() to get Octo images).
    # When the user wants a live viewer we open a passive MuJoCo viewer separately
    # so both image capture and live display work at the same time.
    base_env = SO100PickPlaceEnv(
        scene_path=str(SCENE_XML),
        task=args.task,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
        assist_grasp=not args.no_assist_grasp,
        random_reset_pose=not args.no_random_reset,
    )
    env = OctoSO100Wrapper(
        base_env,
        image_size=args.image_size,
        include_proprio=False,   # Octo does not use proprio
    )

    # Optionally open a live passive viewer (separate from the rgb_array renderer)
    passive_viewer: Any = None
    if args.render_human:
        try:
            import mujoco.viewer as _mv
            passive_viewer = _mv.launch_passive(base_env.model, base_env.data)
            print("[viewer] Live MuJoCo viewer opened.")
        except Exception as exc:
            print(f"[viewer] Could not open live viewer ({exc}). Continuing without it.")

    collect_frames = True   # always collect for visualisation / GIF

    # ------------------------------------------------------------------
    # 3. (Optional) Generate goal image
    # ------------------------------------------------------------------
    goal_image: np.ndarray | None = None
    if args.goal_conditioned:
        if not GOAL_IMAGE.exists():
            raise FileNotFoundError(
                f"Goal image not found: {GOAL_IMAGE}\n"
                "Placez votre image de but ici avant de lancer avec --goal_conditioned."
            )
        try:
            import imageio.v2 as _iio
        except ImportError:
            import imageio as _iio  # type: ignore
        raw = _iio.imread(str(GOAL_IMAGE))          # (H, W, 3) ou (H, W, 4)
        if raw.ndim == 2:                            # greyscale → RGB
            raw = np.stack([raw] * 3, axis=-1)
        goal_image = raw[:, :, :3].astype(np.uint8) # drop alpha si présent
        # Redimensionner si nécessaire
        if goal_image.shape[0] != args.image_size or goal_image.shape[1] != args.image_size:
            from so100_rl.octo_wrapper import _resize_image
            goal_image = _resize_image(goal_image, args.image_size)
        print(f"[goal] Image chargée → {GOAL_IMAGE.relative_to(ROOT)}  shape={goal_image.shape}")

    print(f"[env] SO-100 MuJoCo environment built.")
    print(f"      scene  : {SCENE_XML.relative_to(ROOT)}")
    print(f"      task   : {args.task}")
    print(f"      mode   : {'goal-conditioned' if goal_image is not None else 'language-conditioned'}")
    print(f"      render : rgb_array  (live viewer: {passive_viewer is not None})\n")

    # ------------------------------------------------------------------
    # 4. Run evaluation
    # ------------------------------------------------------------------
    all_rewards: list[float] = []
    all_successes: list[bool] = []
    all_frames: list[list[np.ndarray]] = []

    rng = jax.random.PRNGKey(args.seed)

    for ep in range(args.episodes):
        rng, ep_rng = jax.random.split(rng)
        print(f"─── Episode {ep + 1}/{args.episodes} " + "─" * 40)
        reward, success, frames = run_episode(
            env=env,
            model=model,
            max_steps=args.max_steps,
            seed=args.seed + ep,
            rng_key=ep_rng,
            action_scale=args.action_scale,
            collect_frames=collect_frames,
            passive_viewer=passive_viewer,
            goal_image=goal_image,
        )
        all_rewards.append(reward)
        all_successes.append(success)
        all_frames.append(frames)
        print(f"  → reward={reward:.3f}  success={success}  frames={len(frames)}")

    env.close()
    if passive_viewer is not None:
        passive_viewer.close()

    # ------------------------------------------------------------------
    # 4. Print summary
    # ------------------------------------------------------------------
    mean_reward = float(np.mean(all_rewards))
    success_rate = float(np.mean(all_successes))

    print("\n" + "=" * 50)
    print("  Evaluation Summary")
    print("=" * 50)
    print(f"  Task         : {args.task}")
    print(f"  Model        : {args.model_path}")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Mean reward  : {mean_reward:.3f}")
    print(f"  Success rate : {success_rate * 100:.1f}%")
    print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # 5. Visualisation & export
    # ------------------------------------------------------------------
    out_dir = ROOT / "outputs" / "octo_so100"

    if collect_frames and all_frames:
        save_summary_plot(all_rewards, all_successes, all_frames, args.task, out_dir)

        if args.save_gif:
            save_gif(all_frames, args.save_gif)
        else:
            # Default GIF path when --save_gif not specified
            default_gif = str(out_dir / "eval.gif")
            save_gif(all_frames, default_gif)

    save_metrics_json(all_rewards, all_successes, args.task, args.model_path, out_dir)


if __name__ == "__main__":
    main()
