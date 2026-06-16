"""
Task Manager — chains three SAC agents for a complete pick-and-place sequence
==============================================================================

Paper reference: MDPI Biomimetics 8(2):240
  "The three agents are composed sequentially: at inference time, the
   simulator state is transferred from one agent to the next upon success.
   If a sub-task fails (truncated without success), the episode is aborted
   and counted as a failure."

Execution pipeline:
  Phase 1 — REACH   : home pose → EE above object, gripper open
  Phase 2 — GRASP   : EE above object → object lifted, gripper closed
  Phase 3 — PLACE   : object lifted → placed at goal, gripper open

State transfer mechanism:
  Between phases, env.get_state_snapshot() returns (qpos, qctrl), and the
  downstream env.reset_from_*() injects the exact MuJoCo simulator state.
  This guarantees seamless continuity — no teleportation, no keyframe gap.

Usage:
  # Run 5 full pick-and-place episodes with rendering
  python task_manager.py --episodes 5 --render

  # Run 100 evaluation episodes without rendering (headless)
  python task_manager.py --episodes 100

  # Specify custom model paths
  python task_manager.py \\
    --reach  models/best_reach/best_model.zip \\
    --grasp  models/best_grasp/best_model.zip \\
    --place  models/best_place/best_model.zip \\
    --episodes 10 --render

  # Use from mjpython for macOS MuJoCo viewer
  mjpython task_manager.py --episodes 5 --render
"""

import argparse
import time
import sys
import os
import numpy as np
import mujoco
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(__file__))

from envs import ReachEnv, PlaceEnv, BoxPlaceEnv
from envs.base_env import CONFIG as _CFG
from envs.grasp_env import ScriptedGrasp


# ─────────────────────────────────────────────────────────────────────────────
# Default model paths
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_REACH     = "models/best_reach/best_model.zip"
DEFAULT_PLACE     = "models/best_place/best_model.zip"
DEFAULT_PLACE_BOX = "models/best_place_box/best_model.zip"


# ─────────────────────────────────────────────────────────────────────────────
# Scripted finish: open gripper + return to home pose
# ─────────────────────────────────────────────────────────────────────────────

_HOME_QPOS = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]   # robot home joint angles
_OPEN_JAW  = 1.2                                      # fully open jaw angle


def _scripted_finish(place_env, render_fn, dt: float, open_steps: int = 60,
                     home_steps: int = 120):
    """
    Scripted post-place sequence:
      Phase A — open the gripper (increment jaw ctrl over open_steps)
      Phase B — return all joints to home pose (linear interpolation)
    """
    # Phase A: open jaw
    for _ in range(open_steps):
        place_env.data.ctrl[5] = min(place_env.data.ctrl[5] + 0.03, _OPEN_JAW)
        mujoco.mj_step(place_env.model, place_env.data, place_env.frame_skip)
        render_fn()
        time.sleep(dt)

    # Phase B: interpolate joints to home
    start_ctrl = place_env.data.ctrl[:].copy()
    for step in range(home_steps):
        alpha = (step + 1) / home_steps
        for i in range(6):
            place_env.data.ctrl[i] = (1 - alpha) * start_ctrl[i] + alpha * _HOME_QPOS[i]
        mujoco.mj_step(place_env.model, place_env.data, place_env.frame_skip)
        render_fn()
        time.sleep(dt)


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    reach_model: SAC,
    scripted_grasp: ScriptedGrasp,
    place_model: SAC,
    reach_env:   ReachEnv,
    place_env:   PlaceEnv,
    render:      bool  = True,
    speed:       float = 1.0,
    debug:       bool  = False,
) -> dict:
    """
    Execute one complete pick-and-place episode.

    Returns a statistics dict with per-phase outcomes:
      {
        "reach_success": bool,  "reach_steps": int,
        "grasp_success": bool,  "grasp_steps": int,
        "place_success": bool,  "place_steps": int,
        "full_success":  bool   (all three phases succeeded)
      }
    """
    stats = {
        "reach_success": False, "reach_steps": 0,
        "grasp_success": False, "grasp_steps": 0,
        "place_success": False, "place_steps": 0,
        "full_success":  False,
    }

    # dt controls real-time playback speed (render=True only)
    dt = reach_env.model.opt.timestep * reach_env.frame_skip / speed

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1 — REACH
    # ──────────────────────────────────────────────────────────────────────
    print("  [REACH] starting ...")
    obs, _ = reach_env.reset()

    # For the box scene: apply the fixed box position to reach_env immediately
    # so it is visible in the viewer from the first step of reach.
    from envs.box_place_env import BoxPlaceEnv as _BoxPlaceEnv
    _box_scene = isinstance(place_env, _BoxPlaceEnv)
    if _box_scene:
        reach_env.model.body("goal").pos[:] = [
            place_env.BOX_FIXED_X, place_env.BOX_FIXED_Y, 0.0
        ]
        mujoco.mj_forward(reach_env.model, reach_env.data)

    done = truncated = False

    while not (done or truncated):
        t0 = time.perf_counter()
        action, _ = reach_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = reach_env.step(action)
        stats["reach_steps"] += 1

        if render:
            reach_env.render()
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

        if debug and stats["reach_steps"] % 20 == 0:
            bd = info.get("reward_breakdown", {})
            print(
                f"  [REACH] step={stats['reach_steps']:4d} "
                f"dist={bd.get('dist_m', '?'):.3f}m "
                f"jaw={bd.get('jaw_rad', '?'):.2f}rad "
                f"r={reward:.3f}"
            )

    stats["reach_success"] = info.get("success", False)
    print(f"  [REACH] success={stats['reach_success']}  steps={stats['reach_steps']}")

    if not stats["reach_success"]:
        print("  ✗ Reach failed — aborting episode.\n")
        return stats

    # Transfer state to ScriptedGrasp
    qpos, qctrl = reach_env.get_state_snapshot()

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2 — GRASP (scripted: close gripper)
    # ──────────────────────────────────────────────────────────────────────
    print("  [GRASP] scripted jaw closure ...")
    qpos, qctrl, grasp_ok = scripted_grasp.run(reach_env)
    stats["grasp_steps"] = scripted_grasp.close_steps + scripted_grasp.settle_steps
    stats["grasp_success"] = grasp_ok
    print(f"  [GRASP] success={grasp_ok}  steps={stats['grasp_steps']}")

    if not grasp_ok:
        print("  ✗ Grasp failed — aborting episode.\n")
        return stats

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3 — PLACE
    # ──────────────────────────────────────────────────────────────────────
    print("  [PLACE] starting from Grasp terminal state ...")
    obs, _ = place_env.reset_from_grasp(qpos, qctrl)
    done = truncated = False

    # Sync the goal body position into reach_env so the viewer shows it.
    # For box scene: sync the box centre; for flat: sync the goal marker.
    if place_env.goal_pos is not None:
        if _box_scene:
            reach_env.model.body("goal").pos[:] = place_env.model.body("goal").pos
        else:
            reach_env.model.body("goal").pos[:] = place_env.goal_pos

    def _render_place():
        if not render:
            return
        # Both scenes now share the same XML loaded in reach_env → always mirror.
        reach_env.data.qpos[:] = place_env.data.qpos
        reach_env.data.qvel[:] = place_env.data.qvel
        reach_env.data.ctrl[:] = place_env.data.ctrl
        mujoco.mj_forward(reach_env.model, reach_env.data)
        reach_env.render()

    _render_place()  # show initial place state before first step

    # For the box scene the goal is 10-15 cm to the side — the flat-trained
    # model doesn't reliably keep the jaw closed over that distance.
    # Force jaw action to -1 (close direction) while the cube is far from the
    # box centre, and only release control once the agent is close enough.
    JAW_LOCK_DIST = place_env.place_threshold * 2.0  # within 2× threshold → free jaw

    while not (done or truncated):
        t0 = time.perf_counter()
        action, _ = place_model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = place_env.step(action)
        stats["place_steps"] += 1

        if render:
            _render_place()
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

        if debug and stats["place_steps"] % 20 == 0:
            bd = info.get("reward_breakdown", {})
            if "goal_dist_m" in bd:
                detail = f"goal_dist={bd['goal_dist_m']:.3f}m"
            else:
                detail = (f"obj_z={bd.get('obj_z', '?')}"
                          f"  xy={bd.get('xy_dist', '?')}"
                          f"  3d={bd.get('dist3d', '?')}")
            print(f"  [PLACE] step={stats['place_steps']:4d} {detail}  r={reward:.3f}")

    stats["place_success"] = info.get("success", False)
    stats["full_success"]  = stats["place_success"]
    print(f"  [PLACE] success={stats['place_success']}  steps={stats['place_steps']}")

    # Phase 4 — scripted finish: open gripper + return to home
    # Runs regardless of success so the robot always ends in a clean state.
    print("  [FINISH] opening gripper and returning to home ...")
    _scripted_finish(place_env, _render_place, dt)
    print("  [FINISH] done")

    status = "✓ SUCCESS" if stats["full_success"] else "✗ FAILED"
    print(f"  {status}\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run full pick-and-place with three chained SAC agents."
    )
    parser.add_argument(
        "--reach",  default=DEFAULT_REACH,
        help="Path to trained Reach SAC model (.zip)"
    )
    parser.add_argument(
        "--place",  default=None,
        help="Path to trained Place SAC model (.zip). Auto-selected from --scene if omitted."
    )
    parser.add_argument(
        "--scene",
        choices=["flat", "box"],
        default="flat",
        help="Scene variant: 'flat' (ground goal marker) or 'box' (open box to drop into).",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of full pick-and-place episodes to run."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Open MuJoCo viewer. Use 'mjpython' on macOS."
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (>1 = faster, <1 = slower)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print per-step reward breakdowns."
    )
    args = parser.parse_args()

    # Resolve default place model path from scene
    if args.place is None:
        args.place = DEFAULT_PLACE_BOX if args.scene == "box" else DEFAULT_PLACE
    PlaceEnvCls = BoxPlaceEnv if args.scene == "box" else PlaceEnv

    # ── macOS viewer requires mjpython ─────────────────────────────────────
    if args.render and sys.platform == "darwin":
        import subprocess, shutil
        if shutil.which("mjpython") is None and "mjpython" not in sys.executable:
            print("WARNING: On macOS, run with mjpython for the viewer to work:")
            print("  mjpython task_manager.py --render")

    # ── Load models ────────────────────────────────────────────────────────
    for path, name in [(args.reach, "Reach"), (args.place, "Place")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} model not found at '{path}'")
            print(f"  Train it first: python train.py --task {name.lower()}")
            sys.exit(1)

    print("\nLoading models ...")
    reach_model = SAC.load(args.reach)
    place_model = SAC.load(args.place)
    render_mode = "human" if args.render else None
    scripted_grasp = ScriptedGrasp(render_mode=render_mode)
    print(f"  Reach: {args.reach}")
    print(f"  Grasp: scripted (no model)")
    print(f"  Place: {args.place}")

    # ── Create environments ────────────────────────────────────────────────
    # When using the box scene, reach_env must also load scene_box.xml so
    # that (a) the box is visible during reach/grasp, and (b) both envs share
    # the same model → the passive viewer can be mirrored without close/reopen.
    if args.scene == "box":
        _orig_xml = _CFG["env"]["xml_path"]
        _CFG["env"]["xml_path"] = _CFG["scene_variants"]["box"]["xml_path"]
        reach_env = ReachEnv(render_mode=render_mode)
        _CFG["env"]["xml_path"] = _orig_xml
    else:
        reach_env = ReachEnv(render_mode=render_mode)
    place_env = PlaceEnvCls(render_mode=render_mode)

    # ── Run episodes ───────────────────────────────────────────────────────
    all_stats = []
    print(f"\nRunning {args.episodes} episode(s)  (render={args.render})\n")

    for ep in range(args.episodes):
        print(f"─── Episode {ep + 1} / {args.episodes} " + "─" * 40)
        stats = run_episode(
            reach_model, scripted_grasp, place_model,
            reach_env, place_env,
            render=args.render,
            speed=args.speed,
            debug=args.debug,
        )
        all_stats.append(stats)

    # ── Summary ────────────────────────────────────────────────────────────
    n  = len(all_stats)
    r_ok = sum(s["reach_success"] for s in all_stats)
    g_ok = sum(s["grasp_success"] for s in all_stats)
    p_ok = sum(s["place_success"] for s in all_stats)

    print("\n" + "=" * 60)
    print(f"  SUMMARY  ({n} episodes)")
    print(f"  Reach  success rate : {r_ok}/{n}  ({100*r_ok/n:.0f}%)")
    print(f"  Grasp  success rate : {g_ok}/{n}  ({100*g_ok/n:.0f}%)")
    print(f"  Place  success rate : {p_ok}/{n}  ({100*p_ok/n:.0f}%)")
    print(f"  Overall (all three) : {p_ok}/{n}  ({100*p_ok/n:.0f}%)")
    print("=" * 60)

    reach_env.close()
    place_env.close()


if __name__ == "__main__":
    main()
