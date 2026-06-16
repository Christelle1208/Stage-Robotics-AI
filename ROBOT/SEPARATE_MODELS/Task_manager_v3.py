"""
V3 Task manager: chains three trained PPO policies to execute a full
pick-and-place sequence.

  Stage 1 — REACH  : move open gripper to cube (ReachEnv)
  Stage 2 — GRASP  : close gripper on cube     (GraspEnv)
  Stage 3 — CARRY  : lift, carry to goal, release (CarryEnv)

Usage:
    mjpython Task_manager_v3.py \
        --reach models/so100_reach_v1_r-v3.zip \
        --grasp models/so100_grasp_v1_r-v3.zip \
        --carry models/so100_carry_v1_r-v3.zip \
        --episodes 5 --speed 1 --debug-reward
"""

import argparse
import os
import time
import yaml
import mujoco
import numpy as np
from stable_baselines3 import PPO

from Env import ReachEnv, GraspEnv, CarryEnv

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH) as _f:
    config = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
def run_episode(model_reach, model_grasp, model_carry,
                reach_env, grasp_env, carry_env,
                render=True, speed=1.0, debug_reward=False):
    """Runs one full Reach → Grasp → Carry episode. Returns outcome stats."""
    stats = {
        "reach_success": False, "grasp_success": False, "carry_success": False,
        "reach_steps": 0, "grasp_steps": 0, "carry_steps": 0,
    }
    dt = reach_env.model.opt.timestep * reach_env.frame_skip / speed

    # ------------------------------------------------------------------
    # Stage 1 — REACH
    # ------------------------------------------------------------------
    print("  [REACH] starting...")
    if debug_reward:
        print("  {:>6} {:>7} {:>7} {:>7} {:>9} {:>9} {:>8} {:>7} {:>8}".format(
              "step", "dist_m", "jaw", "wrist", "dist_r", "center", "jaw_op", "hold_st", "TOTAL"))
    obs, _ = reach_env.reset()
    done = truncated = False
    while not (done or truncated):
        t0 = time.perf_counter()
        action, _ = model_reach.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = reach_env.step(action)
        stats["reach_steps"] += 1
        if debug_reward and stats["reach_steps"] % 10 == 0:
            bd = info.get("reward_breakdown", {})
            print("  {:>6} {:>7.4f} {:>7.3f} {:>7.3f} {:>9.3f} {:>9.3f} {:>8.3f} {:>7d} {:>8.3f}".format(
                  stats["reach_steps"],
                  bd.get("dist_m", 0), bd.get("jaw_rad", 0),
                  bd.get("wrist", 0),
                  bd.get("dist", 0), bd.get("centering", 0),
                  bd.get("jaw_open", 0),
                  int(bd.get("hold_steps", 0)),
                  reward), end="\r")
        if render:
            reach_env.render()
            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    stats["reach_success"] = info.get("success", False)
    print(f"\n  [REACH] success={stats['reach_success']}  steps={stats['reach_steps']}")
    if not stats["reach_success"]:
        print("  REACH failed — skipping GRASP and CARRY.")
        return stats

    # ------------------------------------------------------------------
    # Stage 2 — GRASP
    # Inject reach state into GraspEnv (curriculum handoff).
    # ------------------------------------------------------------------
    print("  [GRASP] starting...")
    if debug_reward:
        print("  {:>6} {:>7} {:>7} {:>7} {:>9} {:>8} {:>7} {:>8}".format(
              "step", "dist_m", "jaw", "wrist", "jaw_cl", "dist_p", "hold", "TOTAL"))
    qpos, qctrl = reach_env.get_state_snapshot()
    obs, _ = grasp_env.reset_from_reach(qpos, qctrl)
    done = truncated = False
    while not (done or truncated):
        t0 = time.perf_counter()
        action, _ = model_grasp.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = grasp_env.step(action)
        stats["grasp_steps"] += 1
        if debug_reward and stats["grasp_steps"] % 10 == 0:
            bd = info.get("reward_breakdown", {})
            print("  {:>6} {:>7.4f} {:>7.3f} {:>7.3f} {:>9.3f} {:>8.3f} {:>7.3f} {:>8.3f}".format(
                  stats["grasp_steps"],
                  bd.get("dist_m", 0), bd.get("jaw_rad", 0),
                  bd.get("wrist", 0),
                  bd.get("jaw_close", 0), bd.get("dist_pen", 0),
                  bd.get("hold", 0),
                  reward), end="\r")
        if render:
            # Mirror grasp_env physics into reach_env viewer (single viewer)
            reach_env.data.qpos[:] = grasp_env.data.qpos
            reach_env.data.qvel[:] = grasp_env.data.qvel
            reach_env.data.ctrl[:] = grasp_env.data.ctrl
            mujoco.mj_forward(reach_env.model, reach_env.data)
            reach_env.render()
            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    stats["grasp_success"] = info.get("success", False)
    print(f"\n  [GRASP] success={stats['grasp_success']}  steps={stats['grasp_steps']}")
    if not stats["grasp_success"]:
        print("  GRASP failed — skipping CARRY.")
        return stats

    # ------------------------------------------------------------------
    # Stage 3 — CARRY
    # Inject grasp state into CarryEnv.
    # ------------------------------------------------------------------
    print("  [CARRY] starting...")
    if debug_reward:
        print("  {:>6} {:>7} {:>7} {:>8} {:>8} {:>7} {:>8}".format(
              "step", "g_dist", "jaw", "drop", "goal_d", "rel", "TOTAL"))
    qpos, qctrl = grasp_env.get_state_snapshot()
    obs, _ = carry_env.reset_from_grasp(qpos, qctrl)
    done = truncated = False
    while not (done or truncated):
        t0 = time.perf_counter()
        action, _ = model_carry.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = carry_env.step(action)
        stats["carry_steps"] += 1
        if debug_reward and stats["carry_steps"] % 10 == 0:
            bd = info.get("reward_breakdown", {})
            print("  {:>6} {:>7.4f} {:>7.3f} {:>8.3f} {:>8.3f} {:>7.3f} {:>8.3f}".format(
                  stats["carry_steps"],
                  bd.get("dist_m", 0), bd.get("jaw_rad", 0),
                  bd.get("drop", 0), bd.get("goal_dist", 0),
                  bd.get("release", 0),
                  reward), end="\r")
        if render:
            reach_env.data.qpos[:] = carry_env.data.qpos
            reach_env.data.qvel[:] = carry_env.data.qvel
            reach_env.data.ctrl[:] = carry_env.data.ctrl
            mujoco.mj_forward(reach_env.model, reach_env.data)
            reach_env.render()
            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    stats["carry_success"] = info.get("success", False)
    print(f"\n  [CARRY] success={stats['carry_success']}  steps={stats['carry_steps']}")
    return stats


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach",     default="models/best_reach/best_model.zip")
    parser.add_argument("--grasp",     default="models/best_grasp/best_model.zip")
    parser.add_argument("--carry",     default="models/best_carry/best_model.zip")
    parser.add_argument("--episodes",  type=int,   default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--speed",     type=float, default=1.0,
                        help="1.0=real time, 0.5=half speed")
    parser.add_argument("--debug-reward", action="store_true",
                        help="Print per-step reward breakdown")
    args = parser.parse_args()

    render      = not args.no_render
    render_mode = "human" if render else None

    print("Loading REACH model:", args.reach)
    model_reach = PPO.load(args.reach)
    print("Loading GRASP model:", args.grasp)
    model_grasp = PPO.load(args.grasp)
    print("Loading CARRY model:", args.carry)
    model_carry = PPO.load(args.carry)

    reach_env = ReachEnv(render_mode=render_mode)
    grasp_env = GraspEnv()   # no viewer — shares reach_env's viewer
    carry_env = CarryEnv()

    results = []
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}")
        stats = run_episode(
            model_reach, model_grasp, model_carry,
            reach_env, grasp_env, carry_env,
            render=render, speed=args.speed,
            debug_reward=args.debug_reward,
        )
        results.append(stats)

    # Summary
    print("\n" + "="*50)
    print(f"  Reach  success: {sum(r['reach_success'] for r in results)}/{args.episodes}")
    print(f"  Grasp  success: {sum(r['grasp_success'] for r in results)}/{args.episodes}")
    print(f"  Carry  success: {sum(r['carry_success'] for r in results)}/{args.episodes}")
    print("="*50)

    reach_env.close()
    grasp_env.close()
    carry_env.close()


if __name__ == "__main__":
    main()
