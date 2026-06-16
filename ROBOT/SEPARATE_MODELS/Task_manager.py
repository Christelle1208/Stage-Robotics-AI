"""
Task manager: chains two trained PPO policies to execute a full pick-and-place
sequence in MuJoCo.

  PICK  : PPO policy trained on PickEnv  — moves gripper to cube, grasps, lifts.
  PLACE : PPO policy trained on PlaceEnv — carries cube to goal, releases.

Usage:
    python Task_manager.py --episodes 5
    python Task_manager.py --episodes 5 --speed 0.5
    mjpython Task_manager.py --pick models/best_pick/best_model.zip \
                           --place models/best_place/best_model.zip
"""

import argparse
import os
import time
import yaml
import mujoco
import numpy as np
from stable_baselines3 import PPO

from Env import PickEnv, PlaceEnv

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    config = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
def run_episode(model_pick: PPO, model_place: PPO,
                pick_env: PickEnv, place_env: PlaceEnv,
                render: bool = True, speed: float = 1.0,
                debug_reward: bool = False) -> dict:
    """Runs one full pick-and-place episode. Returns outcome statistics."""
    stats = {"pick_success": False, "place_success": False,
             "pick_steps": 0, "place_steps": 0}
    dt = pick_env.model.opt.timestep * pick_env.frame_skip / speed

    # ------------------------------------------------------------------
    # Phase 1 — PICK
    # ------------------------------------------------------------------
    print("  [PICK]  starting...")
    _v = config["env"].get("reward_version", "v1")
    if debug_reward:
        if _v == "v2":
            print("  {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7} {:>8}".format(
                  "step", "dist_m", "jaw", "above_xy", "descend", "xy_lock", "jaw_al",
                  "hold", "lift_d", "drop", "TOTAL"))
        else:
            print("  {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7} {:>8}".format(
                  "step", "dist_m", "jaw", "align", "orient", "grasp", "jaw_b",
                  "hold", "lift_d", "obj_z", "TOTAL"))
    obs, _ = pick_env.reset()
    done = truncated = False
    while not (done or truncated):
        t0 = time.perf_counter()
        action, _ = model_pick.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = pick_env.step(action)
        stats["pick_steps"] += 1
        if debug_reward and stats["pick_steps"] % 10 == 0:
            bd = info.get("reward_breakdown", {})
            if _v == "v2":
                print("  {:>6} {:>7.4f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>8.4f} {:>7.3f} {:>8.3f}".format(
                      stats["pick_steps"],
                      bd.get("dist_m", 0), bd.get("jaw_rad", 0),
                      bd.get("above_xy", 0), bd.get("descend", 0),
                      bd.get("xy_lock", 0), bd.get("jaw_align", 0),
                      bd.get("hold", 0), bd.get("lift_dense", 0),
                      bd.get("drop", 0),
                      reward), end="\r")
            else:
                print("  {:>6} {:>7.4f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>8.4f} {:>7.3f} {:>8.3f}".format(
                      stats["pick_steps"],
                      bd.get("dist_m", 0), bd.get("jaw_rad", 0),
                      bd.get("align", 0), bd.get("orient", 0),
                      bd.get("grasp", 0), bd.get("jaw", 0),
                      bd.get("hold", 0), bd.get("lift_dense", 0),
                      bd.get("obj_z", 0),
                      reward), end="\r")
        if render:
            pick_env.render()
            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)
    stats["pick_success"] = info.get("success", False)
    print("  [PICK]  success={}  steps={}".format(
          stats["pick_success"], stats["pick_steps"]))
    if not stats["pick_success"]:
        print("  PICK failed — skipping PLACE.")
        return stats

    # ------------------------------------------------------------------
    # Phase 2 — PLACE
    # Inject the picked state into PlaceEnv, reuse pick_env's viewer.
    # ------------------------------------------------------------------
    print("  [PLACE] PPO taking over...")
    qpos, qctrl = pick_env.get_state_snapshot()
    obs, _ = place_env.reset_from_pick(qpos, qctrl)
    done = truncated = False
    while not (done or truncated):
        t0 = time.perf_counter()
        action, _ = model_place.predict(obs, deterministic=True)
        obs, _, done, truncated, info = place_env.step(action)
        stats["place_steps"] += 1
        if render:
            # Mirror place_env physics into pick_env viewer (only 1 viewer allowed)
            pick_env.data.qpos[:] = place_env.data.qpos
            pick_env.data.qvel[:] = place_env.data.qvel
            pick_env.data.ctrl[:] = place_env.data.ctrl
            mujoco.mj_forward(pick_env.model, pick_env.data)
            pick_env.render()
            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)
    stats["place_success"] = info.get("success", False)
    print("  [PLACE] success={}  steps={}".format(
          stats["place_success"], stats["place_steps"]))
    return stats


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick",      default="models/so100_pick_v1.zip")
    parser.add_argument("--place",     default="models/so100_place_v1.zip")
    parser.add_argument("--episodes",  type=int,   default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--speed",     type=float, default=1.0,
                        help="1.0=real time, 0.5=half speed")
    parser.add_argument("--debug-reward", action="store_true",
                        help="Affiche les composantes de reward en temps réel")
    args = parser.parse_args()

    render      = not args.no_render
    render_mode = "human" if render else None

    print("Loading PICK  model:", args.pick)
    model_pick  = PPO.load(args.pick)
    print("Loading PLACE model:", args.place)
    model_place = PPO.load(args.place)

    pick_env  = PickEnv(render_mode=render_mode)
    place_env = PlaceEnv(render_mode=render_mode)

    results = []
    for ep in range(args.episodes):
        print("\nEpisode {}/{}".format(ep + 1, args.episodes))
        stats = run_episode(model_pick, model_place, pick_env, place_env,
                            render=render, speed=args.speed,
                            debug_reward=args.debug_reward)
        results.append(stats)
        if render:
            time.sleep(0.5)

    pick_env.close()
    place_env.close()

    pick_sr  = sum(r["pick_success"]  for r in results) / len(results)
    place_sr = sum(r["place_success"] for r in results) / len(results)
    full_sr  = sum(r["pick_success"] and r["place_success"] for r in results) / len(results)
    print("\n" + "=" * 50)
    print("  Episodes     :", args.episodes)
    print("  Pick  SR     : {:.1f}%".format(pick_sr  * 100))
    print("  Place SR     : {:.1f}%".format(place_sr * 100))
    print("  Full task SR : {:.1f}%".format(full_sr  * 100))
    print("=" * 50)


if __name__ == "__main__":
    main()
