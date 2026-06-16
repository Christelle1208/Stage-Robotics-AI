"""
Evaluation script — SAC Three-Agent Pick-and-Place
===================================================

Runs detailed evaluation of individual agents or the full three-agent pipeline.
Reports success rates, mean episode lengths, reward statistics, and optionally
saves a JSON summary for benchmarking.

Usage:
  # Full pipeline evaluation (all three agents chained)
  python evaluate.py --mode full --episodes 100

  # Evaluate a single agent
  python evaluate.py --mode reach --episodes 50
  python evaluate.py --mode grasp --episodes 50
  python evaluate.py --mode place --episodes 50

  # Save results to JSON
  python evaluate.py --mode full --episodes 100 --out results.json

  # With rendering (use mjpython on macOS)
  mjpython evaluate.py --mode full --episodes 10 --render
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(__file__))

from envs import ReachEnv, PlaceEnv
from envs.grasp_env import ScriptedGrasp


# ─────────────────────────────────────────────────────────────────────────────
# Single-agent evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_single_agent(
    model:     SAC,
    env,
    n_episodes: int = 50,
    render:     bool = False,
    speed:      float = 1.0,
) -> dict:
    """
    Evaluate one agent for n_episodes and return aggregate statistics.

    Returns:
      {
        "success_rate":   float,     ← fraction of successful episodes
        "mean_steps":     float,     ← mean steps per episode
        "mean_reward":    float,     ← mean total episode reward
        "std_reward":     float,     ← standard deviation of episode rewards
        "min_reward":     float,
        "max_reward":     float,
        "episodes":       list[dict] ← per-episode detail
      }
    """
    dt = env.model.opt.timestep * env.frame_skip / speed if render else None

    successes   = []
    ep_steps    = []
    ep_rewards  = []
    episodes    = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        total_reward = 0.0
        steps        = 0

        while not (done or truncated):
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1

            if render:
                env.render()
                rem = dt - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)

        success = bool(info.get("success", False))
        successes.append(success)
        ep_steps.append(steps)
        ep_rewards.append(total_reward)
        episodes.append({"ep": ep, "success": success,
                         "steps": steps, "reward": total_reward})

        print(
            f"  ep {ep+1:3d}/{n_episodes}  "
            f"success={success}  steps={steps:4d}  reward={total_reward:+.1f}"
        )

    return {
        "success_rate": float(np.mean(successes)),
        "mean_steps":   float(np.mean(ep_steps)),
        "mean_reward":  float(np.mean(ep_rewards)),
        "std_reward":   float(np.std(ep_rewards)),
        "min_reward":   float(np.min(ep_rewards)),
        "max_reward":   float(np.max(ep_rewards)),
        "episodes":     episodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline evaluation (all three agents chained)
# ─────────────────────────────────────────────────────────────────────────────

def eval_full_pipeline(
    reach_model: SAC, scripted_grasp: ScriptedGrasp, place_model: SAC,
    reach_env: ReachEnv, place_env: PlaceEnv,
    n_episodes: int = 50,
    render:     bool = False,
    speed:      float = 1.0,
) -> dict:
    """
    Evaluate the full three-agent pick-and-place pipeline.

    Tracks per-phase success rates and the compound success rate
    (all three phases succeeded in a single episode).
    """
    from task_manager import run_episode

    per_ep   = []
    r_successes, g_successes, p_successes = [], [], []

    for ep in range(n_episodes):
        stats = run_episode(
            reach_model, scripted_grasp, place_model,
            reach_env, place_env,
            render=render, speed=speed, debug=False,
        )
        per_ep.append(stats)
        r_successes.append(stats["reach_success"])
        g_successes.append(stats["grasp_success"])
        p_successes.append(stats["place_success"])

    n = len(per_ep)
    return {
        "n_episodes":       n,
        "reach_success_rate": float(np.mean(r_successes)),
        "grasp_success_rate": float(np.mean(g_successes)),
        "place_success_rate": float(np.mean(p_successes)),
        "full_success_rate":  float(np.mean(p_successes)),  # all three must succeed
        "mean_reach_steps":   float(np.mean([s["reach_steps"] for s in per_ep])),
        "mean_grasp_steps":   float(np.mean([s["grasp_steps"] for s in per_ep])),
        "mean_place_steps":   float(np.mean([s["place_steps"] for s in per_ep])),
        "episodes":           per_ep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAC three-agent pick-and-place for SO-100."
    )
    parser.add_argument(
        "--mode",
        choices=["reach", "grasp", "place", "full"],
        default="full",
        help="What to evaluate: a single agent or the full pipeline.",
    )
    parser.add_argument("--reach",    default="models/best_reach/best_model.zip")
    parser.add_argument("--place",    default="models/best_place/best_model.zip")
    parser.add_argument("--episodes", type=int,   default=50)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--speed",    type=float, default=1.0)
    parser.add_argument(
        "--out", default=None,
        help="Save results to this JSON file path."
    )
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    results     = {}

    if args.mode == "reach":
        print(f"\nEvaluating REACH agent over {args.episodes} episodes ...")
        model = SAC.load(args.reach)
        env   = ReachEnv(render_mode=render_mode)
        results = eval_single_agent(model, env, args.episodes, args.render, args.speed)
        env.close()

    elif args.mode == "grasp":
        print("\nGRASP is scripted (no model to evaluate).")
        print("  Use --mode full to test the gripper closure in the full pipeline.")
        return

    elif args.mode == "place":
        print(f"\nEvaluating PLACE agent over {args.episodes} episodes ...")
        model = SAC.load(args.place)
        env   = PlaceEnv(render_mode=render_mode)
        results = eval_single_agent(model, env, args.episodes, args.render, args.speed)
        env.close()

    elif args.mode == "full":
        for path, name in [
            (args.reach, "Reach"), (args.place, "Place")
        ]:
            if not os.path.exists(path):
                print(f"ERROR: {name} model not found at '{path}'")
                sys.exit(1)

        print(f"\nEvaluating full pipeline over {args.episodes} episodes ...")
        reach_model = SAC.load(args.reach)
        place_model = SAC.load(args.place)
        scripted_grasp = ScriptedGrasp(render_mode=render_mode)

        reach_env = ReachEnv(render_mode=render_mode)
        place_env = PlaceEnv(render_mode=render_mode)

        results = eval_full_pipeline(
            reach_model, scripted_grasp, place_model,
            reach_env, place_env,
            n_episodes=args.episodes,
            render=args.render,
            speed=args.speed,
        )

        reach_env.close()
        place_env.close()

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS  (mode={args.mode})")
    if args.mode == "full":
        print(f"  Reach  success: {results['reach_success_rate']:.1%}")
        print(f"  Grasp  success: {results['grasp_success_rate']:.1%}")
        print(f"  Place  success: {results['place_success_rate']:.1%}")
        print(f"  Full pipeline : {results['full_success_rate']:.1%}")
        print(f"  Mean steps    : reach={results['mean_reach_steps']:.0f}  "
              f"grasp={results['mean_grasp_steps']:.0f}  "
              f"place={results['mean_place_steps']:.0f}")
    else:
        print(f"  Success rate  : {results['success_rate']:.1%}")
        print(f"  Mean steps    : {results['mean_steps']:.0f}")
        print(f"  Mean reward   : {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Reward range  : [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("=" * 60)

    # ── Save to JSON ───────────────────────────────────────────────────────
    if args.out:
        # Remove per-episode detail from top-level for readability
        summary = {k: v for k, v in results.items() if k != "episodes"}
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results saved → {args.out}")


if __name__ == "__main__":
    main()
