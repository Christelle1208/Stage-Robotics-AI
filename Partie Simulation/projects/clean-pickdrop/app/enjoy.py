"""Watch / evaluate a trained PPO or SAC model in the SO-100 pick-and-drop env.

Three modes:

  video        Run N episodes and save an mp4. No GUI needed.
                   python -m clean_pickdrop.enjoy --run ppo_seed0 --mode video

  stats        Run N episodes headless and print aggregate success rate /
               return -- no rendering, so you can run hundreds of episodes
               quickly for a robust generalization estimate.
                   python -m clean_pickdrop.enjoy --run ppo_seed0 --mode stats --episodes 100

  interactive  Open MuJoCo's interactive viewer and step the model live.
               On macOS this MUST run with mjpython:
                   mjpython -m clean_pickdrop.enjoy --run ppo_seed0 --mode interactive

Episodes use FRESH RANDOM cube placements every run by default (drawn from OS
entropy), so success rate reflects generalization rather than memorized seeds.
Pass --seed for a reproducible-but-still-varied sequence of episodes.

By default the script loads the best_model checkpoint (highest eval reward).
Pass --final to use the final_model saved at the end of training instead.

You can also compare both models back-to-back:
    python -m clean_pickdrop.enjoy --run ppo_seed0 sac_seed0 --mode stats --episodes 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from clean_pickdrop.env import SO100PickDropEnv

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def load_model(run_name: str, use_final: bool, env):
    from stable_baselines3 import PPO, SAC

    run_dir = RUNS_DIR / run_name
    model_path = run_dir / ("final_model.zip" if use_final else "best_model/best_model.zip")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}\n"
            f"Run `python -m clean_pickdrop.train --run-name {run_name} ...` first."
        )
    # Detect algorithm from the zip (PPO and SAC save different policy keys).
    for Algo in (PPO, SAC):
        try:
            model = Algo.load(str(model_path))
            print(f"Loaded {Algo.__name__} from {model_path}")
            return model
        except Exception:
            continue
    raise RuntimeError(f"Could not load model at {model_path} as PPO or SAC.")


def run_episodes(model, env: SO100PickDropEnv, n_episodes: int, deterministic: bool,
                 seed: int | None = None):
    """Yield (frames_list, info_dict) for each episode.

    Episode seeds are drawn from a master RNG. With seed=None (default) that
    RNG is itself seeded from OS entropy, so every run uses different cube
    placements. Pass --seed for a reproducible sequence (still varied across
    episodes, but the same sequence every run).
    """
    seed_rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        episode_seed = int(seed_rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=episode_seed)
        frames = []
        ep_reward = 0.0
        for step in range(env._max_episode_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if env.render_mode == "rgb_array":
                frames.append(env.render())
            if terminated or truncated:
                break
        yield frames, {
            "episode": ep,
            "steps": step + 1,
            "return": ep_reward,
            "success": info["is_success"],
            "phase": info["phase"],
        }


def record_video(run_names: list[str], use_final: bool, n_episodes: int, out_path: str,
                 seed: int | None) -> None:
    import imageio

    env = SO100PickDropEnv(render_mode="rgb_array")
    all_frames = []
    stats = []

    for run_name in run_names:
        model = load_model(run_name, use_final, env)
        print(f"\n--- {run_name} ---")
        for frames, ep_info in run_episodes(model, env, n_episodes, deterministic=True, seed=seed):
            all_frames.extend(frames)
            stats.append({"run": run_name, **ep_info})
            print(
                f"  ep {ep_info['episode']}: {ep_info['steps']} steps | "
                f"return={ep_info['return']:.2f} | "
                f"phase={ep_info['phase']} | success={ep_info['success']}"
            )

    env.close()
    imageio.mimsave(out_path, all_frames, fps=env.metadata["render_fps"])

    successes = [s["success"] for s in stats]
    print(f"\nSaved {len(all_frames)} frames → {out_path}")
    print(f"Overall success rate: {np.mean(successes):.0%}  ({sum(successes)}/{len(successes)} episodes)")


def run_stats(run_names: list[str], use_final: bool, n_episodes: int, seed: int | None) -> None:
    """Headless evaluation over many random episodes -- prints success rate / return stats."""
    env = SO100PickDropEnv()  # render_mode=None: no rendering overhead

    for run_name in run_names:
        model = load_model(run_name, use_final, env)
        successes, returns, lengths = [], [], []
        for _, ep_info in run_episodes(model, env, n_episodes, deterministic=True, seed=seed):
            successes.append(ep_info["success"])
            returns.append(ep_info["return"])
            lengths.append(ep_info["steps"])

        successes = np.array(successes, dtype=float)
        returns = np.array(returns)
        lengths = np.array(lengths)
        print(
            f"{run_name:<14} "
            f"success_rate={successes.mean():>5.0%} ({int(successes.sum())}/{n_episodes})  "
            f"return={returns.mean():>7.2f} ± {returns.std():.2f}  "
            f"length={lengths.mean():>5.1f}"
        )

    env.close()


def run_interactive(run_name: str, use_final: bool, seed: int | None, speed: float) -> None:
    import time

    import mujoco
    import mujoco.viewer

    env = SO100PickDropEnv()
    model = load_model(run_name, use_final, env)
    seed_rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=int(seed_rng.integers(0, 2**31 - 1)))
    ep = 0

    step_dt = (1.0 / env.metadata["render_fps"]) / speed

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                ep += 1
                print(f"episode {ep}: phase={info['phase']} | success={info['is_success']}")
                obs, _ = env.reset(seed=int(seed_rng.integers(0, 2**31 - 1)))
            elapsed = time.perf_counter() - step_start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", nargs="+", default=["ppo_seed0"],
        help="One or more run names (subdirs of clean_pickdrop/runs/)"
    )
    parser.add_argument("--mode", choices=["video", "stats", "interactive"], default="video")
    parser.add_argument("--final", action="store_true",
                        help="Load final_model instead of best_model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", default=None,
                        help="Output mp4 path (default: runs/<run>_eval.mp4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Master seed for the episode-seed sequence "
                             "(default: random, different every run)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Interactive mode only: playback speed multiplier "
                             "(1.0 = real-time, 0.5 = half speed, 2.0 = double speed)")
    args = parser.parse_args()

    if args.mode == "video":
        out = args.out or str(RUNS_DIR / ("_vs_".join(args.run) + "_eval.mp4"))
        record_video(args.run, args.final, args.episodes, out, args.seed)
    elif args.mode == "stats":
        run_stats(args.run, args.final, args.episodes, args.seed)
    else:
        if len(args.run) > 1:
            print("Interactive mode supports one run at a time; using the first:", args.run[0])
        run_interactive(args.run[0], args.final, args.seed, args.speed)


if __name__ == "__main__":
    main()
