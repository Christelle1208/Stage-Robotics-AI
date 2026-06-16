"""Train PPO or SAC on the SO-100 pick-and-drop task with Stable-Baselines3.

Fresh run:
    python -m clean_pickdrop.train --algo ppo --timesteps 1_000_000
    python -m clean_pickdrop.train --algo sac --timesteps 1_000_000

Continue an existing run:
    python -m clean_pickdrop.train --algo ppo --run-name ppo_seed0 --timesteps 500_000 --continue
    python -m clean_pickdrop.train --algo sac --run-name sac_seed0 --timesteps 500_000 --continue

Train until a success-rate target is reached (keeps going until ≥ target or max steps hit):
    python -m clean_pickdrop.train --algo ppo --run-name ppo_seed0 --target-success 0.5 --continue
    python -m clean_pickdrop.train --algo sac --run-name sac_seed0 --target-success 0.5 --continue

--timesteps acts as a per-chunk budget when --target-success is set (default 100k per chunk).
Use --max-timesteps to cap the total in case the target is never reached (default: 10M).

Each run writes, under clean_pickdrop/runs/<run-name>/:
    eval/evaluations.npz   -- timesteps, rewards, success rate
    best_model/            -- checkpoint with the highest mean eval reward
    final_model.zip        -- model weights at the end of training
    replay_buffer.pkl      -- SAC replay buffer (saved/loaded on continuation)

TensorBoard logs land in clean_pickdrop/runs/tensorboard/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from clean_pickdrop.env import SO100PickDropEnv

RUNS_DIR = Path(__file__).resolve().parent / "runs"
TENSORBOARD_DIR = RUNS_DIR / "tensorboard"

NET_ARCH = [256, 256]

PPO_KWARGS = dict(
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=NET_ARCH),
)

SAC_KWARGS = dict(
    buffer_size=300_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    learning_starts=5_000,
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=NET_ARCH),
)

ALGO_CLS = {"ppo": PPO, "sac": SAC}
ALGO_KWARGS = {"ppo": PPO_KWARGS, "sac": SAC_KWARGS}


class StopOnSuccessRate(BaseCallback):
    """Stop training as soon as the last evaluation hits the target success rate.

    Reads from evaluations.npz written by EvalCallback after each eval round
    so there is no tight coupling to EvalCallback internals.
    """

    def __init__(self, eval_log_path: Path, target: float, verbose: int = 1):
        super().__init__(verbose)
        self._npz = eval_log_path / "evaluations.npz"
        self._target = target
        self._last_n_evals = 0

    def _on_step(self) -> bool:
        if not self._npz.exists():
            return True
        data = np.load(self._npz, allow_pickle=True)
        if "successes" not in data.files:
            return True
        n_evals = len(data["successes"])
        if n_evals <= self._last_n_evals:
            return True  # no new eval since last check
        self._last_n_evals = n_evals
        rate = float(np.mean(data["successes"][-1]))
        if self.verbose:
            print(f"  [success-rate check] {rate:.0%}  (target {self._target:.0%},"
                  f"  step {self.num_timesteps:,})")
        return rate < self._target  # returning False stops training


def build_fresh(algo: str, env, seed: int) -> PPO | SAC:
    cls = ALGO_CLS[algo]
    return cls("MlpPolicy", env, verbose=1, seed=seed,
                tensorboard_log=str(TENSORBOARD_DIR), **ALGO_KWARGS[algo])


def load_and_continue(algo: str, run_dir: Path, env) -> PPO | SAC:
    cls = ALGO_CLS[algo]
    model_path = run_dir / "final_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run without --continue first."
        )
    model = cls.load(str(model_path), env=env,
                     tensorboard_log=str(TENSORBOARD_DIR))
    print(f"Resumed {algo.upper()} from {model_path}  "
          f"(trained for {model.num_timesteps:,} steps so far)")

    if algo == "sac":
        buf_path = run_dir / "replay_buffer.pkl"
        if buf_path.exists():
            model.load_replay_buffer(str(buf_path))
            print(f"  Loaded replay buffer ({model.replay_buffer.size():,} transitions)")
        else:
            print("  Warning: no replay_buffer.pkl found -- SAC starts with an empty buffer.")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Number of *additional* timesteps to train for")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Parallel envs (default: 8 for PPO, 1 for SAC)")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--continue", dest="cont", action="store_true",
                        help="Resume training from the existing final_model in this run dir")
    parser.add_argument("--target-success", type=float, default=None, metavar="RATE",
                        help="Stop as soon as eval success rate reaches this value (e.g. 0.5 for 50%%)")
    parser.add_argument("--max-timesteps", type=int, default=10_000_000,
                        help="Hard cap on total steps when --target-success is set (default 10M)")
    return parser.parse_args()


def main():
    args = parse_args()
    n_envs = args.n_envs if args.n_envs is not None else (8 if args.algo == "ppo" else 1)
    run_name = args.run_name or f"{args.algo}_seed{args.seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        SO100PickDropEnv, n_envs=n_envs, seed=args.seed,
        monitor_dir=str(run_dir / "monitor_train"),
    )
    eval_env = make_vec_env(
        SO100PickDropEnv, n_envs=1, seed=args.seed + 1_000,
        monitor_dir=str(run_dir / "monitor_eval"),
    )

    if args.cont:
        model = load_and_continue(args.algo, run_dir, train_env)
    else:
        model = build_fresh(args.algo, train_env, args.seed)

    eval_log_path = run_dir / "eval"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(eval_log_path),
        eval_freq=max(args.eval_freq // n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    if args.target_success is not None:
        stop_callback = StopOnSuccessRate(eval_log_path, target=args.target_success)
        callbacks = [eval_callback, stop_callback]
        budget = args.max_timesteps
        print(f"Training until success rate ≥ {args.target_success:.0%} "
              f"(hard cap: {budget:,} steps)")
    else:
        callbacks = eval_callback
        budget = args.timesteps

    first_chunk = not args.cont
    model.learn(
        total_timesteps=budget,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
        reset_num_timesteps=first_chunk,
    )

    _save(model, args.algo, run_dir)
    train_env.close()
    eval_env.close()

    total = model.num_timesteps
    if args.target_success is not None:
        rate = _last_success_rate(eval_log_path)
        status = "TARGET REACHED" if (rate is not None and rate >= args.target_success) else "cap hit"
        print(f"\nDone ({status}). Success rate: {rate:.0%}  |  Total steps: {total:,}")
    else:
        print(f"\nDone. Total timesteps trained: {total:,}")
    print(f"Run artifacts: {run_dir}")
    print(f"TensorBoard:   tensorboard --logdir {TENSORBOARD_DIR}")


def _save(model, algo: str, run_dir: Path) -> None:
    model.save(str(run_dir / "final_model"))
    if algo == "sac":
        model.save_replay_buffer(str(run_dir / "replay_buffer"))
        print(f"  Saved replay buffer → {run_dir / 'replay_buffer.pkl'}")


def _last_success_rate(eval_log_path: Path) -> float | None:
    npz = eval_log_path / "evaluations.npz"
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True)
    if "successes" not in data.files or len(data["successes"]) == 0:
        return None
    return float(np.mean(data["successes"][-1]))


if __name__ == "__main__":
    main()
