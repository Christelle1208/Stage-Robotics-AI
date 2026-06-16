"""Compare PPO and SAC on the SO-100 pick-and-drop task: success rate + loss curves.

Reads the artifacts that train.py writes for two finished runs (one per
algorithm) and produces a single comparison figure with four panels:

    [1] success rate vs. timesteps        (from EvalCallback's is_success log)
    [2] mean episodic return vs. timesteps (same eval log)
    [3] PPO training losses (policy / value / entropy) -- from TensorBoard
    [4] SAC training losses (actor / critic / entropy-coefficient) -- from TensorBoard

Usage:
    python -m clean_pickdrop.compare --ppo-run ppo_seed0 --sac-run sac_seed0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = Path(__file__).resolve().parent / "runs"
TENSORBOARD_DIR = RUNS_DIR / "tensorboard"

PPO_LOSS_TAGS = ["train/policy_gradient_loss", "train/value_loss", "train/entropy_loss"]
SAC_LOSS_TAGS = ["train/actor_loss", "train/critic_loss", "train/ent_coef_loss"]


def load_eval_log(run_name: str):
    npz_path = RUNS_DIR / run_name / "eval" / "evaluations.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"No eval log at {npz_path}\nRun `python -m clean_pickdrop.train --algo ... "
            f"--run-name {run_name}` first."
        )
    data = np.load(npz_path, allow_pickle=True)
    timesteps = data["timesteps"]
    mean_reward = data["results"].mean(axis=1)
    success_rate = (
        np.array([np.mean(s) for s in data["successes"]])
        if "successes" in data.files else None
    )
    return timesteps, mean_reward, success_rate


def find_tb_dir(run_name: str) -> Path | None:
    candidates = sorted(TENSORBOARD_DIR.glob(f"{run_name}_*"))
    return candidates[-1] if candidates else None


def load_scalars(tb_dir: Path, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            out[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
    return out


def plot_losses(ax, tb_dir: Path | None, tags: list[str], title: str) -> None:
    if tb_dir is None:
        ax.set_title(f"{title}\n(no TensorBoard log found)")
        ax.axis("off")
        return
    scalars = load_scalars(tb_dir, tags)
    for tag, (steps, values) in scalars.items():
        ax.plot(steps, values, label=tag.split("/")[-1])
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_eval_metric(ax, runs: dict[str, tuple], values_idx: int, title: str, ylabel: str) -> None:
    colors = {"ppo": "tab:blue", "sac": "tab:orange"}
    for label, eval_log in runs.items():
        steps, values = eval_log[0], eval_log[values_idx]
        if values is None:
            continue
        ax.plot(steps, values, label=label.upper(), color=colors.get(label))
    ax.set_title(title)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-run", default="ppo_seed0")
    parser.add_argument("--sac-run", default="sac_seed0")
    parser.add_argument("--out", default=str(RUNS_DIR / "ppo_vs_sac.png"))
    args = parser.parse_args()

    ppo_eval = load_eval_log(args.ppo_run)
    sac_eval = load_eval_log(args.sac_run)
    runs = {"ppo": ppo_eval, "sac": sac_eval}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    plot_eval_metric(axes[0, 0], runs, values_idx=2, title="Success rate (eval)", ylabel="success rate")
    axes[0, 0].set_ylim(-0.05, 1.05)

    plot_eval_metric(axes[0, 1], runs, values_idx=1, title="Mean episodic return (eval)", ylabel="mean return")

    plot_losses(axes[1, 0], find_tb_dir(args.ppo_run), PPO_LOSS_TAGS, "PPO training losses")
    plot_losses(axes[1, 1], find_tb_dir(args.sac_run), SAC_LOSS_TAGS, "SAC training losses")

    fig.suptitle("PPO vs SAC -- SO-100 pick-and-drop", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"Saved comparison figure to {args.out}")


if __name__ == "__main__":
    main()
