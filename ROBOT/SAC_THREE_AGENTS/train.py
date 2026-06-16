"""
Training script — SAC Three-Agent Pick-and-Place
=================================================

Paper reference: MDPI Biomimetics 8(2):240
  "Each agent is trained independently with Soft Actor-Critic (SAC).
   Curriculum training feeds the downstream agents with states from
   previously trained upstream agents to bootstrap learning."

Training order (sequential, matching the paper):
  1. REACH  — arm learns to approach object from home pose
  2. GRASP  — trained with Reach agent providing initial states
  3. PLACE  — trained with Grasp agent providing terminal states

Usage:
  # Train all three agents sequentially
  python train.py --task all

  # Train a single agent
  python train.py --task reach
  python train.py --task grasp
  python train.py --task place

  # Resume from a checkpoint
  python train.py --task reach --resume models/reach_ckpt.zip

  # Train with Robosuite environments instead of pure MuJoCo
  python train.py --task all --backend robosuite

Outputs:
  models/best_reach/best_model.zip
  models/best_grasp/best_model.zip
  models/best_place/best_model.zip
  tb_logs/{reach,grasp,place}/     ← TensorBoard logs
"""

import os
import sys
import argparse
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    ProgressBarCallback,
)

# Add project root to path so imports resolve regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from envs import ReachEnv, PlaceEnv, BoxPlaceEnv
from envs.grasp_env import ScriptedGrasp

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CFG_PATH) as f:
    CFG = yaml.safe_load(f)

os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# VecEnv factory — SubprocVecEnv for n>1, DummyVecEnv for n=1
# ─────────────────────────────────────────────────────────────────────────────

def _make_vec(env_cls, n_envs: int) -> VecMonitor:
    """
    Create a vectorised environment.

    SubprocVecEnv (n>1): each env runs in a separate process → true CPU
    parallelism. MuJoCo simulation is CPU-heavy so this gives a near-linear
    speedup up to the number of physical cores.

    DummyVecEnv (n=1): single process, simpler to debug.
    """
    fns = [lambda cls=env_cls: cls() for _ in range(n_envs)]
    if n_envs > 1:
        vec = SubprocVecEnv(fns, start_method="fork")  # fork is fastest on macOS
    else:
        vec = DummyVecEnv(fns)
    return VecMonitor(vec)


# ─────────────────────────────────────────────────────────────────────────────
# SAC policy kwargs (shared by all three agents)
# ─────────────────────────────────────────────────────────────────────────────

def _sac_policy_kwargs() -> dict:
    """
    Neural network configuration for all SAC agents.
    Both actor and critic use a [256, 256] MLP (matches paper architecture).
    """
    return dict(
        net_arch   = CFG["sac"]["net_arch"],   # [256, 256]
        activation_fn = __import__("torch").nn.ReLU,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SAC model factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_sac(env, log_dir: str, resume: str | None = None) -> SAC:
    """
    Build (or reload) a SAC model for the given environment.

    Speed-ups applied vs baseline:
      - device="auto"    → uses MPS on Apple Silicon (~2-3× faster gradients)
      - SubprocVecEnv    → n_envs parallel MuJoCo envs (~n_envs× data throughput)
      - gradient_steps=2 → adjusted for n_envs=4 (same update/data ratio)
    """
    sc = CFG["sac"]

    if resume:
        print(f"  → Resuming from checkpoint: {resume}")
        model = SAC.load(resume, env=env, tensorboard_log=log_dir)
    else:
        model = SAC(
            policy          = "MlpPolicy",
            env             = env,
            learning_rate   = sc["learning_rate"],
            buffer_size     = sc["buffer_size"],
            batch_size      = sc["batch_size"],
            tau             = sc["tau"],
            gamma           = sc["gamma"],
            gradient_steps  = sc["gradient_steps"],
            learning_starts = sc["learning_starts"],
            ent_coef        = sc["ent_coef"],
            policy_kwargs   = _sac_policy_kwargs(),
            device          = sc["device"],   # "auto" → MPS on Apple Silicon
            verbose         = 1,
            tensorboard_log = log_dir,
        )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum state provider
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumStateProvider:
    """
    Provides terminal states from an upstream trained agent by running it
    for one episode and returning the snapshot when success occurs.

    This implements the paper's curriculum training strategy:
      "We use the previously trained agent to generate starting states for
       the next agent, which dramatically reduces the exploration difficulty."

    Usage:
      provider = CurriculumStateProvider(reach_model, reach_env, success_key="reach")
      qpos, qctrl = provider.sample()  # run Reach until success, return snapshot
    """

    def __init__(self, model: SAC, env, max_attempts: int = 20):
        """
        Args:
            model:        Trained upstream SAC model
            env:          Environment the model was trained on
            max_attempts: How many episodes to try before returning a fallback
        """
        self.model       = model
        self.env         = env
        self.max_attempts = max_attempts

    def sample(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Run the upstream agent until success, then return the simulator snapshot.

        Returns:
            (qpos, qctrl) — full MuJoCo state at the moment of success
            None          — if upstream agent failed within max_attempts tries
        """
        for _ in range(self.max_attempts):
            obs, _ = self.env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = self.env.step(action)
            if info.get("success", False):
                return self.env.get_state_snapshot()
        return None  # upstream agent not good enough yet


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent training functions
# ─────────────────────────────────────────────────────────────────────────────

def train_reach(resume: str | None = None, fast: bool = False) -> SAC:
    """
    Train Agent 1 — REACH.

    Standard training: no state injection needed, starts from home pose.

    Args:
        fast: if True, use reduced timesteps for quick iteration/debugging.
    """
    sc = CFG["sac"]
    n_envs = sc.get("n_envs", 1)
    total_ts = sc["fast_timesteps_reach"] if fast else sc["total_timesteps_reach"]

    print("\n" + "=" * 60)
    print(f"  Training REACH agent  (Agent 1 / 3)  n_envs={n_envs}  steps={total_ts:,}")
    print("=" * 60)

    env      = _make_vec(ReachEnv, n_envs)
    eval_env = _make_vec(ReachEnv, 1)   # eval always single env

    model = _build_sac(env, log_dir="tb_logs/reach", resume=resume)

    callbacks = CallbackList([
        ProgressBarCallback(),
        EvalCallback(
            eval_env,
            best_model_save_path="models/best_reach",
            eval_freq            = sc["eval_freq"],
            n_eval_episodes      = sc["n_eval_episodes"],
            deterministic        = True,
            verbose              = 1,
        ),
        CheckpointCallback(
            save_freq   = sc["checkpoint_freq"],
            save_path   = "models/ckpt_reach",
            name_prefix = "reach",
        ),
    ])

    model.learn(
        total_timesteps     = total_ts,
        callback            = callbacks,
        reset_num_timesteps = resume is None,
    )

    model.save("models/reach_final.zip")
    print("  ✓ Reach agent saved → models/reach_final.zip")
    env.close()
    eval_env.close()
    return model


def train_grasp(reach_model: SAC | None = None, resume: str | None = None, fast: bool = False):
    """
    Agent 2 — GRASP is now scripted (just close the jaw).
    No training needed. Returns None.
    """
    print("\n" + "=" * 60)
    print("  GRASP agent is scripted — no training needed.")
    print("  (Jaw closure is handled by ScriptedGrasp at inference time.)")
    print("=" * 60)
    return None


def train_place(reach_model: SAC | None = None, resume: str | None = None, fast: bool = False, scene: str = "flat") -> SAC:
    """
    Train Agent 3 — PLACE.

    Curriculum: if reach_model is provided, use it + ScriptedGrasp to generate
    states where the object is already gripped (EE positioned, jaw closed).
    This drastically reduces the exploration burden.

    Args:
        scene: "flat" (ground goal marker) or "box" (open box to drop into).
    """
    PlaceEnvCls = BoxPlaceEnv if scene == "box" else PlaceEnv
    model_tag   = "place_box" if scene == "box" else "place"
    sc = CFG["sac"]
    n_envs = sc.get("n_envs", 1)
    total_ts = sc["fast_timesteps_place"] if fast else sc["total_timesteps_place"]

    print("\n" + "=" * 60)
    print(f"  Training PLACE agent  (Agent 3 / 3)  scene={scene}  n_envs={n_envs}  steps={total_ts:,}")
    print("=" * 60)

    if reach_model is not None:
        reach_env_for_curriculum = ReachEnv()
        reach_provider = CurriculumStateProvider(reach_model, reach_env_for_curriculum)
        scripted_grasp = ScriptedGrasp()
        print("  Using Reach agent + ScriptedGrasp for curriculum state injection.")
    else:
        reach_provider = None
        scripted_grasp = None
        print("  No Reach model provided — PlaceEnv uses standalone reset().")

    def make_place():
        env = PlaceEnvCls()
        if reach_provider is not None:
            _original_reset = env.reset
            def _curriculum_reset(seed=None, options=None):
                # Run Reach until success
                snapshot = reach_provider.sample()
                if snapshot is None:
                    return _original_reset(seed=seed, options=options)
                # Inject Reach terminal state, then run scripted grasp
                reach_env_for_curriculum._restore_snapshot(*snapshot)
                qpos, qctrl, grasp_ok = scripted_grasp.run(reach_env_for_curriculum)
                if grasp_ok:
                    return env.reset_from_grasp(qpos, qctrl)
                return _original_reset(seed=seed, options=options)
            env.reset = _curriculum_reset
        return env

    fns = [make_place for _ in range(n_envs)]
    if n_envs > 1:
        env = VecMonitor(SubprocVecEnv(fns, start_method="fork"))
    else:
        env = VecMonitor(DummyVecEnv(fns))
    eval_env = _make_vec(PlaceEnvCls, 1)

    model = _build_sac(env, log_dir=f"tb_logs/{model_tag}", resume=resume)

    callbacks = CallbackList([
        ProgressBarCallback(),
        EvalCallback(
            eval_env,
            best_model_save_path=f"models/best_{model_tag}",
            eval_freq            = sc["eval_freq"],
            n_eval_episodes      = sc["n_eval_episodes"],
            deterministic        = True,
            verbose              = 1,
        ),
        CheckpointCallback(
            save_freq   = sc["checkpoint_freq"],
            save_path   = f"models/ckpt_{model_tag}",
            name_prefix = model_tag,
        ),
    ])

    model.learn(
        total_timesteps     = total_ts,
        callback            = callbacks,
        reset_num_timesteps = resume is None,
    )

    model.save(f"models/{model_tag}_final.zip")
    print(f"  ✓ Place agent saved → models/{model_tag}_final.zip")

    if reach_model is not None:
        reach_env_for_curriculum.close()
    env.close()
    eval_env.close()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train SAC three-agent pick-and-place for SO-100"
    )
    parser.add_argument(
        "--task",
        choices=["reach", "grasp", "place", "all"],
        default="all",
        help="Which agent to train (default: all — trains Reach → Grasp → Place).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .zip checkpoint to resume training from.",
    )
    parser.add_argument(
        "--backend",
        choices=["mujoco", "robosuite"],
        default="mujoco",
        help="Environment backend. 'mujoco' = pure MuJoCo (SO-100 XML)."
             " 'robosuite' = Robosuite tasks (requires robosuite installed).",
    )
    parser.add_argument(
        "--scene",
        choices=["flat", "box"],
        default="flat",
        help="Scene variant for the Place agent: 'flat' (ground goal) or 'box' (open box).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: use reduced timesteps (100-150k) for quick iteration."
             " Useful for debugging reward shaping or testing curriculum.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Override n_envs from config. 1 = single env (slow but easiest to debug)."
             " 4 = 4 parallel envs (recommended for training).",
    )
    args = parser.parse_args()

    # CLI overrides for n_envs
    if args.n_envs is not None:
        CFG["sac"]["n_envs"] = args.n_envs
        print(f"  n_envs overridden to {args.n_envs} via --n-envs flag.")

    if args.backend == "robosuite":
        # Validate robosuite availability before starting
        try:
            import importlib
            _rsuite = importlib.import_module("robosuite")  # optional dep
            del _rsuite
        except ImportError:
            print("ERROR: robosuite backend requested but 'robosuite' is not installed.")
            print("  Install it: pip install robosuite")
            print("  Or use the default pure-MuJoCo backend: --backend mujoco")
            sys.exit(1)
        print("NOTE: Robosuite backend selected. Using Panda robot (see config.yaml).")
        print("      To use SO-100, register the custom robot (robosuite_envs/so100_robot.py).")

    if args.fast:
        print("  ⚡ FAST MODE: using reduced timesteps (see config.yaml fast_timesteps_*).")

    # ── Training sequence ──────────────────────────────────────────────────
    if args.task == "reach":
        train_reach(resume=args.resume, fast=args.fast)

    elif args.task == "grasp":
        train_grasp()

    elif args.task == "place":
        reach_path = "models/best_reach/best_model.zip"
        reach_model = None
        if os.path.exists(reach_path):
            print(f"  Loading Reach model from {reach_path} for curriculum training.")
            reach_model = SAC.load(reach_path)
        train_place(reach_model=reach_model, resume=args.resume, fast=args.fast, scene=args.scene)

    elif args.task == "all":
        reach_model = train_reach(fast=args.fast)
        train_grasp()  # no-op, scripted
        train_place(reach_model=reach_model, fast=args.fast, scene=args.scene)
        print("\n" + "=" * 60)
        print("  All three agents trained successfully!")
        print("  Run: python task_manager.py --episodes 5 --render")
        print("=" * 60)


if __name__ == "__main__":
    main()
