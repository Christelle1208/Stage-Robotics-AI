"""Training script for SO-100 PPO skills.

Usage:
    python Train.py --task PICK          # train pick policy (v1/v2 reward)
    python Train.py --task PLACE         # train place policy
    python Train.py --task BOTH          # train pick then place
    python Train.py --task REACH         # v3 stage 1
    python Train.py --task GRASP         # v3 stage 2
    python Train.py --task CARRY         # v3 stage 3
    python Train.py --task ALL_V3        # train all 3 v3 stages sequentially

Saved under models/ with auto-incremented version + reward tag.
"""

import os
import argparse
import glob
import yaml
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from Env import PickEnv, PlaceEnv, ReachEnv, GraspEnv, CarryEnv
from Models import PickExtractor, PlaceExtractor, ReachExtractor, GraspExtractor, CarryExtractor


def _make_env(env_cls):
    """Returns a factory function that creates one unwrapped env instance."""
    def _init():
        return env_cls()
    return _init


def _make_vec(env_cls, n_envs, normalize_reward=True):
    venv = VecMonitor(DummyVecEnv([_make_env(env_cls) for _ in range(n_envs)]))
    if normalize_reward:
        # norm_obs=False: observations are already meaningful physically
        # norm_reward=True: keeps reward in ~[-5, 5], critical for PPO stability
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return venv


def _next_model_path(task: str) -> str:
    """Return the next auto-versioned save path, e.g. models/so100_pick_v2_r-v2.zip"""
    reward_tag = config["env"].get("reward_version", "v1")
    prefix = f"models/so100_{task.lower()}_"
    existing = glob.glob(f"{prefix}v*.zip")
    if existing:
        nums = []
        for p in existing:
            name = os.path.basename(p).replace(".zip", "")
            # extract the vN part (first token after task name)
            parts = name.split("_")
            for part in parts:
                if part.startswith("v") and part[1:].isdigit():
                    nums.append(int(part[1:]))
                    break
        next_v = max(nums) + 1 if nums else 1
    else:
        next_v = 1
    return f"{prefix}v{next_v}_r-{reward_tag}"

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------------------
def train_pick(resume=None):
    print("\n" + "="*60)
    print("  Training PICK policy" + (f" (resuming from {resume})" if resume else ""))
    print("="*60)

    if resume:
        # Build a RAW venv (no VecNormalize) so VecNormalize.load wraps it once,
        # avoiding the double-wrap that breaks sync_envs_normalization.
        _raw = VecMonitor(DummyVecEnv([_make_env(PickEnv)
                                       for _ in range(config["learning"]["n_envs"])]))
        model = PPO.load(resume, env=_raw, tensorboard_log="./tb_logs/pick")
        vecnorm_path = "models/so100_pick_vecnorm.pkl"
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, _raw)
            print(f"  Loaded VecNormalize stats from {vecnorm_path}")
        else:
            env = VecNormalize(_raw, norm_obs=False, norm_reward=True, clip_reward=10.0)
        model.set_env(env)
        print(f"  Resuming at timestep {model.num_timesteps:,}")
    else:
        env = _make_vec(PickEnv, n_envs=config["learning"]["n_envs"])
        policy_kwargs = dict(
            features_extractor_class=PickExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=config["learning"]["learning_rate"],
            batch_size=config["learning"]["batch_size"],
            n_steps=config["learning"]["n_steps"],
            ent_coef=config["learning"]["ent_coef"],
            gamma=config["learning"]["gamma"],
            gae_lambda=config["learning"]["gae_lambda"],
            clip_range=config["learning"]["clip_range"],
            verbose=1,
            tensorboard_log="./tb_logs/pick",
        )

    # Eval env must be VecNormalize-wrapped with training=False so EvalCallback
    # can sync normalisation stats without errors.
    _raw_eval = VecMonitor(DummyVecEnv([_make_env(PickEnv)]))
    eval_env = VecNormalize(_raw_eval, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, training=False)
    callbacks = [
        EvalCallback(eval_env, best_model_save_path="models/best_pick",
                     eval_freq=10_000, n_eval_episodes=10, verbose=1),
        CheckpointCallback(save_freq=50_000, save_path="models/ckpt_pick",
                           name_prefix="pick"),
    ]

    model.learn(total_timesteps=config["learning"]["total_timesteps"],
                callback=callbacks, progress_bar=True)
    save_path = _next_model_path("pick")
    model.save(save_path)
    env.save("models/so100_pick_vecnorm.pkl")  # save reward normalizer stats
    print(f"PICK model saved → {save_path}.zip")
    env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
def train_place(resume=None):
    print("\n" + "="*60)
    print("  Training PLACE policy" + (f" (resuming from {resume})" if resume else ""))
    print("="*60)

    if resume:
        # Build raw venv (no VecNormalize) to avoid double-wrapping on load
        _raw = VecMonitor(DummyVecEnv([_make_env(PlaceEnv)
                                       for _ in range(config["learning"]["n_envs"])]))
        model = PPO.load(resume, env=_raw, tensorboard_log="./tb_logs/place")
        vecnorm_path = "models/so100_place_vecnorm.pkl"
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, _raw)
            print(f"  Loaded VecNormalize stats from {vecnorm_path}")
        else:
            env = VecNormalize(_raw, norm_obs=False, norm_reward=True, clip_reward=10.0)
        model.set_env(env)
        print(f"  Resuming at timestep {model.num_timesteps:,}")
    else:
        env = _make_vec(PlaceEnv, n_envs=config["learning"]["n_envs"])
        policy_kwargs = dict(
            features_extractor_class=PlaceExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        )
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=config["learning"]["learning_rate"],
            batch_size=config["learning"]["batch_size"],
            n_steps=config["learning"]["n_steps"],
            ent_coef=config["learning"]["ent_coef"],
            gamma=config["learning"]["gamma"],
            gae_lambda=config["learning"]["gae_lambda"],
            clip_range=config["learning"]["clip_range"],
            verbose=1,
            tensorboard_log="./tb_logs/place",
        )

    # Explicit VecNormalize(training=False) so EvalCallback sync always works
    _raw_eval = VecMonitor(DummyVecEnv([_make_env(PlaceEnv)]))
    eval_env = VecNormalize(_raw_eval, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, training=False)
    callbacks = [
        EvalCallback(eval_env, best_model_save_path="models/best_place",
                     eval_freq=10_000, n_eval_episodes=10, verbose=1),
        CheckpointCallback(save_freq=50_000, save_path="models/ckpt_place",
                           name_prefix="place"),
    ]

    model.learn(total_timesteps=config["learning"]["total_timesteps"],
                callback=callbacks, progress_bar=True)
    save_path = _next_model_path("place")
    model.save(save_path)
    env.save("models/so100_place_vecnorm.pkl")  # save reward normalizer stats
    print(f"PLACE model saved → {save_path}.zip")
    env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# V3 training helpers
# ---------------------------------------------------------------------------
def _make_v3_model(env_cls, extractor_cls, tb_log):
    env     = _make_vec(env_cls, n_envs=config["learning"]["n_envs"])
    policy_kwargs = dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config["learning"]["learning_rate"],
        batch_size=config["learning"]["batch_size"],
        n_steps=config["learning"]["n_steps"],
        ent_coef=config["learning"]["ent_coef"],
        gamma=config["learning"]["gamma"],
        gae_lambda=config["learning"]["gae_lambda"],
        clip_range=config["learning"]["clip_range"],
        verbose=1,
        tensorboard_log=tb_log,
    ), env


def _train_v3_stage(task_name, env_cls, extractor_cls, resume=None):
    print("\n" + "="*60)
    print(f"  Training V3-{task_name}" + (f" (resuming from {resume})" if resume else ""))
    print("="*60)

    tb_log = f"./tb_logs/{task_name.lower()}"
    if resume:
        _raw = VecMonitor(DummyVecEnv([_make_env(env_cls) for _ in range(config["learning"]["n_envs"])]))
        model = PPO.load(resume, env=_raw, tensorboard_log=tb_log)
        vecnorm_path = f"models/so100_{task_name.lower()}_vecnorm.pkl"
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, _raw)
        else:
            env = VecNormalize(_raw, norm_obs=False, norm_reward=True, clip_reward=10.0)
        model.set_env(env)
    else:
        model, env = _make_v3_model(env_cls, extractor_cls, tb_log)

    _raw_eval = VecMonitor(DummyVecEnv([_make_env(env_cls)]))
    eval_env  = VecNormalize(_raw_eval, norm_obs=False, norm_reward=True,
                             clip_reward=10.0, training=False)
    save_dir  = f"models/best_{task_name.lower()}"
    ckpt_dir  = f"models/ckpt_{task_name.lower()}"
    callbacks = [
        EvalCallback(eval_env, best_model_save_path=save_dir,
                     eval_freq=10_000, n_eval_episodes=10, verbose=1),
        CheckpointCallback(save_freq=50_000, save_path=ckpt_dir,
                           name_prefix=task_name.lower()),
    ]
    model.learn(total_timesteps=config["learning"]["total_timesteps"],
                callback=callbacks, progress_bar=True)
    save_path = _next_model_path(task_name.lower())
    model.save(save_path)
    env.save(f"models/so100_{task_name.lower()}_vecnorm.pkl")
    print(f"{task_name} model saved → {save_path}.zip")
    env.close()
    eval_env.close()


def train_reach(resume=None):
    _train_v3_stage("reach", ReachEnv, ReachExtractor, resume)


def train_grasp(resume=None):
    _train_v3_stage("grasp", GraspEnv, GraspExtractor, resume)


def train_carry(resume=None):
    _train_v3_stage("carry", CarryEnv, CarryExtractor, resume)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["PICK", "PLACE", "BOTH", "REACH", "GRASP", "CARRY", "ALL_V3"],
        default="BOTH", help="Which skill to train"
    )
    parser.add_argument("--resume", default=None, metavar="MODEL.zip",
                        help="Path to a saved model (.zip) to continue training from")
    args = parser.parse_args()

    if args.task in ("PICK", "BOTH"):
        train_pick(resume=args.resume if args.task == "PICK" else None)

    if args.task in ("PLACE", "BOTH"):
        train_place(resume=args.resume if args.task == "PLACE" else None)

    if args.task in ("REACH", "ALL_V3"):
        train_reach(resume=args.resume if args.task == "REACH" else None)

    if args.task in ("GRASP", "ALL_V3"):
        train_grasp(resume=args.resume if args.task == "GRASP" else None)

    if args.task in ("CARRY", "ALL_V3"):
        train_carry(resume=args.resume if args.task == "CARRY" else None)
