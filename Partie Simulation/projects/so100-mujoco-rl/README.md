# SO-100 MuJoCo RL Workspace

A clean, modular repository for training reinforcement-learning policies on the
[SO-100 robot arm](https://github.com/TheRobotStudio/SO-ARM100) inside
[MuJoCo](https://mujoco.org/), using
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO & SAC) and
assets from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

---

## Repository layout

```
.
├── configs/
│   ├── robot/so100.yaml              # Joint/actuator names, EE site
│   ├── env/so100_reach.yaml          # Reaching task config
│   ├── env/so100_pick_place.yaml     # Pick-and-place task config
│   ├── train/ppo_so100_reach.yaml
│   ├── train/ppo_so100_pick_place.yaml
│   └── train/sac_so100_pick_place.yaml
├── assets/
│   ├── mujoco_menagerie/             # ← clone/symlink Menagerie here
│   └── robots/so100/                 # extra robot assets if needed
├── mujoco/
│   ├── scenes/                       # Scene XMLs (include robot + objects)
│   └── objects/                      # Reusable object XMLs (cube, etc.)
├── src/so100_mujoco_rl/
│   ├── envs/                         # Gymnasium environments
│   ├── robots/                       # Robot descriptor (SO100Robot)
│   ├── tasks/                        # Reward / termination logic
│   ├── wrappers/                     # SB3 VecEnv helpers
│   ├── utils/                        # Config loader, MuJoCo utilities
│   └── train/train_sb3.py            # Generic PPO/SAC trainer
├── scripts/                          # CLI entry-points
├── tests/                            # pytest test suite
└── outputs/                          # Models, logs, videos (git-ignored)
```

---

## Prerequisites

- Python 3.10+
- macOS, Linux (Windows untested)

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -e ".[dev]"
# Or without editable install:
pip install -r requirements.txt
pip install -e .
```

### Optional: LeRobot

```bash
pip install "git+https://github.com/huggingface/lerobot.git"
```

---

## Setting up MuJoCo Menagerie assets

The SO-100 robot arm XML lives in
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

**Clone into the expected location:**

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie assets/mujoco_menagerie
```

**Or symlink an existing checkout:**

```bash
ln -s /path/to/mujoco_menagerie assets/mujoco_menagerie
```

After cloning, the SO-100 XML should be at:
```
assets/mujoco_menagerie/so_arm100/so_arm100.xml
```

> **TODO**: Verify the exact directory name inside the Menagerie repository.
> The expected sub-directory is `so_arm100/`. If it differs, update
> `configs/robot/so100.yaml` → `xml_path`.

### Verify asset names

Once Menagerie is in place, inspect the model to confirm joint/actuator names:

```bash
python scripts/check_env.py --env SO100PickPlace-v0
```

This prints every joint, actuator, and site in the loaded model.
Compare with `configs/robot/so100.yaml` and update as needed.

---

## Switching from placeholder to real Menagerie model

The scene XMLs in `mujoco/scenes/` currently contain a **placeholder arm**
(a simple capsule chain) so the code runs without the Menagerie assets.

To switch to the real SO-100:

1. Ensure `assets/mujoco_menagerie/so_arm100/so_arm100.xml` exists.
2. In each scene XML, **uncomment** the `<include>` line and **delete** the
   placeholder `<body name="so100_base">` block.
3. Run `python scripts/check_env.py` and fix any name mismatches in
   `configs/robot/so100.yaml`.

---

## Quickstart

### 1 — Check the environment

```bash
python scripts/check_env.py --env SO100PickPlace-v0 --steps 10
python scripts/check_env.py --env SO100Reach-v0 --steps 10
```

### 2 — View the scene

```bash
python scripts/view_scene.py --scene mujoco/scenes/so100_pick_place.xml
python scripts/view_scene.py --scene mujoco/scenes/so100_reach.xml
```

### 3 — Train PPO (pick-and-place)

```bash
python scripts/train_ppo_so100_pick_place.py
```

Checkpoints are saved to `outputs/logs/ppo_so100_pick_place/checkpoints/`.
The best model is saved to `outputs/logs/ppo_so100_pick_place/best_model/`.
The final model is saved to `outputs/models/`.

### 4 — Train SAC (pick-and-place)

```bash
python scripts/train_sac_so100_pick_place.py
```

### 5 — Train PPO (reaching, faster to converge)

```bash
python scripts/train_ppo_so100_reach.py
```

### 6 — Monitor training with TensorBoard

```bash
tensorboard --logdir outputs/logs/tensorboard
```

### 7 — Evaluate a trained policy

```bash
# Evaluate a PPO model
python scripts/evaluate_policy.py \
    --model outputs/logs/ppo_so100_pick_place/best_model/best_model.zip \
    --env SO100PickPlace-v0 \
    --algo ppo \
    --episodes 20 \
    --render

# Evaluate a SAC model
python scripts/evaluate_policy.py \
    --model outputs/logs/sac_so100_pick_place/best_model/best_model.zip \
    --env SO100PickPlace-v0 \
    --algo sac \
    --episodes 20 \
    --render
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Environments

| ID | Task | Obs dim | Act dim |
|----|------|---------|---------|
| `SO100Reach-v0` | Move EE to floating target | 16 (no gripper) | 5 |
| `SO100PickPlace-v0` | Pick cube, place on target | 28+ | 6 |

### Observation spaces

**SO100Reach-v0**
- Arm joint positions (×5)
- Arm joint velocities (×5)
- End-effector Cartesian position (×3)
- Target Cartesian position (×3)

**SO100PickPlace-v0**
- Arm joint positions (×5) + velocities (×5)
- Gripper joint position + velocity (×1 each)
- End-effector position (×3)
- Object (cube) position (×3)
- Target position (×3)
- Vector EE → object (×3)
- Vector object → target (×3)

### Reward shaping (pick-and-place)

| Component | Weight | Description |
|-----------|--------|-------------|
| Reaching  | 1.0    | −‖EE − cube‖ |
| Lifting   | 2.0    | +1 when cube is above table + lift_height |
| Placing   | 4.0    | −‖cube − target‖ × lifted |
| Success   | +10    | When ‖cube − target‖ < threshold |
| Action    | −0.01  | −‖action‖² regularisation |

---

## Configuration

All hyperparameters live in YAML files under `configs/`.  No hardcoded values
in Python source.

Key files:

| File | Purpose |
|------|---------|
| `configs/robot/so100.yaml` | Joint names, actuator names, EE site |
| `configs/env/so100_pick_place.yaml` | Scene XML, reward weights, randomisation |
| `configs/train/sac_so100_pick_place.yaml` | SAC hyperparameters |
| `configs/train/ppo_so100_pick_place.yaml` | PPO hyperparameters |

---

## Extending the workspace

### Adding a new task

1. Add a scene XML to `mujoco/scenes/`.
2. Add a task class in `src/so100_mujoco_rl/tasks/`.
3. Add a Gymnasium env in `src/so100_mujoco_rl/envs/`.
4. Register it in `src/so100_mujoco_rl/envs/__init__.py`.
5. Add YAML configs under `configs/env/` and `configs/train/`.
6. Add a training script in `scripts/`.

### Adding imitation learning / LeRobot integration

The task and robot descriptor modules are designed to be reusable outside
Gymnasium.  A LeRobot dataset-recording wrapper can:
- Import `SO100Robot` and task classes directly.
- Use `BaseMuJoCoEnv.render(mode="rgb_array")` for image observations.
- Replace the SB3 training loop with a LeRobot `Dataset` recorder.

### Sim-to-real hooks

Planned extension points:
- `src/so100_mujoco_rl/robots/so100_real.py` — real robot driver matching
  the `SO100Robot` interface.
- `configs/robot/so100_real.yaml` — real joint limits and control modes.
- Policy export via `model.policy.state_dict()` for deployment.

---

## TODOs

- [ ] Verify `so_arm100.xml` actuator/joint names from MuJoCo Menagerie and
      update `configs/robot/so100.yaml`.
- [ ] Confirm `attachment_site` exists in the Menagerie XML (or update
      `end_effector_site` in the config).
- [ ] Switch scene XMLs from the placeholder arm to the real Menagerie include.
- [ ] Add multi-environment vectorisation (`n_envs > 1`) for faster PPO training.
- [ ] Add `VecNormalize` save/load for stable evaluation.
- [ ] Add video recording callback.
- [ ] Add LeRobot dataset recording wrapper.
- [ ] Add teleoperation script.
