# Partie Simulation Workspace

This workspace has been reorganized into a clean, multi-project layout.

## Folder map

- `projects/`
  - `so100-mujoco-rl/`: main SO-100 MuJoCo RL project (training, sim-to-real tools)
  - `clean-pickdrop/`: separate pick-and-drop experiment project
- `third_party/`
  - `mujoco_menagerie/`: fresh clone of MuJoCo Menagerie
  - `lerobot/`: fresh clone of Hugging Face LeRobot
- `data/`
  - `outputs/`: training runs, checkpoints, models, tensorboard logs, videos
- `docs/`
  - global docs and workspace-level notes
- `archive/`
  - backups of previous in-place assets (kept for safety)

## Quick start

1. Go to the main project:

```bash
cd "projects/so100-mujoco-rl"
```

2. Set up Python environment and dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

3. Check scene/env:

```bash
python scripts/check_env.py --env SO100Grab-v0 --steps 10
mjpython scripts/view_scene.py --scene assets/mujoco_menagerie/trs_so_arm100/scene.xml
```

4. Train PPO for cube-reaching-with-open-gripper:

```bash
python scripts/train_ppo_so100_grab.py
```

## Notes

- Main project now lives in `projects/so100-mujoco-rl`.
- Existing Menagerie path compatibility is preserved through:
  - `projects/so100-mujoco-rl/assets/mujoco_menagerie -> ../../../third_party/mujoco_menagerie`
- Historical artifacts were not deleted; they were moved into structured folders.
