# SAC Three-Agent Pick-and-Place For SO-100

> **Paper replication**: *"Bionic Pick-and-Place Using Three Hierarchical SAC Agents in MuJoCo"*  
> MDPI Biomimetics 8(2):240 — https://www.mdpi.com/2313-7673/8/2/240

---

## Overview

This project implements a three-agent hierarchical reinforcement learning system for the SO-100 6-DoF robotic arm. Each agent is trained independently with **Soft Actor-Critic (SAC)** and specialises in one sub-task of a pick-and-place sequence.

```
HOME POSE
   │
   ▼  Agent 1 — REACH
   │  "Move end-effector to pre-grasp position above the cube"
   │  Obs: 19-dim  |  Max steps: 600
   │
   ▼  Agent 2 — GRASP
   │  "Close gripper around the cube and lift it above the table"
   │  Obs: 19-dim  |  Max steps: 400
   │
   ▼  Agent 3 — PLACE
      "Transport the held cube to the goal zone and release it"
      Obs: 22-dim  |  Max steps: 600
```

**Key differences from two-agent PPO approaches:**
| Aspect | Two-agent PPO | Three-agent SAC (this project) |
|---|---|---|
| Decomposition | Pick + Place | Reach + Grasp + Place |
| Algorithm | PPO (on-policy) | SAC (off-policy) |
| Sample efficiency | Needs parallel envs | Single env, replay buffer |
| Exploration | Entropy bonus in loss | Auto-tuned temperature |
| Environment | MuJoCo only | MuJoCo + Robosuite support |

---

## Project Structure

```
SAC_THREE_AGENTS/
├── config.yaml                  # All hyperparameters (edit here, not in code)
├── train.py                     # Train one or all three SAC agents
├── task_manager.py              # Chain three agents at inference time
├── evaluate.py                  # Evaluate individual agents or full pipeline
│
├── envs/                        # Primary MuJoCo + Gymnasium environments
│   ├── base_env.py              # SO-100 base: MuJoCo loading, state injection
│   ├── reach_env.py             # Agent 1 — ReachEnv
│   ├── grasp_env.py             # Agent 2 — GraspEnv
│   └── place_env.py             # Agent 3 — PlaceEnv
│
├── robosuite_envs/              # Optional: Robosuite-backed environments
│   ├── so100_robot.py           # Custom SO-100 robot class for Robosuite
│   ├── reach_task.py            # Robosuite Reach task (extends Lift)
│   ├── grasp_task.py            # Robosuite Grasp task (extends Lift)
│   ├── place_task.py            # Robosuite Place task (extends PickPlace)
│   └── gym_wrappers.py          # Gym-compatible wrappers for SB3
│
├── assets/                      # Robot XML + mesh assets
│   ├── so_arm100.xml
│   ├── scene_pick_place.xml
│   └── *.stl  (mesh files)
│
├── models/                      # Saved SAC model checkpoints (created at training)
└── tb_logs/                     # TensorBoard training logs
```

---

## Quick Start

### 1 — Install dependencies
```bash
cd /path/to/SAC_THREE_AGENTS
pip install -r requirements.txt
```

### 2 — Train all three agents (sequential, with curriculum handoff)
```bash
python train.py --task all
```

Or train individually:
```bash
python train.py --task reach          # ~600k steps
python train.py --task grasp          # ~800k steps (auto-loads Reach for curriculum)
python train.py --task place          # ~600k steps (auto-loads Grasp for curriculum)
```

### 3 — Run the full pick-and-place pipeline
```bash
# macOS (MuJoCo viewer requires mjpython)
mjpython task_manager.py --episodes 5 --render

# Headless
python task_manager.py --episodes 20
```

### 4 — Evaluate
```bash
python evaluate.py --mode full --episodes 100 --out results.json
```

### 5 — Monitor training with TensorBoard
```bash
tensorboard --logdir tb_logs
```

---

## SAC Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 3e-4 | Standard Adam rate |
| `buffer_size` | 300 000 | Off-policy replay buffer |
| `batch_size` | 256 | Mini-batch for gradient |
| `tau` | 0.005 | Soft Polyak target update |
| `gamma` | 0.99 | Discount, long-horizon tasks |
| `gradient_steps` | 4 | Updates per env step → more sample efficient |
| `learning_starts` | 5 000 | Random exploration phase |
| `ent_coef` | auto | SAC adaptive temperature |
| `net_arch` | [256, 256] | Actor + Critic hidden layers |

---

## Reward Design

### Agent 1 — Reach
```
r = - dist_weight × dist(EE, pre_grasp)           (dense distance)
  + progress_bonus × max(0, prev_dist - dist)      (approach reward)
  + gripper_open_bonus × jaw_openness              (keep gripper open)
  + success_bonus  [sparse, once]
  - step_penalty
```

### Agent 2 — Grasp
```
r = - dist_weight × dist(EE, object)              (dense approach)
  + grasp_bonus × centering × closedness          [only at new height]
  + jaw_nudge × closedness                        [if dist < 4cm]
  + lift_weight × Δheight                         (personal-best lift)
  + lift_bonus  [sparse, once]
  + collision_penalty  [if arm hits table]
  - step_penalty
```

Personal-best height: the agent only earns rewards for reaching **new** height records. Lowering and re-lifting scores zero. This prevents oscillation exploitation.

### Agent 3 — Place
```
r = γ × Φ(s') - Φ(s)   where Φ(s) = -goal_dist_weight × dist(obj, goal)
  - dist_dense_weight × dist(obj, goal)  [while holding]
  + drop_penalty  [per-step when object not held]
  + release_bonus  [sparse: released within threshold]
```

Potential-based shaping (Ng et al. 1999) guarantees the reward does not change the optimal policy — critical for safe agent chaining.

---

## Robosuite Backend

Robosuite is an optional backend. When installed, the `robosuite_envs/` module provides environments that run on standard Robosuite tasks with Panda robot.

```bash
pip install robosuite
python train.py --task all --backend robosuite
```

**Adding SO-100 to Robosuite:**  
See `robosuite_envs/so100_robot.py` for the `SO100` and `SO100Gripper` classes. The main step is adapting `assets/so_arm100.xml` to Robosuite's asset conventions (site names, actuator tags).

---

## State Transfer Between Agents

```python
# At the end of Phase 1 (Reach succeeds):
qpos, qctrl = reach_env.get_state_snapshot()

# Phase 2 starts exactly where Phase 1 ended:
obs, _ = grasp_env.reset_from_reach(qpos, qctrl)

# Phase 3 likewise:
qpos, qctrl = grasp_env.get_state_snapshot()
obs, _ = place_env.reset_from_grasp(qpos, qctrl)
```

The snapshot contains the full MuJoCo `qpos` and `ctrl` vectors — no information is lost between transitions.

---

## Hardware Transfer (SO-100 Physical Robot)

The environments use delta joint-angle control at 50 Hz, matching the SO-100 physical controller. To deploy:

1. Replace the MuJoCo step with your SO-100 hardware interface
2. Map the 6-dim action to motor commands: `delta_rad = action * delta_scale`
3. Ensure latency ≤ 20 ms (one `frame_skip=5` step at 4 ms timestep)

---

## Citation

```bibtex
@article{biomimetics8020240,
  title   = {Pick-and-Place Using Three Hierarchical SAC Agents},
  journal = {Biomimetics},
  volume  = {8},
  number  = {2},
  pages   = {240},
  year    = {2023},
  doi     = {10.3390/biomimetics8020240}
}
```
