# SO-100 Pick-and-Drop RL — Quick Reference

## Project structure

```
clean_pickdrop/
├── env.py          environment definition
├── train.py        train PPO or SAC
├── enjoy.py        watch a trained model in action
├── compare.py      generate PPO-vs-SAC comparison plots
├── view.py         inspect the raw env (no trained policy needed)
└── runs/           all training artifacts (created on first run)
    ├── <run-name>/
    │   ├── final_model.zip          weights at end of training
    │   ├── replay_buffer.pkl        SAC only — replay buffer for continuation
    │   ├── best_model/best_model.zip  best checkpoint (highest eval reward)
    │   └── eval/evaluations.npz     timesteps / rewards / success rate log
    └── tensorboard/                 TensorBoard event files
```

---

## env.py — the environment

Key constants you may want to edit:

| Constant | Default | What it controls |
|---|---|---|
| `HOME_QPOS` | `[0,-3.3,3.14,1.2,1.5,-0.17]` | Robot starting pose (radians, joints: Rotation Pitch Elbow Wrist_Pitch Wrist_Roll Jaw) |
| `CUBE_SPAWN_X` | `(-0.12, 0.08)` | Cube x spawn range (m) |
| `CUBE_SPAWN_Y` | `(-0.34, -0.18)` | Cube y spawn range (m) |
| `BIN_POS` | `(0.17, -0.09, 0.01)` | Fixed bin position (m) |
| `PLACE_TARGET` | `BIN_POS + (0,0,0.09)` | End-effector hover goal during place phase |
| `GRASP_DIST` | `0.045` | EE–cube distance threshold to detect grasp (m) |
| `GRASP_LIFT` | `0.05` | Cube must rise this much above rest to confirm grasp (m) |
| `GRASP_BONUS` | `5.0` | Sparse reward on grasp detection |
| `SUCCESS_BONUS` | `50.0` | Sparse reward on cube landing in bin |

Reward formula (from the paper, implemented exactly):

```
r = r_pos + r_energy
r_pos = Kx*(x_ee-x_g)² + Ky*(y_ee-y_g)² + Kz*(z_ee-z_g)²

While reaching:  Kx=Ky=-2,  Kz=-0.5   (goal = cube position)
While placing:   Kx=Ky=Kz=-1           (goal = PLACE_TARGET above bin)

r_energy = -3e-5 * Σ|F_i|  over all actuator forces
```

---

## Commands

### Train from scratch

```bash
python -m clean_pickdrop.train --algo ppo --timesteps 1_000_000
python -m clean_pickdrop.train --algo sac --timesteps 1_000_000
```

Default run names: `ppo_seed0`, `sac_seed0`.  
Use `--run-name my_name` to give the run a custom name.

### Continue training

```bash
python -m clean_pickdrop.train --algo ppo --run-name ppo_seed0 --timesteps 500_000 --continue
python -m clean_pickdrop.train --algo sac --run-name sac_seed0 --timesteps 500_000 --continue
```

`--timesteps` is always **additional** steps on top of what was already done.  
SAC automatically restores its replay buffer from `replay_buffer.pkl`.

### Useful training flags

| Flag | Default | Description |
|---|---|---|
| `--timesteps` | 1 000 000 | Steps to train (additional if `--continue`) |
| `--n-envs` | 8 (PPO) / 1 (SAC) | Parallel training environments |
| `--eval-freq` | 10 000 | Steps between evaluations |
| `--eval-episodes` | 20 | Episodes per evaluation |
| `--run-name` | `{algo}_seed{seed}` | Name of the run directory |
| `--seed` | 0 | Random seed |
| `--continue` | off | Resume from `final_model.zip` |
| `--target-success` | none | Stop once eval success rate ≥ this value (e.g. `0.5`) |
| `--max-timesteps` | 10 000 000 | Hard cap when `--target-success` is set |

### Watch / evaluate a trained model

Episodes use **fresh random cube placements every run** by default (no fixed seed),
so success rate reflects generalization, not memorized layouts. Pass `--seed` for
a reproducible-but-still-varied sequence.

```bash
# Stats — headless, N random episodes, aggregate success rate / return
python -m clean_pickdrop.enjoy --run ppo_seed0 sac_seed0 --mode stats --episodes 100

# Video — one model, 5 episodes → runs/ppo_seed0_eval.mp4
python -m clean_pickdrop.enjoy --run ppo_seed0 --episodes 5

# Video — PPO vs SAC side-by-side in one file
python -m clean_pickdrop.enjoy --run ppo_seed0 sac_seed0 --episodes 5

# Interactive 3D viewer (macOS: must use mjpython)
mjpython -m clean_pickdrop.enjoy --run ppo_seed0 --mode interactive

# Interactive viewer at half speed (--speed 1.0 = real-time, default)
mjpython -m clean_pickdrop.enjoy --run sac_seed0 --mode interactive --speed 0.5

# Load final_model instead of best_model
python -m clean_pickdrop.enjoy --run ppo_seed0 --final

# Reproducible episode sequence (same cube placements every run)
python -m clean_pickdrop.enjoy --run ppo_seed0 --mode stats --episodes 100 --seed 42
```

### Compare PPO vs SAC (plots)

```bash
python -m clean_pickdrop.compare --ppo-run ppo_seed0 --sac-run sac_seed0
# → runs/ppo_vs_sac.png  (success rate, mean return, loss curves)
```

### Inspect the env without a trained policy

```bash
# Random-action rollout → clean_pickdrop/rollout.mp4
python -m clean_pickdrop.view --mode video --episodes 3

# Print reward decomposition every 20 steps (verify formula)
python -m clean_pickdrop.view --mode rewards --episodes 1

# Interactive viewer with random actions (macOS: mjpython)
mjpython -m clean_pickdrop.view --mode interactive
```

### TensorBoard

```bash
tensorboard --logdir "/Users/christelle.nollet/Desktop/Partie Simulation/clean_pickdrop/runs/tensorboard"
```

Key tags logged automatically by SB3:

| Tag | Meaning |
|---|---|
| `eval/success_rate` | Fraction of eval episodes where cube landed in bin |
| `eval/mean_reward` | Mean cumulative return over eval episodes |
| `rollout/ep_rew_mean` | Mean training episode return |
| `train/policy_gradient_loss` | PPO policy loss |
| `train/value_loss` | PPO value function loss |
| `train/entropy_loss` | PPO entropy loss |
| `train/actor_loss` | SAC actor loss |
| `train/critic_loss` | SAC critic loss |
| `train/ent_coef_loss` | SAC entropy coefficient loss |
