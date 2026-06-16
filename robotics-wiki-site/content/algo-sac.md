# Algorithm: SAC — Soft Actor-Critic

**Paper:** "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" — Haarnoja, Zhou, Abbeel, Levine (UC Berkeley, 2018); "Soft Actor-Critic Algorithms and Applications" (SAC v2, 2019)
**Category:** Off-policy, model-free, maximum-entropy actor-critic (continuous actions)
**Related:** [[reinforcement-learning]], [[algo-sql-sac]], [[algo-ppo]], [[algo-cql]], [[algo-dqn]]

> **Note:** SAC evolved from SQL (Soft Q-Learning). For the SQL formulation and composable policies, see [[algo-sql-sac]]. This page covers SAC as a practical algorithm used across the robotics papers in this wiki.

---

## The Problem SAC Solves

**DDPG (Deep Deterministic Policy Gradient)** — the leading off-policy continuous-action RL algorithm before SAC — had three critical problems:
1. **Brittle exploration:** Adds fixed Gaussian noise to the policy; doesn't adapt to the difficulty of the task
2. **Q-value overestimation:** A single Q-network overestimates Q-values for underexplored actions, causing the policy to exploit phantom good actions
3. **Hyperparameter sensitivity:** Learning rate, exploration noise, and target network update rate all require careful per-task tuning

SAC fixes all three via **maximum-entropy RL**: the policy explicitly maximizes entropy alongside reward, creating robust and natural exploration, and the twin-critic trick eliminates overestimation.

---

## How It Works

### Maximum-Entropy Objective

SAC maximizes:

```
J(π) = Σ_t E_{(sₜ,aₜ)~π} [r(sₜ, aₜ) + α · H(π(·|sₜ))]
```

Where `H(π(·|s)) = -E_{a~π}[log π(a|s)]` is the policy entropy and `α` is the **temperature** controlling how much entropy is rewarded.

**Why entropy maximization helps:**
- A high-entropy policy explores broadly — it tries many different actions
- This exploration is *shaped* by the reward: actions with higher Q-values still get more probability mass, but all feasible actions retain non-zero probability
- The policy naturally maintains multiple modes (doesn't collapse to one solution)
- Higher α = more exploration; lower α = more exploitation

### Architecture: Three Networks

SAC maintains five networks in total (three trainable + two frozen targets):

```
Actor (policy):     π_θ(a|s) — outputs Gaussian μ(s), σ(s); sample via reparameterization
Critic 1:           Q_φ1(s, a) — estimates Q-values
Critic 2:           Q_φ2(s, a) — twin critic for overestimation reduction
Target Critic 1:    Q_φ1'      — slowly updated copy of Critic 1
Target Critic 2:    Q_φ2'      — slowly updated copy of Critic 2
```

The **twin critics** (Q_φ1, Q_φ2) are independently initialized and trained. Using `min(Q_φ1, Q_φ2)` as the Q-value estimate provides a pessimistic lower bound — if one critic overestimates, the other pulls the estimate down.

### Training: Three Objectives

**Critic loss (minimize Bellman error, for each critic i=1,2):**

```
L_Qi(φᵢ) = E_{(s,a,r,s')~D} [(Qφᵢ(s,a) - y)²]

y = r + γ · (min(Q_φ1'(s',ã'), Q_φ2'(s',ã')) - α · log π_θ(ã'|s'))
ã' ~ π_θ(·|s')
```

The target `y` uses:
- `min` of twin target critics → conservative Q-estimate
- `- α · log π_θ(ã')` → entropy reward for the next state

**Actor loss (maximize Q-value minus entropy penalty):**

```
L_π(θ) = E_{s~D, ã~π_θ} [α · log π_θ(ã|s) - min(Q_φ1(s,ã), Q_φ2(s,ã))]
```

This pushes the actor toward actions with high Q-values while maintaining entropy. The gradients flow through the reparameterization trick: `ã = μ_θ(s) + σ_θ(s) · ε`, `ε ~ N(0,1)`.

**Temperature loss (SAC v2 — automatic entropy tuning):**

```
L_α = E_{s~D, ã~π_θ} [-α · (log π_θ(ã|s) + H̄)]
```

Where `H̄` is the **target entropy** (a hyperparameter, typically set to `-dim(action_space)`). This automatically:
- Increases α if the policy is too deterministic (entropy < target)
- Decreases α if the policy is too random (entropy > target)

No manual temperature tuning needed.

### Full Training Loop

```
Initialize replay buffer D, networks π_θ, Q_φ1, Q_φ2, Q_φ1', Q_φ2', log α

For each training step:
  1. Sample action: ã = μ_θ(s) + σ_θ(s)·ε  [reparameterization]
  2. Execute ã, observe (s, ã, r, s')
  3. Store (s, ã, r, s') in D
  4. Sample mini-batch B ~ D
  5. Update critics: minimize L_Q1(φ1), L_Q2(φ2) on B
  6. Update actor: minimize L_π(θ) on B
  7. Update temperature: minimize L_α on B
  8. Soft update targets: φ1' ← τ·φ1 + (1-τ)·φ1'  [τ ≈ 0.005]
                          φ2' ← τ·φ2 + (1-τ)·φ2'
```

### Action Space: Squashed Gaussian

SAC's actor outputs a **Squashed Gaussian** policy. The raw Gaussian sample `u ~ N(μ, σ²)` is passed through tanh:

```
a = tanh(u) ∈ (-1, 1)^d
```

This bounds actions to a finite range (required for real robots with joint limits). The log-probability includes the Jacobian correction:

```
log π(a|s) = log N(u; μ, σ²) - Σᵢ log(1 - tanh(uᵢ)²)
```

---

## Why SAC Works Better Than DDPG / PPO

### vs. DDPG

| Problem | DDPG | SAC |
|---------|------|-----|
| Q-overestimation | Single Q-network → overestimates | Twin critics → pessimistic estimate |
| Exploration | Fixed Gaussian noise | Entropy maximization → adaptive |
| Hyperparameter sensitivity | High (noise σ, learning rates) | Low (auto-α, robust to LR) |
| Stability | Unstable in practice | Reliable across tasks |

### vs. PPO

| Property | PPO | SAC |
|----------|-----|-----|
| Sample efficiency | Low (on-policy; no replay) | High (off-policy; replay buffer) |
| Multimodal behavior | No (deterministic-ish) | Yes (stochastic policy) |
| Offline data use | No | Yes (store in replay buffer) |
| Best for | Fast iteration, simple tasks | Complex manipulation, high DoF |

---

## Evaluation

### Original SAC Paper Benchmarks (MuJoCo continuous control)

| Environment | SAC | TD3 | DDPG | PPO |
|-------------|-----|-----|------|-----|
| HalfCheetah-v2 | **16,700** | 9,600 | 8,600 | 1,800 |
| Ant-v2 | **5,400** | 4,400 | 900 | 2,900 |
| Humanoid-v2 | **5,100** | 5,300 | 300 | 2,100 |
| Walker2d-v2 | **5,000** | 4,700 | 3,000 | 3,100 |

*(Average return after 3M steps; SAC best or tied on 3/4)*

SAC achieves comparable or better performance with significantly fewer environment interactions than on-policy methods.

### SAC in Robotics Papers (this wiki)

**Task Decomposition Reward RL for P&P (2023, MuJoCo/Robosuite):**
- 3-subtask decomposition (approach, grasp, place)
- SAC per subtask with axis-based reward shaping
- **93.2% average success rate** across 4 trials
- See [[pick-and-place]]

**CQL/SAC for Human-Robot Collaboration (2025):**
- SAC online + CQL offline regularization
- P&P in HRC environments
- **100% simulation success, 80% real-hardware success**
- See [[algo-cql]]

**Simulated + Real P&P with Franka Panda (2023):**
- SAC + traditional controls on Franka Panda
- Trains in simulation, transfers to real hardware via ROS
- See [[simulation-and-tools]]

**Vision-Based Robotic Grasping with YOLO (2023):**
- SAC + YOLO for 6-DOF industrial grasping
- Handles small-volume, large-variety production environments
- See [[grasping-and-manipulation]]

---

## Key Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| Replay buffer size | 10⁶ | Larger = more data diversity |
| Batch size | 256 | Larger = more stable gradients |
| Learning rate | 3e-4 | Same for actor, critics, α |
| Target entropy H̄ | `-dim(A)` | Default: negative of action dimensionality |
| τ (target update) | 0.005 | Slow soft update; stability |
| Discount γ | 0.99 | Standard |
| Initial α | 1.0 | Auto-tuned; initial value matters less |

---

## Pros

- **Sample efficient** — replay buffer reuses all past experience; 3-10× fewer env steps than PPO on same tasks
- **Stable** — twin critics + auto-α + target networks; rarely diverges in practice
- **Natural exploration** — entropy maximization; no noise schedule to tune
- **Handles high-DoF** — designed for continuous, high-dimensional action spaces (7-DoF arms, dexterous hands)
- **Extensible** — can incorporate CQL for offline data, HER for sparse rewards, demonstrations for bootstrapping

## Cons

- **Continuous actions only** — not directly applicable to discrete action spaces (though discrete SAC exists)
- **Image input is harder** — raw pixel SAC requires additional tricks (data augmentation, DrQ, CURL) to be stable; tabular/state-based SAC is more robust
- **Replay memory** — large buffers (1M transitions) need significant RAM (>16GB for image obs)
- **Not for offline RL** — standard SAC needs online interaction; use [[algo-cql]] for purely offline settings
- **Slower per step than PPO** — multiple network forward/backward passes per step; compensated by higher sample efficiency

---

## Variants Used in This Wiki

| Variant | Modification | Paper |
|---------|-------------|-------|
| SAC + task decomposition | Per-subtask SAC with shaped rewards | Task Decomp Reward P&P |
| SAC + CQL | CQL regularization on critic for offline data | CQL/SAC HRC |
| SAC + traditional controls | SAC policy + classical controller for safety | Sim+Real P&P |
| SAC + YOLO | SAC with visual object detection | Vision-Based Grasping |

---

## In This Wiki

SAC is central to: [[pick-and-place]], [[grasping-and-manipulation]], [[reinforcement-learning]].
For the SQL (Soft Q-Learning) formulation and composability: [[algo-sql-sac]].
For offline extension: [[algo-cql]].
For hybrid IL+RL using SAC: [[algo-awac]], [[algo-hitl-rl]].
