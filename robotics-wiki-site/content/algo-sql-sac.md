# Algorithm: SQL and SAC — Soft Q-Learning and Soft Actor-Critic

**Papers:**
- "Composable Deep Reinforcement Learning for Robotic Manipulation" — Haarnoja et al. (2018) → SQL
- SAC: "Soft Actor-Critic" — Haarnoja et al. (2018/2019) → evolved from SQL
**Category:** Maximum-entropy off-policy RL
**Related:** [[reinforcement-learning]], [[algo-ppo]], [[algo-cql]], [[algo-dqn]]

---

## The Core Idea: Maximum-Entropy RL

Standard RL maximizes expected return:

```
J(π) = E[Σ γᵗ r(sₜ, aₜ)]
```

Maximum-entropy RL maximizes return **plus entropy**:

```
J_maxent(π) = E[Σ γᵗ (r(sₜ, aₜ) + α·H(π(·|sₜ))]
```

Where `H(π(·|s)) = -E[log π(a|s)]` is the policy's entropy at state s, and α is the temperature (how much entropy to encourage).

**Intuition:** By maximizing entropy, the agent is incentivized to be as random as possible while still collecting reward. This means:
- Natural exploration without explicit ε-greedy or Gaussian noise
- Learning multiple modes of solving a task (not just one)
- Robustness — the policy doesn't over-commit to a single solution

---

## SQL: Soft Q-Learning

### How It Works

SQL represents the policy as a **Boltzmann (energy-based) distribution** over actions:

```
π(a|s) ∝ exp(Q_soft(s, a) / α)
```

Where `Q_soft` is the *soft* Q-function that accounts for future entropy:

```
Q_soft(s, a) = r(s, a) + γ · E_{s'}[V_soft(s')]
V_soft(s) = α · log ∫ exp(Q_soft(s, a')/α) da'
```

`V_soft` is the **soft value function** — the log-sum-exp of Q-values, which is the entropy-regularized value.

**Training:** Minimize the soft Bellman error using samples from a replay buffer. The policy is implicit — no separate policy network, just sample from the Boltzmann distribution at inference time (requires SVGD or similar for continuous actions).

### Composability: The Key Innovation

The SQL paper's main contribution for robotics is **composable policies**. If you have two policies π₁ and π₂ trained for subtasks (e.g., "reach object" and "grasp object"), you can create a composed policy:

```
π_composed(a|s) ∝ π₁(a|s) · π₂(a|s)
         = exp(Q₁(s,a)/α) · exp(Q₂(s,a)/α)
         = exp((Q₁(s,a) + Q₂(s,a))/α)
```

**This means: Q-functions for different tasks add directly!** The composed policy approximately optimizes both objectives simultaneously — no retraining needed.

---

## SAC: Soft Actor-Critic

SAC evolved SQL into a practical actor-critic algorithm for continuous action spaces. It keeps maximum-entropy RL but replaces the implicit policy with an explicit parameterized policy.

### Architecture

SAC maintains three networks:
1. **Policy network (actor) π_θ:** Outputs a Gaussian distribution over actions `N(μ_θ(s), σ_θ(s)²)`. Actions are sampled and squashed through tanh.
2. **Two Q-networks Q_φ1, Q_φ2 (critics):** Twin critics to prevent Q-value overestimation. Takes (s, a) → Q-value.
3. **Target Q-networks:** Slowly updated copies for stable Bellman targets.

### The Training Objectives

**Critic update:** Minimize soft Bellman error

```
L_Q(φ) = E[(Q_φ(s,a) - y)²]
where y = r + γ·(min(Q_φ1', Q_φ2')(s',ã') - α·log π(ã'|s'))
ã' ~ π(·|s')
```

The `min` of twin critics reduces overestimation. The `- α·log π` term is the future entropy reward.

**Actor update:** Maximize Q-value minus entropy cost

```
L_π(θ) = E[-min(Q_φ1, Q_φ2)(s, ã) + α·log π_θ(ã|s)]
ã ~ π_θ(·|s)
```

**Temperature update (SAC-v2):** α is learned automatically to target a desired entropy level:

```
L_α = E[-α·(log π(a|s) + H_target)]
```

This eliminates the need to tune α manually.

### Training Loop

```
1. Sample action a ~ π_θ(·|s)
2. Execute a, observe (s, a, r, s')
3. Store in replay buffer
4. Sample mini-batch from buffer
5. Update critics (minimize Bellman error)
6. Update actor (maximize Q - entropy)
7. Update α (minimize entropy deviation from target)
8. Soft update target networks: φ_target ← τ·φ + (1-τ)·φ_target
```

---

## Why SAC Works Better Than PPO/DDPG

| Property | PPO | DDPG | SAC |
|----------|-----|------|-----|
| On/off-policy | On | Off | Off |
| Sample efficiency | Low | Medium | High |
| Exploration | Entropy bonus (optional) | Gaussian noise | Entropy maximization (built-in) |
| Stability | Good (clip) | Poor (Q overestimation) | Good (twin critics + entropy) |
| Hyperparameter sensitivity | Medium | High | Low (auto-α) |
| Multimodal policies | No | No | Yes |

SAC's automatic entropy tuning (SAC-v2) is a major practical advantage — the temperature α adjusts itself to maintain the right level of exploration throughout training.

---

## Evaluation

### SQL Evaluation (Composable DRL paper)

**Environment:** MuJoCo tasks with Sawyer robot arm
- **Tasks tested:** Reaching, Lego block stacking, multi-task composition
- **Composability demo:** Train separate policies for "reach position A" and "reach position B", compose them to get a policy that reaches a point avoiding obstacles

**Baselines:** Compared to standard Q-learning, DDPG, standard policy gradients

**Results:**
- SQL trains successfully on Sawyer reaching and stacking
- **Composed policies work without retraining** — quantitative policy composition outperforms independent policies on constrained navigation tasks
- First demonstration that robotic manipulation policies can be arithmetically composed

### SAC Evaluation (various robotics papers in this wiki)

**Used in:**
1. **Task Decomposition Reward RL (2023):** SAC achieves **93.2% average success** on P&P in MuJoCo/Robosuite across 4 trials. Three-subtask decomposition (approach, grasp, place).
2. **CQL/SAC HRC (2025):** SAC + CQL achieves **100% success in simulation, 80% in real hardware** for P&P in human-robot collaboration.
3. **Simulated + Real P&P (2023):** SAC on Franka Panda; transfers to real hardware.
4. **Vision-Based Grasping (2023):** SAC + YOLO for 6-DOF industrial grasping.

---

## Pros (SQL/SAC)

- **Sample efficient** — replay buffer allows data reuse; learns faster than PPO per environment step
- **Natural exploration** — entropy maximization eliminates need for hand-tuned exploration noise
- **Stable training** — twin critics, auto-α, and target networks prevent common DDPG failures
- **Multimodal** — policy maintains probability mass on multiple action modes; doesn't over-specialize
- **SQL composability** — unique ability to combine independently trained policies without retraining
- **Off-policy** — can learn from any past data in the buffer, including demonstrations

## Cons

- **Continuous actions only** (SAC) — requires reparameterization trick; discrete SAC is more complex
- **Replay buffer memory** — stores large amounts of (s, a, r, s') tuples; memory-intensive for high-dim observations (images)
- **More complex than PPO** — three networks, multiple loss terms; more things to debug
- **Composability is approximate** (SQL) — works best when subtask reward functions are independent; breaks down with strong interactions
- **Image-based SAC** — requires additional tricks (data augmentation, pixel encoder like DrQ) to work stably with pixel inputs

---

## In This Wiki

SAC is used in: [[pick-and-place]] (Task Decomp, CQL/SAC, Sim+Real), [[grasping-and-manipulation]] (Vision-Based Grasping).
SQL is discussed in: [[reinforcement-learning]] (Composable DRL section).

Compare with: [[algo-ppo]] (on-policy, simpler), [[algo-cql]] (offline extension of SAC-style critics), [[algo-dqn]] (value-based, discrete actions).
