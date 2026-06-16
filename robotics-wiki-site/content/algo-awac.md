# Algorithm: AWAC — Advantage-Weighted Actor-Critic (and AW-Opt)

**Papers:**
- "Accelerating Online Reinforcement Learning with Offline Datasets" — Nair et al. (2020) → AWAC
- "AW-Opt: Learning Robotic Skills with Imitation and Reinforcement at Scale" — CoRL 2021 → AW-Opt (AWAC + QT-Opt)
**Category:** Hybrid offline→online RL; policy constraint via importance weighting
**Related:** [[hybrid-il-rl]], [[reinforcement-learning]], [[algo-cql]], [[algo-sql-sac]]

---

## The Problem AWAC Solves

You have a dataset of demonstrations (from human experts or a scripted policy) and want to:
1. Bootstrap a policy that's already reasonable (using the demos)
2. Then improve it further with online RL (to exceed demonstration performance)

The challenge: standard online RL from a demonstration-initialized policy often **forgets** the demonstrations (catastrophic interference) or **overfits** to them and fails to explore. You need the policy to stay *near* the demonstrations during early RL, but not be *constrained* to them forever.

AWAC solves this without explicitly constraining the policy — it implicitly stays close via **advantage-weighted regression**.

---

## How AWAC Works

### Core Mechanism: Advantage-Weighted Regression (AWR)

Instead of maximizing Q-values directly (which can lead the policy far from demonstrations), AWAC trains the policy by solving a **weighted imitation problem**:

```
L_actor(θ) = -E_{(s,a)~buffer} [log π_θ(a|s) · exp(A_φ(s,a) / β)]
```

Where:
- `A_φ(s,a) = Q_φ(s,a) - V_φ(s)` is the **advantage** under the current critic
- `β` is a temperature hyperparameter
- `exp(A/β)` is the **importance weight** — actions with high advantage get weighted up; low-advantage actions get weighted down

**Intuition:** AWAC is doing supervised learning on actions from the buffer, but weighting each sample by how *good* that action was (according to the current Q-function). Actions that are better than average get high weight; bad actions get near-zero weight. The policy moves toward the high-advantage subset of the data.

### Why This Avoids the Constraint Problem

Classical policy constraint methods (like BEAR or BCQ) explicitly project the policy onto the behavioral distribution:

```
max J(π) subject to D_KL(π || π_behavior) ≤ ε
```

This is hard to optimize and often overly restrictive. AWAC *implicitly* constrains through the weighting:
- High β → weights are near-uniform → policy is free to diverge from data
- Low β → weights are sharply peaked on best actions → policy stays very close to demonstrated behaviors
- No hard constraint, just a soft preference via the temperature

### Critic Update

AWAC uses a standard off-policy critic (like SAC's critic):

```
L_critic(φ) = E[(Q_φ(s,a) - (r + γ·V_φ(s')))²]
V_φ(s) = E_{a~π}[Q_φ(s,a)]
```

Crucially, the critic is trained **off-policy** — it can use *any* data in the buffer (demos + online experience). The actor is then guided by this critic.

### Online Learning Loop

```
Initialize: pre-load demo data into replay buffer
          initialize actor via behavioral cloning on demos

Loop:
1. Sample action a ~ π_θ(·|s), execute in environment
2. Store (s, a, r, s') in replay buffer (alongside demos)
3. Update critic using full buffer (off-policy)
4. Update actor via advantage-weighted regression
```

As online experience accumulates, the critic learns which demo behaviors are actually good, and the actor learns to exploit those — while discarding suboptimal demonstration modes.

---

## AW-Opt: AWAC at Scale

AW-Opt (CoRL 2021) extends AWAC with two key components for scaling to real robot learning:

### 1. Positive Filtering

Standard AWAC uses all transitions from the buffer, weighted by advantage. **Positive filtering** only uses transitions with positive advantage:

```
L_actor(θ) = -E_{(s,a): A(s,a)>0} [log π_θ(a|s) · exp(A(s,a) / β)]
```

This prevents the policy from imitating bad actions (even with very low weight) and speeds up learning by focusing gradient computation on useful transitions.

### 2. QT-Opt Integration

QT-Opt is a Q-learning variant designed for robot grasping (Google, 2018) that:
- Uses a **CEM (Cross-Entropy Method)** to optimize continuous actions from Q-values
- Handles high-dimensional continuous action spaces efficiently
- Scales to large replay buffers

AW-Opt uses QT-Opt as the critic backbone, replacing AWAC's standard SAC-style critics. This provides better scaling to real-world robot datasets (millions of grasping demos).

### 3. Hybrid Actor-Critic Exploration

AW-Opt maintains two exploration modes simultaneously:
- **Actor-based exploration:** Sample from the learned policy (exploitative)
- **Critic-based exploration:** Use CEM on the Q-function directly (can find high-Q actions outside policy support)

This mix allows AW-Opt to both exploit learned demonstrations and explore novel high-reward actions.

---

## Evaluation

### AWAC (Original Paper)
**Environments:** Dexterous robot hand manipulation (door opening, object relocation, pen spinning — from DAPG benchmark) and MuJoCo locomotion tasks

**Setup:**
- Small set of human demonstrations (25-200 episodes)
- Online RL from demo initialization

**Baselines:** SAC from scratch, behavioral cloning, BEAR, BRAC

**Results:**
- AWAC with demos converges in **2-5× fewer episodes** than SAC from scratch
- AWAC substantially outperforms BC (can surpass demonstrator)
- More stable than BEAR/BRAC (no hard constraint failures)
- Key: can **exceed demonstration performance** — unlike BC which is capped by demo quality

### AW-Opt (CoRL 2021)

**Task:** Real-robot grasping and stacking (challenging manipulation tasks)

**Scale:** Uses large datasets (human tele-operation + scripted policy data, order of 100k+ episodes)

**Results:**
- AW-Opt significantly outperforms pure RL (AWAC/SAC from scratch)
- Outperforms pure IL (behavioral cloning)
- Positive filtering: ~15% improvement over standard AWAC
- Key result: demonstrates the hybrid IL+RL combination scales to real robot tasks at CoRL-level difficulty

**Evaluation protocol:**
- Real robot trials (not simulation)
- 50-100 evaluation grasps per condition
- Success rate measured

---

## Pros

- **Smooth offline→online transition** — starts from demonstrations, improves with RL, no sudden policy collapse
- **No explicit constraint** — soft implicit constraint via temperature β is more flexible than hard KL constraints
- **Can exceed demonstrator** — unlike behavioral cloning
- **Off-policy critic** — can use all available data (demos + online) efficiently
- **Simple to implement** — just a weighted regression loss on top of standard actor-critic

## Cons

- **Still requires good demonstrations** — if demos are highly suboptimal, advantage weights will mostly be negative and learning is slow
- **Temperature β sensitivity** — needs tuning per task; wrong β leads to either too conservative (slow improvement) or too aggressive (forgetting demos)
- **Q-value quality** — advantage weights depend on the critic; early in training when the critic is poor, the weights are noisy
- **Positive filtering** (AW-Opt) may be too aggressive — discards potentially useful low-advantage transitions
- **Scaling** — AW-Opt requires substantial infrastructure (large replay buffers, distributed training) to work at the scale shown

---

## Comparison to CQL

| | AWAC | CQL |
|-|------|-----|
| **Goal** | Offline init → online improvement | Purely offline |
| **Constraint mechanism** | Advantage weighting | Q-value penalization |
| **Can use online data** | Yes (designed for it) | Optional fine-tuning only |
| **Requires demos** | Yes | No (any offline data) |
| **Complexity** | Medium | Medium |

---

## In This Wiki

AWAC/AW-Opt: [[hybrid-il-rl]], [[reinforcement-learning]].
Compare with: [[algo-cql]] (offline RL alternative), [[algo-relay-policy]] (different hybrid approach using hierarchical goals), [[algo-hitl-rl]] (human corrections instead of offline data).
