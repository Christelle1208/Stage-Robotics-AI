# Algorithm: PPO — Proximal Policy Optimization

**Paper:** "Proximal Policy Optimization Algorithms" — Schulman, Wolski, Dhariwal, Radford, Klimov (OpenAI, 2017)
**Category:** On-policy policy gradient RL
**Related:** [[reinforcement-learning]], [[algo-her]], [[algo-sql-sac]]

---

## The Problem It Solves

Standard policy gradient methods (REINFORCE, vanilla actor-critic) are unstable: a single bad gradient step can collapse the policy, requiring it to be retrained from scratch. TRPO (Trust Region Policy Optimization) fixes this with a hard constraint on how much the policy can change per update, but TRPO is complex to implement (requires conjugate gradients and line search) and can only make one gradient update per data batch.

**PPO goal:** achieve TRPO-level stability with a much simpler implementation, while allowing multiple gradient update epochs on each collected batch of data.

---

## How It Works

### The Core Mechanism: Clipped Surrogate Objective

PPO defines a *probability ratio* between the new policy and the old policy for each action taken:

```
r(θ) = π_θ(a|s) / π_θ_old(a|s)
```

If the new policy assigns higher probability to an action than the old one, `r > 1`. If lower, `r < 1`.

The standard policy gradient objective would simply maximize `r(θ) × A(s,a)` where `A` is the advantage (how much better this action was than average). But this can cause destructively large updates.

PPO clips this ratio to stay within `[1-ε, 1+ε]` (typically ε = 0.2):

```
L_CLIP(θ) = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
```

**The clip works as follows:**
- If the advantage is **positive** (good action): we want to increase `r`, but we stop increasing reward credit once `r > 1+ε`
- If the advantage is **negative** (bad action): we want to decrease `r`, but we stop reducing reward credit once `r < 1-ε`

In both cases, we take the *minimum* of clipped and unclipped objectives, ensuring the gradient never pushes the policy too far in any direction.

### The Full Objective

PPO combines three terms:

```
L(θ) = L_CLIP(θ) - c₁·L_VF(θ) + c₂·H(π_θ)
```

- `L_VF`: value function loss (critic, trained via MSE on returns)
- `H(π_θ)`: entropy bonus to encourage exploration (coefficient c₂)
- `c₁`, `c₂`: hyperparameter weights

### Training Loop

```
1. Collect T timesteps of data with current policy π_θ_old
2. Compute advantages A_t using GAE (Generalized Advantage Estimation)
3. For K epochs:
   a. Sample mini-batches from the collected data
   b. Compute L_CLIP(θ) using current θ and stored π_θ_old
   c. Update θ via gradient ascent
4. Set π_θ_old ← π_θ
5. Repeat
```

The key innovation: steps 3a-3c can run **multiple times** on the same batch because the clip prevents the ratio from growing too large.

---

## Why It Works

### Intuition

The clip acts as an automatic trust region. Instead of solving a constrained optimization problem (like TRPO), PPO simply ignores gradient signal once the policy has moved "far enough" from the old policy. This is:
- Simpler (no conjugate gradients, no KL constraint)
- More flexible (works with Adam and standard deep learning infrastructure)
- More stable than unconstrained policy gradients (because large updates are silently ignored)

### The Min Operator

The `min(clipped, unclipped)` is crucial. It means:
- When `r` is already outside the clip range, the gradient is **zero** — the policy simply won't move further in that direction
- When `r` is inside the clip range, the gradient is the standard policy gradient

This creates a pessimistic lower bound on the objective: PPO never takes credit for pushing the policy further than the trust region.

---

## Evaluation

### Benchmarks Used
- **Simulated robotic locomotion** in MuJoCo: HalfCheetah, Hopper, Walker2d, Ant, Humanoid
- **Atari games** (discrete action space)
- Comparison: other online policy gradient methods (TRPO, A3C, ACER)

### Metrics
- Total reward per episode
- Sample complexity (reward vs. number of environment interactions)
- Wall-clock time

### Results
- PPO outperforms A2C and ACER on most Atari games
- PPO matches or exceeds TRPO on continuous control tasks while being substantially simpler
- Key benefit: **better sample complexity** — multiple update epochs per data batch means fewer total environment interactions needed

---

## Results (Numbers)

From the paper:
- On MuJoCo continuous control: PPO achieves comparable or superior performance to TRPO
- On Atari: 11/49 games where PPO clearly beats prior methods; competitive on the rest
- Multiple update epochs (K=3-10) per batch without destabilization

In downstream robotics papers using PPO:
- Simulated + Real P&P (Lobbezoo 2023): PPO trains Franka Panda to reach/grasp/place; successfully transfers to real hardware
- MuJoCo Playground (2025): PPO + MJX achieves sim-to-real on quadrupeds and humanoids

---

## Pros

- **Simple to implement** — one function call in most RL libraries (Stable Baselines 3, RLlib, CleanRL)
- **Stable** — clip prevents catastrophic policy collapse
- **Versatile** — works for discrete and continuous action spaces, on-policy settings
- **Allows multiple epochs** per data batch — more data efficient than single-update policy gradients
- **Well-understood** — massive body of tuning knowledge available

## Cons

- **On-policy** — data collected with old policy cannot be reused efficiently (no replay buffer); sample efficiency is worse than off-policy methods (SAC, DQN)
- **Hyperparameter sensitivity** — ε, learning rates, and GAE λ need tuning
- **High variance** advantage estimates — requires large batch sizes for stability
- **Not for offline RL** — requires online environment interaction; cannot learn from fixed datasets (use CQL or IQL instead)
- **Slow in sparse reward settings** — without reward shaping or HER, PPO struggles when rewards are rare

---

## Key Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| Clip ε | 0.1–0.2 | Larger = more aggressive updates; smaller = more conservative |
| Learning rate | 3e-4 | Use cosine annealing or linear decay |
| GAE λ | 0.95 | Bias-variance tradeoff for advantage estimation |
| Epochs K | 4–10 | More epochs per batch = more efficient but risk instability |
| Batch size | 2048–4096 | Larger = more stable advantage estimates |
| Entropy coeff c₂ | 0.01 | Encourages exploration; reduce over training |

---

## In This Wiki

PPO is used in: [[pick-and-place]] (multiple papers), [[simulation-and-tools]] (MuJoCo Playground), [[hybrid-il-rl]] (as the RL component).

Compare with: [[algo-sql-sac]] (off-policy, more sample efficient), [[algo-her]] (adds sparse reward handling on top of PPO/DDPG), [[algo-cql]] (offline RL when you can't interact with environment).
