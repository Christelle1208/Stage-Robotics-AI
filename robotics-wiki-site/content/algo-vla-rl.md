# Algorithm: VLA-RL

**Paper:** "VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning" — Lu, Guo, Zhang et al. (Tsinghua University, 2025)
**Category:** Online RL fine-tuning of pretrained auto-regressive VLAs
**Related:** [[vision-language-action-models]], [[hybrid-il-rl]], [[reinforcement-learning]], [[algo-openvla]]

---

## The Problem It Solves

VLAs like OpenVLA achieve strong performance through offline imitation learning on large robot datasets. However:

- **Distribution shift at test time:** The policy has only seen states from the training data. In deployment, it encounters novel configurations, object positions, and task variations not in the training set → failure.
- **Ceiling of imitation:** IL can never exceed the demonstrator's performance; it can only interpolate between demonstrated behaviors.
- **Sparse rewards in robotics:** Online RL with VLAs is attractive, but robot tasks often have sparse terminal rewards (success/fail only) — making naive RL fine-tuning extremely slow.

**VLA-RL goal:** Apply online RL to improve a pretrained VLA (OpenVLA-7B) *after* offline training, enabling it to:
1. Recover from distribution shift (explore and learn from failures)
2. Surpass the demonstrator
3. Scale with more test-time compute (inference scaling)

---

## How It Works

### 1. Trajectory-Level RL Formulation for Auto-Regressive VLAs

Standard RL applies per-timestep rewards. But VLAs generate actions token-by-token (7 tokens for a 7-DoF action at each control step) — the "actions" are individual tokens within an LLM.

VLA-RL reframes this: a full manipulation **trajectory** is modeled as a **multi-modal multi-turn conversation**:

```
Turn 1: [visual obs + language task] → [action tokens step 1]
Turn 2: [visual obs step 2] → [action tokens step 2]
...
Turn T: [visual obs T] → [action tokens T] + terminal signal
```

RL is applied at the **trajectory level** — the entire episode's token sequence is the unit of optimization, not individual token predictions. This avoids the credit assignment problem of per-token RL (which token caused the success/failure?).

**Policy gradient at trajectory level:**
```
∇J(θ) = E_trajectory [Σ_t R_t · ∇ log π_θ(a_t | context_t)]
```

Where R_t is derived from the *process reward model* (see below), not just terminal reward.

### 2. Process Reward Model (PRM) — Dense Rewards from Sparse Labels

The key innovation for making RL tractable with sparse rewards:

**Step 1: Automatic task segmentation.**
The manipulation trajectory is automatically segmented into meaningful task phases using a pretrained VLM (e.g., detecting "robot reaches for object," "robot grasps object," "robot moves to goal," "robot places object"). This is done without human annotation.

**Step 2: Pseudo-reward annotation.**
For each trajectory in the replay buffer, a pretrained VLM is used to annotate which task phases were *completed successfully*:
```
If phase "grasps object" completed: reward += +1 for all steps in that phase
If phase "places object" completed: reward += +2 for all steps in that phase
```
This converts sparse terminal reward into a **dense process reward signal** across the trajectory.

**Step 3: Fine-tune the VLM → Process Reward Model (PRM).**
The annotated trajectories are used to fine-tune a separate VLM that can predict per-timestep rewards from visual observations:
```
PRM(o_t, phase_t) → r_t ∈ [0, 1]
```
The PRM provides reward signals during online RL, making training tractable even when terminal rewards are rare.

### 3. Scalability Improvements

Several implementation findings enable stable and efficient RL fine-tuning of 7B VLAs:

**Curriculum selection strategy:** Start RL training with easier task instances (simpler object positions, fewer distractors), gradually adding harder ones. Prevents early catastrophic failures that could collapse the policy.

**GPU-balanced vectorized environments:** Multiple simulation environments run in parallel across GPUs, maximizing GPU utilization. The VLA forward pass dominates compute; balancing this across environments reduces idle time.

**Batch decoding:** Instead of decoding each environment's action serially (one VLA forward pass per env), batch all environments into a single batched VLA forward pass. Achieves near-linear throughput scaling with batch size.

**Critic warmup:** Before RL updates begin, the critic (value function) is pre-warmed by supervised training on trajectory returns from the offline dataset. This prevents noisy, uninformative critic estimates in early RL.

---

## Why It Works

### Online Exploration Fixes Distribution Shift

VLA-RL's RL training sends the robot into diverse states not in the offline dataset. The robot *fails*, then learns from those failures. Over time, the policy's visitation distribution broadens — covering more of the state space. The PRM provides reward signal in these novel states.

### Process Rewards Enable Temporal Credit Assignment

With binary terminal rewards, the RL algorithm can't tell which actions caused success/failure. The PRM's dense reward provides temporal credit: "you grasped correctly at step 30; you failed at step 52 when placing." The policy gradient can adjust the bad steps specifically.

### Pre-trained VLA Is a Strong Starting Point

Unlike RL from scratch, VLA-RL starts from OpenVLA-7B — already a strong policy. RL only needs to *refine* rather than *discover* basic behaviors. This dramatically reduces the exploration required.

---

## Evaluation

### Benchmark: LIBERO

**LIBERO** is a multi-task robot manipulation benchmark with 4 suites:
- **LIBERO-Spatial:** 10 tasks varying object positions
- **LIBERO-Object:** 10 tasks varying object types
- **LIBERO-Goal:** 10 tasks varying goal configurations
- **LIBERO-Long:** 10 long-horizon tasks (3-4 sequential subtasks)

**40 tasks total.** Each task evaluated on 20 trials × 5 seeds.

### Baselines

| Method | Type |
|--------|------|
| OpenVLA (offline fine-tuned) | Offline IL |
| Octo (fine-tuned) | Offline IL |
| RoboFlamingo | Offline IL (VLM-based) |
| π0-FAST | Advanced commercial VLA |
| **VLA-RL** | Online RL fine-tuning |

### Results

| Suite | Best Offline Baseline | VLA-RL | Δ |
|-------|----------------------|--------|---|
| LIBERO-Spatial | 85.3% | **90.1%** | +4.8% |
| LIBERO-Object | 79.2% | **83.4%** | +4.2% |
| LIBERO-Goal | 76.5% | **81.0%** | +4.5% |
| LIBERO-Long | 62.1% | **66.4%** | +4.3% |
| **Average** | **75.8%** | **80.3%** | **+4.5%** |

VLA-RL also **matches π0-FAST** (an advanced commercial system) on most LIBERO tasks.

### Inference Scaling (Test-Time Compute)

A unique finding: VLA-RL **scales with test-time compute**:
- Sample N action candidates from VLA-RL
- Use PRM to score each candidate
- Execute the highest-scoring candidate

With N=1 (standard): 80.3% average
With N=4: 83.1%
With N=8: 85.2%
With N=16: 86.4%

**Interpretation:** This is an early sign of "inference scaling laws" for robotics — analogous to chain-of-thought reasoning in LLMs. More compute at inference → better decisions.

### Ablations
- **No PRM (terminal reward only):** ~62% (massive drop; shows PRM is critical)
- **No curriculum:** ~72% (significant drop; early failures destabilize training)
- **No critic warmup:** ~74% (noisy early training)
- **No batch decoding:** 3× slower training (confirms scalability importance)

---

## Pros

- **Surpasses offline IL** — +4.5% average on LIBERO vs. best offline fine-tuned baseline
- **Matches commercial models** — π0-FAST competitive without proprietary data
- **Inference scaling** — an emergent scaling law: better results with more test-time compute
- **Works from pretrained VLA** — no RL from scratch needed; stable starting point
- **PRM solves sparse reward** — provides dense temporal credit without human reward design

## Cons

- **Complex system** — PRM training + curriculum + vectorized envs + critic warmup = many moving parts
- **Simulation-only** — evaluation is entirely in LIBERO (simulation); real-robot results not reported
- **PRM is trained on pseudo-labels** — automatic phase detection and annotation may be noisy; quality depends on VLM capabilities
- **Compute intensive** — fine-tuning a 7B VLA with online RL + PRM requires significant GPU resources
- **Moderate gains** — +4.5% average improvement is meaningful but not transformative; many challenging tasks remain

---

## In This Wiki

VLA-RL: [[vision-language-action-models]], [[hybrid-il-rl]], [[reinforcement-learning]].
Compare with: [[algo-openvla]] (the base model VLA-RL fine-tunes), [[algo-awac]] (offline→online hybrid with similar motivations but for smaller policies), [[algo-ppo]] (the RL algorithm used inside VLA-RL).
