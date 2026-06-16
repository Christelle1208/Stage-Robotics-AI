# Algorithm: HITL-RL — Human-in-the-Loop Reinforcement Learning

**Paper:** "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning" — Luo, Xu, Wu, Levine (UC Berkeley, 2024)
**Category:** Interactive RL — combines online RL with real-time human corrections
**Related:** [[hybrid-il-rl]], [[reinforcement-learning]], [[grasping-and-manipulation]]

---

## The Problem It Solves

Pure online RL from scratch on dexterous manipulation tasks (precision assembly, dynamic manipulation, dual-arm coordination) requires enormous sample counts — impractical on real hardware.

Pure imitation learning from demos is fast but:
- Caps performance at demonstrator quality
- Brittle to distribution shift (new camera angles, lighting, object positions)
- No mechanism to improve after deployment

**HITL-RL goal:** Achieve near-perfect success rates on *dexterous* tasks in 1-2.5 hours of real robot training by integrating human corrections throughout RL training — not just at the start.

---

## How It Works

### System Overview

HITL-RL is a **vision-based online RL system** where:
- The robot has cameras (wrist + external) as observations
- A human operator watches the robot and can intervene at any moment
- The RL policy learns from both autonomous robot experience and human corrections

### Key Components

**1. Initial demonstration phase (DAgger-style warm start)**

Before RL begins, a human tele-operates the robot for a small number of demonstrations (~10-20). These initialize the policy via behavioral cloning, giving the RL a reasonable starting point. Unlike pure BC, this is just initialization — RL will improve beyond the demos.

**2. Interactive corrections during RL**

As the RL policy runs, the human operator can:
- **Let the robot run** when it's executing correctly
- **Take over control** when the robot is about to fail (provide a correction)
- **Mark the correction endpoint** when the robot can safely resume

Corrections are stored in the replay buffer and treated as high-quality (reward = +1) transitions. They improve the policy more efficiently than waiting for the robot to accidentally discover good actions.

**Human-in-the-loop reward assignment:**
- Robot succeeds autonomously: reward = +1
- Robot fails (falls off track): reward = -1 + episode ends
- Human intervention: reward = +1 for the human-corrected segment

**3. Efficient online RL algorithm**

The RL backbone is a modified **SAC (Soft Actor-Critic)** with additional features:
- **Image observations:** Vision encoder (CNN or ViT) processes camera inputs
- **Replay buffer** stores all experience (autonomous + human-corrected)
- **Frequent updates** — multiple gradient steps per environment step

**4. System-level design choices**

Several engineering choices are highlighted as critical:
- **Low-latency control loop:** 10-25Hz control frequency to allow reactive corrections
- **High-quality camera placement:** Multiple calibrated cameras for clear state estimation
- **Reward function design:** Binary (success/fail) with carefully defined success conditions
- **Scripted exploration noise:** Early training adds structured noise to the policy to encourage exploration of contact-rich states

---

## Why It Works

### Human Intervention as Efficient Data

Random exploration in RL rarely produces successful contact-rich manipulation (precision grasping, assembly). The human corrects the robot *exactly* when it's about to fail, providing:
- High-quality demonstrations *in the context where they're needed* (specific robot pose, specific object configuration that caused failure)
- Much more informative than offline demos (which may not cover the failure modes the RL policy encounters)

This is fundamentally different from behavioral cloning on offline demos: the human provides targeted corrections to the current policy's specific failure modes, not general demonstrations.

### RL Can Improve Beyond Human Corrections

After enough corrections, the RL policy generalizes: it learns the underlying skill (not just memorizes corrections). It then succeeds in situations it hasn't been corrected on — surpassing what pure imitation would achieve.

### Reactive + Predictive Control

An interesting finding: HITL-RL learns both **reactive** (respond to current observation) and **predictive** (anticipate future states) control strategies. Some tasks benefit from looking ahead (object about to fall), others from immediate feedback. RL on vision naturally discovers which mode to use per situation.

---

## Evaluation

### Tasks (4 dexterous manipulation tasks)

| Task | Description | Challenge |
|------|-------------|-----------|
| Dynamic object catching | Catch a thrown object mid-air | Timing, speed |
| Precision assembly | Insert peg into tight-tolerance hole (<1mm) | Precision |
| Dual-arm coordination | Two arms together pick large object | Synchronization |
| Contact-rich in-hand manipulation | Reorient object using fingertips | Dexterity |

These tasks were chosen to represent the hardest manipulation challenges — tasks where prior work either fails entirely or requires hundreds of hours of RL.

### Baselines

| Baseline | Description |
|----------|-------------|
| Behavioral Cloning (BC) | Supervised learning on same demonstrations |
| BC + GAIL | Adversarial IL (imitation + discriminator) |
| Pure RL (SAC, no human) | Online RL without any demonstrations |
| RL from demos only (no corrections) | Online RL warm-started from demos, no subsequent human input |

### Training Protocol
- **1-2.5 hours of real robot training per task** (not simulation)
- ~200-500 robot episodes total
- Human provides corrections during training (not after)

### Results

**Success rates (100 evaluation trials each):**

| Task | BC | BC+GAIL | RL from demos | HITL-RL |
|------|----|---------|---------------|---------|
| Dynamic catching | 22% | 34% | 41% | **89%** |
| Precision assembly | 8% | 15% | 28% | **94%** |
| Dual-arm coordination | 31% | 42% | 55% | **91%** |
| In-hand manipulation | 5% | 9% | 18% | **88%** |
| **Average** | **16.5%** | **25%** | **35.5%** | **90.5%** |

**Key numbers:**
- HITL-RL achieves **~90% average** vs. ~35% for best baseline (RL from demos)
- **~2× better success rate** than RL from demos alone
- **~1.8× faster cycle time** (task completion speed) vs. BC baselines
- Training converges within **1-2.5 hours** on real hardware

### Number of Human Interventions
- Average: ~50-100 corrections per task over the full training period
- Corrections decrease over time as the policy improves (early training: frequent corrections; late training: rare corrections)

---

## Pros

- **Near-perfect performance** — 90%+ success on tasks that stump other approaches
- **Short training time** — 1-2.5 hours on real hardware vs. days of simulation
- **Corrects specific failures** — human targets exactly where the current policy fails; maximum data efficiency
- **Improves beyond demonstrator** — unlike pure IL, RL generalizes from corrections
- **Vision-based** — camera inputs; no privileged state information; deployable in unstructured environments

## Cons

- **Requires expert human operator** — the human must understand the task and react quickly; not an autonomous training process
- **Scales poorly with task complexity** — the human must monitor the robot continuously; becomes exhausting for very long tasks
- **Not reproducible without the operator** — results depend on operator skill and attention; different operators may yield different results
- **Real hardware only** — the correction mechanism requires the physical robot; not directly applicable to simulation-only setups
- **Human bias** — corrections reflect the operator's strategy; may constrain the policy to human-like solutions even when non-human strategies exist

---

## Comparison: HITL-RL vs. Other Hybrid Methods

| | HITL-RL | AWAC | Relay Policy Learning |
|-|---------|------|----------------------|
| Demo type | Interactive corrections | Offline expert demos | Offline unstructured demos |
| Human involvement | Continuous during RL | One-time demo collection | One-time demo collection |
| Training time | 1-2.5h real hardware | Hours-days | Hours-days |
| Improvement mechanism | RL + targeted corrections | Advantage-weighted BC → RL | Hierarchical goals → RL |
| Best for | Dexterous, precise tasks | High-quality offline data available | Long-horizon tasks |

---

## In This Wiki

HITL-RL: [[hybrid-il-rl]], [[grasping-and-manipulation]], [[reinforcement-learning]].
Compare with: [[algo-awac]] (offline demos instead of interactive corrections), [[algo-relay-policy]] (hierarchical structure for long-horizon rather than corrections for dexterity), [[algo-sql-sac]] (SAC as the RL backbone).
