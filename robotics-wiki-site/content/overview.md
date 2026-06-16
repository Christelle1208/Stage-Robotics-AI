# Robot Learning Wiki — Overview & Navigation Guide

This wiki compiles research across **reinforcement learning, imitation learning, and vision-language-action models** applied to robot manipulation. It is built incrementally from source papers, with synthesis and cross-references maintained throughout. It accompanies an internship project on pick-and-place learning for the **SO-100** robot arm.

---

## The Three Paradigm Families

Every paper in this wiki ultimately falls into one of three families. Understanding what each family *fundamentally is* — not just what it does — is the most important orientation tool.

### 🎯 Reinforcement Learning (RL)
> *The robot learns by trying things and receiving feedback on whether they worked.*

The robot explores an environment, takes actions, and receives a reward signal. Over thousands (or millions) of episodes, it learns a policy that maximizes cumulative reward. No demonstrations needed — but a reward function must be designed, and a simulator (or very tolerant hardware) is required for exploration.

**The core bet:** if you can specify what "good" looks like as a scalar number, RL will find a way to achieve it — even superhuman ways you didn't anticipate.

→ See [[reinforcement-learning]] for algorithms and papers  
→ See [[algo-sac]], [[algo-ppo]], [[algo-dqn]], [[algo-her]] for algorithm deep-dives

---

### 🎭 Imitation Learning (IL)
> *The robot learns by watching expert demonstrations and replicating the behavior.*

A human (or another robot) performs the task, and the robot learns to reproduce it. No reward needed — but demonstrations must be collected, and the robot can rarely exceed the demonstrator's performance without additional RL fine-tuning.

**The core bet:** humans can perform the task well and demonstrate it cheaply; the robot just needs to generalize from those examples.

→ See [[imitation-learning]] for algorithms and papers  
→ See [[algo-act]], [[algo-diffusion-policy]], [[algo-vq-bet]] for algorithm deep-dives

---

### 🌐 Vision-Language-Action Models (VLAs)
> *A large pre-trained model encodes general world knowledge and is fine-tuned to control a robot.*

VLAs inherit visual and language understanding from internet-scale pretraining (like GPT, but for robots). They accept natural language instructions and camera images, and output robot actions. Fine-tuning on a small number of robot demonstrations adapts them to a specific setup.

**The core bet:** the hard part of robot intelligence (understanding the world, reasoning about objects) is already solved by large models — we just need to add an "action head."

→ See [[vision-language-action-models]] for models and papers  
→ See [[algo-openvla]], [[algo-octo]] for algorithm deep-dives

---

## How the Families Compare

| Dimension | RL | IL | VLA |
|-----------|----|----|-----|
| **Data needed** | None (reward + sim) | 10–1000 demos | Pretrained; ~50–500 fine-tune demos |
| **Reward design** | Required | Not needed | Not needed |
| **Sim required** | Usually | No | No |
| **Generalization** | Poor (task-specific) | Limited (setup-specific) | Strong (language-conditioned) |
| **Performance ceiling** | Superhuman possible | Bounded by demos | Bounded by pretraining + fine-tune |
| **Compute (training)** | High (exploration) | Low-medium | High (pretraining); medium (fine-tune) |
| **Compute (inference)** | Low | Low | Medium-High |
| **Real-robot safety** | Risky (exploration) | Safe | Safe |
| **Best for** | Precise control, sim-available | Fixed-task dexterity | Multi-task, language-conditioned |

---

## How to Navigate This Wiki

### "I want to solve a specific task"

| Task | Start here |
|------|-----------|
| Pick-and-place | [[pick-and-place]] → synthesis section |
| Grasping an object | [[grasping-and-manipulation]] |
| Multi-step / long-horizon | [[hybrid-il-rl]], [[imitation-learning#Mamba2Diff]] |
| Language-conditioned control | [[vision-language-action-models]] |
| Trajectory / motion planning | [[trajectory-planning]] |

### "I need to choose an approach"

→ See [[decision-guide]] for a full decision flowchart

Quick rules of thumb:
- **Have a simulator? Can define a reward?** → Start with RL (SAC for continuous, PPO for simpler tasks)
- **Have 20+ good demos? Fixed setup?** → Start with ACT or Diffusion Policy
- **Need language conditioning or multi-object generalization?** → SmolVLA or OpenVLA fine-tune
- **Have demos but want to go beyond them?** → Hybrid IL+RL (AWAC, HITL-RL)

### "I want to understand an algorithm"

All algorithm explainer pages are prefixed with `algo-`. Browse the [[index]] or search for the name.

---

## Connection to Experiments

This wiki was built in parallel with experiments on the **SO-100** robot arm for a pick-and-place task. Results are integrated progressively as they become available.

| Method | Environment | Status | Result |
|--------|-------------|--------|--------|
| SAC (3-subtask, recovery) | Simulation (MuJoCo) | ✅ Done | **92% overall** (Reach 96% · Grasp 92% · Place 92%) |
| ACT (Dataset_v4, 100k steps) | Real hardware (SO-100) | ✅ Done | **83% ID / 94% @ 45° / 75% distractor** |
| SmolVLA (Dataset_v4, 20k steps) | Real hardware (SO-100) | ✅ Done | **58% ID / 0% distractor** (many near-successes) |

**Meta-finding:** Dataset iteration (v1→v4) was the biggest performance driver. ACT outperforms SmolVLA at ~111 episodes — advantage likely reverses with more diverse fine-tuning data. See [[decision-guide]] REX section and [[evaluation-protocol]] for full breakdown.

---

## Wiki Structure

```
wiki/
├── overview.md          ← You are here
├── decision-guide.md    ← How to choose an approach
├── index.md             ← Full catalog of all pages
├── log.md               ← Chronological record of updates
├── lint-report.md       ← Health check / open issues
│
├── Topic pages (task/paradigm level)
│   ├── reinforcement-learning.md
│   ├── imitation-learning.md
│   ├── hybrid-il-rl.md
│   ├── vision-language-action-models.md
│   ├── pick-and-place.md
│   ├── grasping-and-manipulation.md
│   ├── trajectory-planning.md
│   ├── simulation-and-tools.md
│   ├── world-models.md
│   └── llms-for-robotics.md
│
└── Algorithm explainer pages (algo-*.md)
    ├── algo-sac.md, algo-ppo.md, algo-dqn.md ...
    ├── algo-act.md, algo-diffusion-policy.md ...
    └── algo-openvla.md, algo-octo.md ...
```
