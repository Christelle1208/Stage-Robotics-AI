# Hybrid Imitation Learning + Reinforcement Learning

Hybrid methods combine the sample efficiency of imitation learning (learning from demonstrations) with the optimization power of reinforcement learning (improving beyond the demonstrator). The core insight: demonstrations bootstrap the policy into a reasonable region of behavior space; RL then explores and improves from there. See also [[imitation-learning]], [[reinforcement-learning]].

---

## The Problem Hybrid Methods Solve

Pure IL is bounded by demonstrator quality and suffers distribution shift. Pure RL from scratch requires enormous interaction data and carefully designed reward functions. Hybrid methods get the best of both: demonstrations reduce the exploration burden; RL pushes past the ceiling of demonstration performance.

---

## Key Papers

### AW-Opt: Learning Robotic Skills with Imitation and Reinforcement at Scale → [[algo-awac]]
**CoRL 2021**

Combines **AWAC** (Advantage-Weighted Actor-Critic) and **QT-Opt** in a hybrid IL+RL framework at scale. Uses *positive filtering* — a critic screens candidate actions and only those with positive advantage are used for IL updates — preventing the policy from imitating suboptimal behavior. The hybrid actor-critic exploration strategy mixes on-policy and off-policy data. Demonstrates significant improvement over pure IL and pure RL baselines.

*Algorithms:* [[algo-awac|AWAC]] + QT-Opt | *Tags:* hybrid, positive filtering, actor-critic, scale | See: [[reinforcement-learning]]

---

### Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning → [[algo-relay-policy]]
**Gupta, Kumar, Lynch, Levine, Hausman — Google/Berkeley, 2019**

Introduces **relay policy learning**: a two-phase approach for multi-stage, long-horizon tasks.
1. **Imitation phase:** Learn goal-conditioned hierarchical policies from unstructured, unsegmented demonstrations using a data-relabeling algorithm. The low-level policy acts for a fixed number of steps regardless of goal achieved.
2. **RL phase:** Fine-tune these policies via environment interaction.

The key insight: demonstrations do not need to cover every specific task — semantically meaningful but unsegmented behaviors are sufficient to bootstrap, then RL completes the optimization.

Demonstrated on a challenging kitchen simulation environment (multi-stage manipulation).

*Algorithms:* [[algo-relay-policy|Goal-conditioned hierarchical IL → RL fine-tuning]] | *Tags:* long-horizon, hierarchical, kitchen env, 2019

---

### Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning (HITL-RL) → [[algo-hitl-rl]]
**Luo, Xu, Wu, Levine — UC Berkeley, 2024**

A HITL-RL system where human operators provide corrections during RL training, not just initial demonstrations. Key features:
- Integrates demonstrations and human corrections
- Efficient RL algorithms with system-level optimizations
- Achieves **near-perfect success rates** in 1-2.5 hours of training
- **2x better success rate** and **1.8x faster execution** vs. IL baselines

Tasks: dynamic manipulation, precision assembly, dual-arm coordination. Demonstrates that RL can learn reactive and predictive control strategies that surpass IL with modest human input.

*Tags:* HITL, human corrections, dexterous, dual-arm, 2024 | See: [[grasping-and-manipulation]]

---

### Hybrid Robot Learning for Automatic Robot Motion Planning in Manufacturing
**GE Aerospace, 2025**

Applies hybrid RL+IL to industrial robot motion planning. Combines demonstration-based initialization with RL optimization in manufacturing environments. Developed with GE Aerospace, targeting real-world deployability.

*Tags:* manufacturing, industrial, GE Aerospace, motion planning, 2025 | See: [[pick-and-place]]

---

### Actionable Models: Unsupervised Offline RL of Robotic Skills
**Ghosh et al. — ICML 2021**

While primarily an offline RL method, Actionable Models occupies the boundary between IL and RL. Learns from unlabeled datasets (no explicit reward) using hindsight goal relabeling, then chains subgoals at test time. The "unsupervised" framing means it can leverage any behavioral data — demonstration quality is irrelevant because goals are relabeled retroactively.

*Algorithms:* Goal-conditioned Q-learning, [[algo-her|hindsight relabeling]] | See: [[reinforcement-learning]], [[algo-her]]

---

## Comparison of Hybrid Approaches

| Method | Demo Type | RL Component | Key Strength |
|--------|-----------|--------------|--------------|
| AW-Opt | Expert demos | QT-Opt off-policy | Scale; positive filtering prevents mode collapse |
| Relay Policy Learning | Unsegmented demos | Goal-conditioned fine-tuning | Long-horizon; doesn't need task-specific demos |
| HITL-RL | Demos + corrections | Online RL | Near-perfect dexterity in 1-2.5h |
| Hybrid Robot Learning | Expert demos | RL fine-tuning | Industrial settings |

---

## Related Topics
- [[imitation-learning]] — pure IL methods (ACT, Diffusion Policy, BC)
- [[reinforcement-learning]] — pure RL methods (PPO, SAC, CQL)
- [[vision-language-action-models]] — VLA-RL as a hybrid approach at scale
- [[pick-and-place]] — hybrid methods applied to P&P tasks
