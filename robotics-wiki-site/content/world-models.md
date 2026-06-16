# World Models and Action Models

World models learn a predictive model of the environment — how states transition given actions — which can then be used for planning, simulation, or zero-shot policy derivation. Action-centered world models (WAMs) go further, making the world model's internal dynamics directly actionable as a policy. See also [[vision-language-action-models]], [[simulation-and-tools]].

---

## The Core Idea

Traditional RL requires many real-environment interactions. World models compress this: once an internal model of environment dynamics is learned, the agent can plan entirely "in imagination," reducing real-world interaction. Recent work asks: can a world model be *so good* that it becomes a policy without any additional RL training?

---

## Key Papers

### GigaWorld-Policy: An Efficient Action-Centered World-Action Model
**2026**

Introduces an **action-centered world model** where the model's internal dynamics are organized around action transitions rather than pixel-level predictions. Key results:
- **9× faster** than prior world-action model (Motus)
- **7% higher task success rate**
- Pixel-level action dynamics representation

The efficiency gains come from a more compact state representation focused on what changes due to actions, rather than reconstructing full pixel observations.

*Tags:* WAM, action-centered, efficient, 2026 | See: [[vision-language-action-models]]

---

### World Action Models are Zero-Shot Policies
**Ye, Ge, Zheng, Gao et al. — 2026**

A large collaborative work demonstrating that world models trained at scale can function as **zero-shot robot policies** — without task-specific training. The world model learns general physical dynamics; at test time, model predictive control (MPC) or similar planning extracts actions.

Key implications:
- Scaling world model training can yield generalizable manipulation policies
- Bridges the gap between video prediction models and executable robot policies
- Demonstrates a path to generalization without per-task demonstration collection

*Tags:* zero-shot, world model, scaling, planning, 2026 | See: [[vision-language-action-models]]

---

### MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation → [[algo-molmobot]]
**2026**

Achieves zero-shot manipulation by leveraging large-scale simulation data in the [[algo-molmospaces|MolmoSpaces]] ecosystem. Built on the [[algo-molmo2|Molmo2]] VLM backbone. Shows that simulation scale — not just model scale — is a key lever for zero-shot generalization.

*Tags:* zero-shot, simulation scale, Molmo ecosystem, 2026 | See: [[simulation-and-tools]], [[algo-molmo2]], [[algo-molmospaces]]

---

### Learning High-Level Robotic Manipulation Actions with Visual Predictive Model
**2024**

Learns a visual predictive model that predicts future visual observations given high-level actions. An **action decomposer** breaks high-level commands into primitive P&P actions that the visual model can execute.

The world-model approach here operates at the semantic/high-level rather than pixel level, making it interpretable and compositional.

*Tags:* visual predictive model, action decomposer, high-level planning, 2024 | See: [[pick-and-place]]

---

### Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning
**NVIDIA, 2026**

A foundation model for embodied reasoning built on physical common sense. NVIDIA's contribution to the WAM space, focusing on physical intuition as a prerequisite for robust robot manipulation.

*Tags:* physical common sense, embodied reasoning, NVIDIA, 2026 | See: [[llms-for-robotics]]

---

## Connection to VLAs

The boundary between VLAs and world models is blurring:
- **VLAs** (see [[vision-language-action-models]]) predict actions directly from visual/language inputs
- **WAMs** predict world-state dynamics and derive actions from planning
- Modern systems (GigaWorld-Policy, World Action Models) unify these: a single model both predicts world state and produces actions

---

## Related Topics
- [[vision-language-action-models]] — VLAs as the direct-action counterpart to world models
- [[simulation-and-tools]] — simulation environments for world model training
- [[reinforcement-learning]] — world models used as environments for RL training
- [[pick-and-place]] — task setting for world model evaluation
