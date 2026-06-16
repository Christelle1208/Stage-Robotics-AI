# System: MolmoSpaces — Large-Scale Open Simulation Ecosystem

**Paper:** "MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation" — Kim, Pumacay, Rayyan et al. (University of Washington, 2026)
**Category:** Simulation ecosystem and benchmark for robot learning at scale
**Related:** [[simulation-and-tools]], [[algo-molmo2]], [[algo-molmobot]], [[world-models]]

---

## What It Is

MolmoSpaces is not a single algorithm but a **simulation ecosystem** — an open-source collection of environments, assets, benchmarks, and tools designed for large-scale robot learning research. It is the infrastructure layer that enables [[algo-molmobot]] (zero-shot manipulation) and provides standardized benchmarks for evaluating VLA policies including the Molmo family.

The scale is unprecedented among open robotics simulation systems:
- **230,000+ indoor environments**
- **130,000 richly annotated object assets**
- **42,000,000 stable grasp annotations**

---

## Components

### 1. Environment Library

**Handcrafted household scenes:** Carefully designed rooms (kitchens, living rooms, offices, bedrooms) with realistic furniture placement and object arrangements. Provides high-quality evaluation scenarios.

**Procedurally generated environments:** Algorithm-generated multi-room houses. Scale: 230k+ unique floor plans, furniture configurations, and object arrangements. Provides training diversity.

**Scene diversity covers:**
- Room types (kitchen, bathroom, office, garage, etc.)
- Lighting conditions (daylight, night, directional/ambient)
- Object density (sparse to cluttered)
- Spatial layouts (open vs. narrow corridors, different room sizes)

### 2. Object Asset Library

**130,000 annotated objects** organized into categories:
- Household items (cups, bottles, tools, electronics)
- Food items (fruits, vegetables, packaged goods)
- Office supplies
- Furniture (manipulable sub-components: drawers, doors)

**Annotation richness:**
- **Semantic category and attributes** (material, color, functional purpose)
- **Geometric properties** (dimensions, center of mass, contact surfaces)
- **48,000 manipulable objects** with **42 million stable grasp poses** pre-computed using standard grasp planners

The 42M pre-computed grasps are a significant practical contribution: computing grasp poses for each object at training time is expensive; having them pre-computed allows direct use during simulation rollouts.

### 3. Simulator Agnosticism

MolmoSpaces environments are exported in formats compatible with:
- **MuJoCo / MJX** (contact-rich physics; fast GPU simulation)
- **Isaac Sim** (NVIDIA; photorealistic rendering; GPU-accelerated)
- **ManiSkill** (manipulation-focused; standard benchmark API)

This allows different research groups to use their preferred simulator while training on the same diverse environment distribution.

### 4. Task Specification System

MolmoSpaces supports the full spectrum of embodied tasks:

| Task Type | Description |
|-----------|-------------|
| **Static manipulation** | Fixed-base arm; grasp, place, push objects |
| **Mobile manipulation** | Mobile base + arm; navigate to object, then manipulate |
| **Navigation** | Navigate to goal locations across rooms |
| **Long-horizon multi-room** | Chain navigation + manipulation across multiple rooms |

Tasks are specified via language instructions ("pick up the blue cup from the kitchen counter and bring it to the dining table") — compatible with VLA evaluation.

### 5. MolmoSpaces-Bench

A **standardized benchmark suite of 8 tasks** for evaluating robot policies:

| Task | Type | Key Challenge |
|------|------|--------------|
| Object Pick | Static manipulation | Precise grasp of diverse objects |
| Object Place | Static manipulation | Accurate placement on varied surfaces |
| Object Stack | Static manipulation | Coordination; grasp + place with clearance |
| Drawer Open | Articulated manipulation | Handle grasp + pull direction |
| Mobile Pick | Mobile manipulation | Navigation + grasp |
| Mobile Place | Mobile manipulation | Navigation + precise placement |
| Room Navigation | Navigation | Long-distance goal reaching |
| Multi-Room Task | Long-horizon | Navigate + manipulate across rooms |

---

## Sim-to-Real Correlation: The Key Validation

The most important result in the MolmoSpaces paper is the **sim-to-real correlation analysis**:

### Methodology
- Evaluate multiple policies (different VLA models, different training stages) on MolmoSpaces-Bench tasks *in simulation*
- Deploy the same policies on a real robot and measure real-world success rates on the same tasks
- Compute correlation between simulation and real-world performance across policies

### Results

```
Pearson correlation R = 0.96
Spearman rank correlation ρ = 0.98
```

**Interpretation:** A policy that performs 10% better than another in simulation will reliably perform better in the real world too. The *relative ranking* of policies is almost perfectly preserved across the sim-to-real gap.

This validates MolmoSpaces as a **reliable proxy for real-world evaluation** — researchers can iterate on policy development using simulation alone and trust that simulation rankings predict real-world rankings.

### Why This Matters

Previously, simulation evaluation in robotics was treated with skepticism: a policy might be #1 in simulation but #5 in the real world due to the visual and physics gap. MolmoSpaces' high sim-to-real correlation (achieved via domain randomization + realistic physics + VLM visual features) resolves this — making simulation a trustworthy evaluation platform.

---

## Key Sensitivity Findings

MolmoSpaces-Bench experiments identify three major failure modes of current VLA policies:

**1. Prompt phrasing sensitivity**
The same task described differently causes large performance differences:
- "Pick up the cup" → 72% success
- "Grasp the mug" → 61% success
- "Take the drinking vessel" → 43% success
All three describe the same action, but VLAs respond inconsistently to phrasing variation.

**2. Initial joint position sensitivity**
Performance varies significantly based on the robot arm's starting configuration — even for tasks where the starting configuration shouldn't matter. Suggests policies are fitting to specific starting poses in training data.

**3. Camera occlusion sensitivity**
Objects partially hidden by the robot body or other objects cause disproportionate failures, even when enough of the object is visible for a human to identify it.

These findings provide concrete directions for future VLA improvement.

---

## Evaluation

### Policies Evaluated
- Molmo family ([[algo-molmobot]], [[algo-molmoact2]])
- Other VLA baselines (unnamed in publicly available information)
- Ablations: different training data scales, different simulation environments

### Key Result: Scale Matters
- Newer, larger policies trained on more MolmoSpaces data outperform older versions in both simulation and real-world — confirming that simulation scale translates to real-world improvement

---

## Pros

- **Unprecedented scale** — 230k environments, 130k objects, 42M grasps; far larger than any prior open simulation dataset
- **Simulator-agnostic** — works with MuJoCo, Isaac, ManiSkill; not locked to one simulator
- **High sim-to-real correlation** — R=0.96; reliable proxy for real-world evaluation
- **Full task spectrum** — static manipulation, mobile manipulation, navigation, long-horizon
- **Open ecosystem** — all assets, code, and benchmarks released publicly
- **Pre-computed grasps** — 42M grasp annotations eliminate expensive per-object grasp planning

## Cons

- **Scene fidelity** — even 230k environments may not cover all real-world scene types (industrial settings, outdoor environments, hospital rooms)
- **Prompt sensitivity unresolved** — identifies the problem but doesn't solve it
- **Deformable object limitation** — like all rigid-body simulators, soft objects (cloth, cables) are not well represented
- **Benchmark saturation risk** — as policies improve, the 8-task benchmark may become too easy (needs expansion)
- **Computational requirements** — generating and storing 230k environments requires significant infrastructure; reproducing the full dataset requires substantial storage

---

## In This Wiki

MolmoSpaces: [[simulation-and-tools]], [[world-models]].
Enables: [[algo-molmobot]] (trained on MolmoSpaces), [[algo-molmoact2]] (evaluated on MolmoSpaces-Bench).
Foundation: [[algo-molmo2]] (policies use Molmo2 backbone).
Compare with: MuJoCo Playground ([[simulation-and-tools]] — tool focus; not object diversity), MARTENS ([[simulation-and-tools]] — adversarial testing rather than diversity), Multi-Goal RL benchmark ([[reinforcement-learning]] — sparse reward tasks; smaller scale).
