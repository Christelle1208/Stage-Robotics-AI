# Model: MolmoB0T — Zero-Shot Manipulation via Large-Scale Simulation

**Paper:** "MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation" — 2026
**Category:** Simulation-trained policy for zero-shot real-world manipulation
**Related:** [[world-models]], [[vision-language-action-models]], [[simulation-and-tools]], [[algo-molmo2]], [[algo-molmospaces]]

---

## What It Is

MolmoB0T is a robot manipulation policy that achieves **zero-shot real-world manipulation** — without any real-world training data — by leveraging large-scale simulation in the [[algo-molmospaces]] ecosystem and the [[algo-molmo2]] VLM backbone.

The name encodes the approach: **Molmo** (the foundation VLM) + **B0T** (bot, trained on simulation). The "zero-shot" claim means the policy is trained entirely in simulation and deployed on real robots without any real-world fine-tuning.

---

## The Core Thesis: Simulation Scale > Real-World Data

The dominant paradigm for robot learning requires real-world demonstration data (OpenVLA: 970k real demos; Octo: 800k real demos). This data is expensive to collect and hard to scale.

**MolmoB0T's bet:** If simulation environments are sufficiently diverse and realistic, a policy trained purely on simulated data can generalize to real robots — *without any real-world fine-tuning*.

This requires:
1. **Massive simulation diversity** — enough variation in objects, scenes, lighting, and tasks to cover real-world distribution
2. **Sim-to-real robustness** — the VLM backbone (Molmo2) trained on real internet images provides visual robustness; the policy doesn't over-fit to simulation visuals
3. **Realistic physics** — contact dynamics in simulation must match real interactions well enough to transfer

---

## How It Works

### Simulation Training Pipeline

**Step 1: Environment generation (MolmoSpaces)**
Using the [[algo-molmospaces]] ecosystem:
- Sample from 230,000+ diverse indoor environments
- Populate with objects from the 130,000-asset library (with pre-computed 42M stable grasps)
- Randomize: lighting conditions, object positions, camera angles, textures, robot configurations

This **domain randomization** strategy forces the policy to learn representations that work across a wide range of conditions — the policy can't rely on specific textures, lighting, or object positions, so it must learn underlying structure.

**Step 2: Policy training**

The policy is structured as a VLA fine-tuned from Molmo2:
```
Simulated camera image + language task description
→ Molmo2 VLM backbone (frozen or lightly fine-tuned)
→ Action head (fine-tuned on simulation data)
→ Robot action (joint deltas or EEF targets)
```

Training uses behavior cloning on successful simulation trajectories. The large-scale simulation generates these trajectories automatically using scripted planners (since simulation provides privileged access to object positions for planning).

**Step 3: Zero-shot real-world deployment**

The trained policy is deployed directly on a real robot:
- No real-world fine-tuning
- The Molmo2 backbone's internet-trained visual representations bridge the visual sim-to-real gap
- The policy generalizes because it has seen millions of simulated variations

### Why Molmo2 Enables the Sim-to-Real Jump

Standard simulation-trained policies fail on real robots because simulation visuals look different from real images (the "visual sim-to-real gap"). MolmoB0T sidesteps this by using [[algo-molmo2]] as the visual backbone:

- Molmo2 was pretrained on **internet images** — diverse real-world photos and videos
- The visual representations it learned are therefore robust to the visual differences between simulation and reality
- Only the action head needs to learn manipulation-specific behavior; the visual features transfer automatically

---

## Why It Works

### The Scale Argument

More simulation diversity → better coverage of real-world states → better generalization. MolmoSpaces provides:
- 230k environments (vs. typically <1k in prior simulation-based training)
- 130k object assets (vs. typically hundreds)
- 42M stable grasps (pre-computed; no expensive grasp search at training time)

This scale makes it statistically likely that the training distribution covers most real-world configurations the robot will encounter.

### The VLM Backbone Argument

The visual gap between simulation and reality is the main failure mode of sim-to-real transfer. By using a VLM pretrained on real-world images (Molmo2), MolmoB0T uses visual representations that never saw simulation at all during the visual pretraining phase — so the visual backbone is inherently real-world robust.

### Domain Randomization + Realistic Physics

The MolmoSpaces simulator supports MuJoCo, Isaac Sim, and ManiSkill — all physics engines with realistic contact dynamics. Combined with domain randomization of non-physical parameters (lighting, texture, object positions), the policy learns to generalize physically (grasps that work across object shapes) and visually (recognizes objects across appearances).

---

## Evaluation

### Benchmark: MolmoSpaces-Bench

MolmoSpaces-Bench provides 8 manipulation tasks across diverse simulated environments. Key measurement: **sim-to-real correlation**.

**Sim-to-real correlation:**
- Pearson R = 0.96 between simulated and real success rates
- Spearman ρ = 0.98
- Interpretation: a model that does well in simulation reliably does well in reality — making simulation a valid proxy for real-world evaluation

**Key findings:**
- Newer, stronger zero-shot policies (MolmoB0T with latest Molmo2) outperform earlier versions in both simulation and reality
- Identified key sensitivities:
  - **Prompt phrasing:** Different wordings of the same instruction cause large performance swings
  - **Initial joint positions:** Starting configuration matters significantly for grasp success
  - **Camera occlusion:** Objects partially occluded by robot body are harder to grasp

### Zero-Shot Transfer Results
MolmoB0T achieves meaningful zero-shot performance on real manipulation tasks:
- Simple pick-and-place: **>70% success rate** zero-shot (without any real-world fine-tuning)
- Novel object types (not in simulation library): performance degrades but remains above random
- With minimal real-world adaptation (10-20 demos): performance matches or exceeds models trained purely on real data

---

## Pros

- **Zero-shot generalization** — no real-world data required for initial deployment
- **Scalable training** — simulation data is free; can train on orders of magnitude more data than real-world collection allows
- **Sim-to-real reliability** — high correlation (R=0.96) means simulation evaluation is a valid proxy
- **Diverse object generalization** — 130k simulated objects gives broad object coverage
- **Open ecosystem** — MolmoSpaces assets are open; community can add objects and environments

## Cons

- **Precision limitations** — zero-shot sim-to-real has reduced precision compared to real-world-fine-tuned policies; acceptable for simple grasps, insufficient for precision assembly
- **Physics gap not fully closed** — deformable objects (cloth, cables), fluids, and soft contacts remain hard to simulate accurately; policies fail on these
- **Prompt sensitivity** — significant performance variation with instruction phrasing; requires prompt engineering in deployment
- **Camera occlusion failures** — partially visible objects cause disproportionate failures
- **Long-horizon tasks** — zero-shot transfer is hardest for multi-step manipulation sequences; error compounds across steps

---

## MolmoB0T in the Zero-Shot Landscape

MolmoB0T is part of a broader trend toward zero-shot robot policies:
- [[world-models]] — "World Action Models are Zero-Shot Policies": world models as zero-shot policies
- MolmoB0T — simulation scale as zero-shot foundation
- OpenVLA (with adaptation) — strong pretrained policies that need minimal adaptation

The approaches are complementary: world models rely on learned dynamics; MolmoB0T relies on simulation diversity; both aim to reduce the real-world data burden.

---

## In This Wiki

MolmoB0T: [[world-models]], [[simulation-and-tools]], [[vision-language-action-models]].
Foundation: [[algo-molmo2]], [[algo-molmospaces]].
Related zero-shot approaches: World Action Models ([[world-models]]), [[algo-octo]] (generalist that enables fast adaptation).
Compare with: [[algo-openvla]] (real-world demos required; higher peak performance), [[algo-hitl-rl]] (human corrections to improve zero-shot → expert policy).
