# Algorithm: CLIPORT — What and Where Pathways for Robotic Manipulation

**Paper:** "CLIPORT: What and Where Pathways for Robotic Manipulation" — Shridhar, Manuelli, Fox (CoRL 2021)
**Category:** Language-conditioned imitation learning — two-stream architecture combining a semantic ("what") and spatial ("where") pathway for pick-and-place affordances
**Related:** [[imitation-learning]], [[vision-language-action-models]], [[pick-and-place]], [[grasping-and-manipulation]], [[algo-act]], [[algo-smolvla]]

---

## The Problem It Solves

Two families of manipulation policy each had a piece of the puzzle but not both:

- **End-to-end manipulation networks** (e.g. Transporter) learn precise spatial pick/place actions from demonstrations, but have no notion of semantics — switching from "pack the red pen" to "pack the blue pen" requires collecting an entirely new dataset.
- **Internet-pretrained vision-language models** (e.g. CLIP) have broad semantic understanding of categories, colors, shapes, and text from millions of image-caption pairs, but lack the fine-grained spatial precision needed to actually place an end-effector.

**CLIPORT's goal:** combine both — a language-conditioned imitation-learning agent that grounds abstract semantic concepts ("pack the scissors", "fold the cloth in half") in precise pick-and-place actions, without any object detectors, instance segmentation, pose estimation, or symbolic state.

---

## How It Works

### Two-step pick-and-place primitive (inherited from Transporter)

Every action is a pair of end-effector poses: `a = (T_pick, T_place) ∈ SE(2)`. Given a top-down RGB-D orthographic heightmap `o_t` and a language instruction `l_t`:

1. A pick network `f_pick` outputs a dense per-pixel "where to pick" heatmap; `T_pick = argmax Q_pick(u,v)`.
2. A query network `Φ_query` crops the image around `T_pick`, and a key network `Φ_key` encodes the full image. Cross-correlating query and key features (over `k=36` discrete rotations) produces a "where to place" heatmap; `T_place = argmax Q_place(Δτ)`.

This action-centric formulation — detect *actions*, not *objects* — is translationally equivariant by design and is what made the original Transporter data-efficient.

### Two-stream (semantic + spatial) architecture

CLIPORT extends all three FCNs (`f_pick`, `Φ_query`, `Φ_key`) to **two parallel pathways**, loosely inspired by the two-stream hypothesis in vision neuroscience:

- **Spatial ("dorsal") stream** — a tabula-rasa ResNet (identical to Transporter), encodes the RGB-D heightmap into dense features. This is where spatial precision lives.
- **Semantic ("ventral") stream** — a **frozen, pretrained CLIP ResNet50** encodes the RGB image, with added decoder layers that upsample to match the spatial stream's resolution.

**Language conditioning:** the instruction is encoded with CLIP's transformer sentence encoder into a goal embedding `g_t`. This embedding is downsampled and *tiled* spatially, then combined with the semantic stream's decoder features via an **element-wise (Hadamard) product** — reusing CLIP's contrastive image-text alignment while preserving spatial structure. This conditioning is repeated for 3 decoder layers (LingUNet-style), with skip connections from the CLIP encoder carrying shape/part/object-level semantics at different depths.

**Fusion:** the spatial stream's features are laterally fused into the semantic stream at each layer (concatenate + 1×1 conv). Final fusion uses addition for `f_pick` and 1×1 conv for `Φ_query`/`Φ_key`.

### Training

Pure imitation learning: cross-entropy loss between predicted pick/place heatmaps and one-hot expert-demonstration pixels. Single-task models train for 200K iterations (~2 days, 1 GPU); multi-task models (10 tasks) train 3× longer (600K iterations, ~6 GPU-days), sampling a random task then a random transition from that task's data each step.

---

## Why It Works

- **CLIP supplies the "what" for free.** Because CLIP was pretrained on internet-scale image-caption pairs, it already has priors for colors, shapes, categories, and text — concepts CLIPORT never has to learn from scratch from a handful of robot demos.
- **Transporter supplies the "where".** The two-stage pick-then-place formulation with cross-correlation is sample-efficient and translationally equivariant, giving the spatial precision CLIP alone lacks.
- **Tiling + Hadamard product preserves both alignment and spatial layout** — the language goal modulates *where* in the image to attend without collapsing the spatial map to a single vector (which would destroy the precision CLIPORT needs for placement).
- **Ablation evidence (Figure 3, seen tasks):** Transporter-only (no language) saturates at **50%** — chance-level once language matters. CLIP-only (no spatial stream) saturates at **76%** — has the goal but not the spatial precision for the "last mile". **CLIPORT (single) exceeds 90%** — both streams are necessary, neither is sufficient alone.

---

## Evaluation

### Simulation (Ravens / PyBullet, UR5e suction gripper)
10 language-conditioned tabletop tasks (e.g. `pack-unseen-shapes`, `put-blocks-in-bowls`, `align-rope`, `assembling-kits-seq`, `towers-of-hanoi-seq`, `stack-block-pyramid`), most with **seen** vs **unseen** attribute splits (colors, shapes, objects), evaluated at n = 1, 10, 100, 1000 demonstrations.

| Model | Seen avg. | Notes |
|-------|-----------|-------|
| Transporter-only | saturates ~50% | no language grounding |
| CLIP-only | saturates ~76% | language but no spatial precision |
| **CLIPORT (single)** | **>90%**, ~86% with just 100 demos | both streams |
| **CLIPORT (multi)** | outperforms single-task in **57%** of 72 task/demo-count combinations | one model, 10 tasks |
| CLIPORT (multi-attr) | large gains on unseen attributes (e.g. 45.8→75.7 on `put-blocks-in-bowls-unseen-colors` @ 1000 demos) | trained on seen+unseen splits of *other* tasks |

### Real robot (Franka Panda, 9 tasks)
One multi-task model trained on just **179 total image-action pairs** (5–10 demos/task) achieved **55–75%** success across 9 tasks (stack blocks, put blocks in bowl, pack objects, move rook, fold cloth, read text, loop rope, sweep beans, pick cherries) — simple block manipulation tasks reached ~70%.

**Failure mode observed:** the model can exploit dataset biases instead of grounding language — e.g., if "yellow blocks" only ever appeared with a "blue bowl" in training, the model struggled to place yellow blocks in other-colored bowls until 1-2 counter-examples were added.

---

## Pros

- **No object detectors, segmentation, pose estimation, or symbolic state** — fully end-to-end from pixels + language to actions.
- **Strong few-shot performance**: ~86% average with only 100 demos per task.
- **Multi-task training often *helps*** rather than hurting — a single model for 10 (sim) or 9 (real) tasks is competitive with or better than per-task models, supporting the idea that language lets the model reuse concepts across tasks.
- **Explicit attribute transfer across tasks** (CLIPORT multi-attr) substantially improves generalization to unseen colors/shapes.
- **Validated on real hardware with very little data** (179 image-action pairs total for 9 tasks).

## Cons

- **Limited to the 2-step `(T_pick, T_place) ∈ SE(2)` primitive** — cannot handle dexterous 6-DOF manipulation, multi-fingered hands, or tasks that don't decompose into pick-then-place.
- **CLIP stream is RGB-only** — depth cannot be used in the semantic pathway since CLIP was pretrained on RGB image-caption pairs.
- **No task-completion signal** — the agent doesn't predict when a multi-step task is done.
- **Multi-task training hurts long-horizon tasks** (e.g. `align-rope`) — longer-horizon tasks get less coverage of input-action pairs when sampling is uniform over tasks.
- **Can exploit spurious correlations** in small real-world datasets rather than truly grounding language, requiring deliberate counter-example demos.

---

## Relation to This Project

CLIPORT was **not run as an experiment in this project** — this page is a literature note only (no SO-100 results; contrast with [[algo-act]] and [[algo-smolvla]], which both have "Results in This Project" sections).

It is, however, an important **conceptual precursor** to the VLA models that *were* used ([[algo-act|ACT]], [[algo-smolvla|SmolVLA]]):
- It's one of the earliest demonstrations that a **frozen internet-pretrained vision-language model (CLIP)** can be grafted onto a robot-specific spatial/action network and meaningfully improve semantic generalization — the same basic idea behind SmolVLA's frozen-vs-fine-tuned backbone tradeoffs (see [[algo-smolvla]] "Fine-Tuning SmolVLA — The Options").
- Its **2-step pick/place primitive with dense affordance prediction** is architecturally very different from ACT's transformer action-chunking or SmolVLA's flow-matching action expert — CLIPORT predicts *where* in pixel space, not a continuous joint trajectory. It would not directly apply to the SO-100's continuous-control pick-and-place setup without significant adaptation (the SO-100 tasks in this project are closer to free 6-DOF reach/grasp than CLIPORT's top-down SE(2) suction primitive).
- The **"multi-task training rarely hurts, often helps"** finding (57% of evaluations) is a useful prior if this project ever explores multi-task fine-tuning across SO-100 task variants.

---

## In This Wiki

CLIPORT: [[imitation-learning]] (landmark papers — language-conditioned IL), [[vision-language-action-models]] (precursor to the VLA paradigm), [[pick-and-place]] (2-step pick/place primitive), [[grasping-and-manipulation]] (affordance-based pick prediction).
Compare with: [[algo-act]] (transformer action-chunking IL, no language conditioning), [[algo-smolvla]] (modern VLA — CLIP-style frozen/fine-tuned backbone tradeoffs apply here too), [[algo-openvla]] (later, much larger language-conditioned manipulation model).
