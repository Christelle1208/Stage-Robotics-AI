# Model: Molmo2 — Open Vision-Language Model with Video Understanding

**Paper:** "Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding" — 2026
**Category:** Open-weights Vision-Language Model (VLM) — foundation for the Molmo robotics ecosystem
**Related:** [[llms-for-robotics]], [[vision-language-action-models]], [[algo-molmoact2]], [[algo-molmobot]]

---

## What It Is

Molmo2 is a **fully open-weights Vision-Language Model (VLM)** — the foundation layer of the Molmo robotics ecosystem. It extends prior VLMs (like LLaVA, InstructBLIP) with two key capabilities absent from most open models:

1. **Video understanding:** Processes temporal sequences of frames, not just single images
2. **Visual grounding:** Localizes objects in images that are referenced by text (predicts bounding boxes or point coordinates for described entities)

These capabilities are specifically chosen to enable robot applications: robots observe the world as video streams, and manipulation requires knowing *where* specific objects are.

---

## How It Works

### Architecture

Molmo2 follows the standard VLM architecture: a visual encoder + language model backbone + cross-modal projection layer. The key distinctions:

**Visual Encoder:**
- Processes both single images and video frames (temporal sequence)
- Uses a ViT-based backbone that can accept multiple frames as a sequence
- Frame features are pooled or attended across time for video understanding

**Language Model Backbone:**
- A transformer LLM (similar scale and architecture to Llama/Mistral families)
- Processes interleaved text and visual tokens
- Generates text responses autoregressively

**Grounding Head:**
- An additional output head that predicts pixel coordinates when asked to localize objects
- Outputs normalized (x, y) point coordinates or bounding boxes
- Activated when the prompt asks "where is the [object]?" or "point to the [object]"

**Cross-Modal Projection:**
- An MLP that maps visual token embeddings into the language model's token space
- Allows the language model to "see" images as just another type of input token

### Open Weights and Data

The key contribution of Molmo2 is its **full openness**:
- **Open weights:** Model checkpoints released under a permissive license
- **Open training data:** The vision-language dataset used for training is released publicly, enabling:
  - Reproduction of results
  - Fine-tuning on custom data
  - Understanding of model biases and capabilities
  - Building derivative models (MolmoAct2, MolmoB0T)

This contrasts with closed models like GPT-4V, Claude 3 Vision, and Gemini Pro Vision, which release no weights or training data.

### Training

Molmo2 is trained in two stages:
1. **Pretraining:** Large-scale multimodal pretraining on image-text pairs + video-text pairs + grounding annotations (bounding box supervision)
2. **Instruction tuning:** Fine-tuning on instruction-following datasets including visual QA, video description, grounding tasks, and robot-relevant tasks

The grounding data is particularly important for robotics: annotations of "the red cup is at (0.43, 0.67)" teach the model to localize objects that will later be manipulation targets.

---

## Why It Matters for Robotics

### Visual Grounding → Manipulation Target Localization

When a robot receives the instruction "pick up the green bottle," it needs to localize the green bottle in its camera image. Molmo2's grounding head directly outputs pixel coordinates for described objects — a natural interface for the pick-and-place pipeline:

```
Camera image + "where is the green bottle?" 
→ Molmo2 
→ (x=0.42, y=0.65)  [normalized image coordinates]
→ 3D point via depth → robot grasp target
```

### Video Understanding → State Tracking

Manipulation often requires tracking state over time: "did I successfully grasp the object?" "Is the lid still on the container?" Video understanding allows Molmo2 to reason about trajectories of observations, not just snapshots.

### Foundation for the Molmo Ecosystem

Molmo2 is the base for the entire Molmo robotics stack:
- [[algo-molmoact2]] fine-tunes Molmo2 for action reasoning (directly predicting robot actions)
- [[algo-molmobot]] uses Molmo2 policies trained on simulation data for zero-shot manipulation
- [[algo-molmospaces]] (MolmoSpaces benchmark) uses Molmo-family policies as the evaluation target

---

## Evaluation

### VLM Benchmarks
Molmo2 is evaluated on standard VLM benchmarks:
- **VQAv2:** Visual question answering on images
- **MSVD/MSRVTT:** Video question answering
- **RefCOCO:** Referring expression comprehension (grounding)
- **Object Detection mAP:** Localization accuracy

Results: Molmo2 achieves competitive or SOTA performance among open-weights models on video understanding and grounding tasks, while being fully open (weights + data).

### Robotics-Specific Evaluation
Via the downstream [[algo-molmoact2]] and [[algo-molmobot]] models:
- **Zero-shot manipulation success rate** (MolmoB0T): Molmo2 foundation enables zero-shot P&P
- **Sim-to-real correlation** (MolmoSpaces): R=0.96 between simulation and real-world performance

---

## Pros

- **Fully open** — weights + data; reproducible; fine-tunable by anyone
- **Video understanding** — temporal reasoning over observation sequences; not just single frames
- **Visual grounding** — directly localizes objects; natural interface for manipulation
- **Strong ecosystem** — foundation for MolmoAct2, MolmoB0T, MolmoSpaces
- **Community extensible** — open data means researchers can add robotics-specific supervision

## Cons

- **Grounding precision** — point/box predictions have limited spatial precision; may not meet sub-centimeter accuracy needed for high-precision manipulation
- **Video processing cost** — processing video frames is more expensive than single images; may limit real-time control rate
- **Robot-specific generalization gap** — trained primarily on internet-style video; robot egocentric views differ in perspective, motion blur, and scene structure
- **Action generation requires fine-tuning** — Molmo2 alone generates text; robot actions require [[algo-molmoact2]] fine-tuning

---

## In This Wiki

Molmo2: [[llms-for-robotics]], [[vision-language-action-models]].
Downstream uses: [[algo-molmoact2]] (action fine-tuning), [[algo-molmobot]] (zero-shot manipulation), [[algo-molmospaces]] (benchmark ecosystem).
Compare with: [[algo-openvla]] (OpenVLA also uses a VLM backbone; different architecture choices), [[algo-octo]] (action-focused generalist without strong VLM backbone).
