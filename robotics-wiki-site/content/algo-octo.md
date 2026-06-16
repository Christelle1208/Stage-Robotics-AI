# Algorithm: Octo

**Paper:** "Octo: An Open-Source Generalist Robot Policy" — Octo Model Team (UC Berkeley / Stanford / Google DeepMind, 2024)
**Category:** Transformer-based generalist robot policy; pre-trainable + fine-tunable
**Related:** [[vision-language-action-models]], [[algo-openvla]], [[algo-vla-rl]]

---

## The Problem It Solves

Robot learning suffers from a fundamental data bottleneck: each lab trains a policy for *their* specific robot, with *their* specific cameras, on *their* specific tasks — and none of this transfers. The robotics community has accumulated enormous datasets across labs (the Open X-Embodiment dataset), but no open model existed that could leverage this data to provide a strong universal starting point.

**Octo goal:** Be the "GPT-2 moment" for robot policies — a pre-trained open-source generalist that any researcher can fine-tune to their setup without training from scratch.

---

## How It Works

### Architecture: Modular Transformer

Octo's design prioritizes **flexibility and fine-tunability** above all else. It uses a modular Transformer architecture:

```
Input:
  Task: [language tokens] or [goal image tokens]
  Observations: [image tokens from multiple cameras]
  
Processing:
  Readout tokens (learnable, one per output modality)
  
Self-attention across all tokens (task + obs + readout)

Output:
  Readout token embeddings → diffusion head → action sequence
```

**1. Input tokenization**

*Task specification:* Octo handles two types of task specification, encoded as token sequences:
- **Language:** "Pick up the can and place it in the pot" → tokenized via a small language encoder (FiLM-conditioned, not a full LLM)
- **Goal image:** A target image showing the desired end state → tokenized via ViT

*Observations:* Camera images from one or more cameras → ViT patch tokens. Octo's architecture supports a **variable number of cameras** (the attention mechanism handles different numbers of input tokens naturally).

*Readout tokens:* Learnable "query" tokens that collect information from the full context for downstream action prediction.

**2. Transformer processing**

All tokens (task + observations from history H + readout tokens) are processed jointly via **causal self-attention**. The Transformer can attend across all modalities and time steps.

Octo's causal masking preserves the time-ordering of observations: at step t, the model sees observations from steps t-H to t, but not future steps.

**3. Action head: Diffusion**

Octo's action head is a **diffusion process** (like Diffusion Policy):
```
Readout tokens → Diffusion head → action sequence (W future steps)
```
Actions are continuous (joint velocities or end-effector deltas).

The diffusion head conditions on the readout token embeddings and denoises a Gaussian action sample into a coherent action chunk. This provides multimodal action distributions and handles multiple robot action spaces.

### Training Data

**Open X-Embodiment dataset:**
- **800,000 trajectories** from multiple research labs and robot types
- 22 robot embodiments: WidowX, Franka, Google Robot, UR5, RT-X, Jaco, Kuka IIWA, etc.
- Diverse tasks: grasping, stacking, drawer manipulation, object rearrangement, navigation
- Multiple control modalities: joint velocity, EEF (end-effector) position, delta EEF

Training with standard behavior cloning (MSE on diffusion-denoised actions or transformer cross-entropy).

### Fine-Tuning Design

Octo's modularity is designed for three types of fine-tuning:

**Type 1: New observation space** (different cameras)
- Patch-embed the new camera images
- Fine-tune the ViT observation tokenizer

**Type 2: New action space** (different control modality or DoF)
- Train a new diffusion head from scratch; keep the Transformer backbone frozen
- Only a few thousand demonstrations needed

**Type 3: New task** (new language instructions or goal images)
- Fine-tune the full model (or just the task encoder + action head) on task-specific data

Fine-tuning reported to complete in **a few hours on standard consumer GPUs** (e.g., 2× RTX 3090).

---

## Why It Works

### Pre-training Amortizes Cost

By training on 800k diverse demonstrations once, Octo learns a rich shared representation of robot behavior. Fine-tuning to a new robot/task only needs to adapt this representation, not learn from scratch. The pre-trained backbone already "knows" how to move robot arms, grasp objects, and follow instructions.

### Flexible Architecture Handles Diversity

Standard architectures struggle with variable cameras, action spaces, and task types. Octo's modular design — tokenize everything, process with self-attention — naturally handles this diversity. The diffusion head is inherently continuous and dimension-flexible.

### Open X-Embodiment = Community-Scale Data

No single lab can collect 800k diverse demonstrations, but the combined robotics community can. The Open X-Embodiment dataset (led by the RT-X effort) makes this scale possible. Octo is the first open model to train on this full dataset.

---

## Evaluation

### Platforms (9 evaluated)

Octo was tested on 9 different robot setups including:
- **WidowX** (same as BridgeV2 training distribution)
- **Franka Panda** (out-of-distribution; fine-tuned)
- **Google Robot** (in-distribution; same data as RT-X)
- **Simulated environments** (MuJoCo-based)
- **Various camera configurations** (single vs. wrist camera)

### Comparison

| Model | Parameters | WidowX Tasks | Fine-tuning Time |
|-------|-----------|--------------|-----------------|
| BC (single task) | ~10M | 52% | Hours per task |
| RT-2-X | 55B | 62% | Days (GPU clusters) |
| OpenVLA | 7B | 79% | 1.5h (QLoRA) |
| **Octo** | **~90M** | **50%–65%** | **Few hours** |

**Note:** Octo is much smaller (~90M vs. 7B for OpenVLA) and primarily evaluated on fine-tuning versatility rather than raw performance on a single benchmark.

### Key Fine-Tuning Results

**New observation space (wrist camera added):**
- Octo + new camera fine-tuning: +15-25% over Octo without camera
- Demonstrates modular adaptation to new sensors

**New action space (delta EEF → joint velocity):**
- Fine-tune action head only: recovers 85-90% of full fine-tuning performance
- Minimal data needed (few hundred demos)

**Zero-shot** (no fine-tuning):
- Octo out-of-the-box on a Franka (not in training data): ~30-40%
- After 2h fine-tuning: ~60-70%

### Ablations
- **Remove pre-training (train from scratch on fine-tuning data):** -20-30% across tasks
- **Language vs. goal image conditioning:** Language conditioning generalizes better to unseen task variations
- **History length H:** H=2-3 frames optimal; longer history provides marginal gains at extra compute

---

## Pros

- **Small and efficient** — ~90M parameters; fine-tunable on a single consumer GPU in hours
- **Highly modular** — handles new cameras, new action spaces, new tasks independently
- **Open source** — weights, code, and training pipeline fully released; reproducible
- **Strong fine-tuning baseline** — pre-trained Octo routinely outperforms training from scratch
- **Diffusion action head** — handles multimodal action distributions and continuous action spaces of any dimension

## Cons

- **Lower raw performance than OpenVLA** — 7B OpenVLA outperforms 90M Octo in head-to-head comparisons (expected: scale matters)
- **Not a language model** — uses a simple language encoder, not a full LLM; less sophisticated language understanding than OpenVLA/Llama2
- **Limited generalization** — despite 800k demos, still fine-tuning needed for reliable performance on new setups
- **Image-based only** — no point cloud or 3D input; limited spatial precision
- **Slower than non-diffusion policies** at inference (same tradeoff as Diffusion Policy)

---

## Octo vs. OpenVLA

| | Octo | OpenVLA |
|-|------|---------|
| Size | ~90M | 7B |
| LLM backbone | No (small encoder) | Yes (Llama 2) |
| Language understanding | Limited | Strong |
| Fine-tuning cost | Hours, single GPU | Hours, single GPU (QLoRA) |
| Action head | Diffusion | Discrete tokens |
| Performance on known robots | ~50-65% | ~79% |
| Modularity | High (designed for it) | Moderate |

**Recommendation:** Use Octo when you need a lightweight, quick-to-fine-tune baseline or a modular architecture; use OpenVLA when language grounding and peak task performance matter.

---

## In This Wiki

Octo: [[vision-language-action-models]], [[simulation-and-tools]].
Compare with: [[algo-openvla]] (larger, stronger language; same open-source ethos), [[algo-vla-rl]] (RL fine-tuning approach applicable to both).
