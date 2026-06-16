# Algorithm: OpenVLA

**Paper:** "OpenVLA: An Open-Source Vision-Language-Action Model" — Kim, Pertsch, Karamcheti et al. (Stanford / UC Berkeley / Google, 2024)
**Category:** Vision-Language-Action Model (VLA) — generalist robot policy
**Related:** [[vision-language-action-models]], [[llms-for-robotics]], [[algo-octo]], [[algo-vla-rl]]

---

## The Problem It Solves

Prior VLA models (RT-2, RT-X) are closed, proprietary, and have 55B+ parameters — inaccessible to most researchers and too large for efficient fine-tuning. Meanwhile, open-source robot policies (like Octo) don't leverage the rich visual and linguistic knowledge of large pretrained VLMs.

**OpenVLA goal:** A fully open-source 7B-parameter VLA that:
1. Outperforms closed models (RT-2-X, 55B) with 7× fewer parameters
2. Is fine-tunable to new robots and tasks via consumer GPU-compatible methods (QLoRA)
3. Serves efficiently via quantization without accuracy loss

---

## How It Works

### Architecture

OpenVLA is built on three components:

**1. Visual Encoder — Dual-encoder fusion:**
```
Image → DINOv2 features  ─┐
Image → SigLIP features   ├─→ Feature fusion → Visual tokens
```
- **DINOv2:** Self-supervised ViT pretrained on internet images; excels at spatial/structural understanding
- **SigLIP:** Contrastive VLM pretrained on image-text pairs; excels at semantic matching between language and visual concepts
- The fused features provide complementary representations: spatial precision (DINOv2) + language grounding (SigLIP)

**2. Language Model Backbone — Llama 2 (7B):**
- Processes the fused visual tokens + language instruction as a sequence
- Generates output tokens autoregressively
- The visual tokens are projected to the Llama embedding space via a learned MLP projector (similar to LLaVA architecture)

**3. Action Tokenizer:**
Actions (continuous joint positions or end-effector deltas) are **discretized** into tokens:
- Each continuous action dimension is binned into 256 discrete values
- A 7-DoF arm with 6 action dimensions = 7 tokens per action step
- These action tokens are part of the Llama vocabulary (special tokens added)

**Inference:**
```
Prompt: "Pick up the apple and place it in the bowl."
         + encoded camera image(s)
→ Llama 2 generates: action token 1, action token 2, ..., action token 7
→ De-tokenize → continuous joint position delta
```

### Training Data

**Open X-Embodiment dataset:** A large-scale community dataset aggregating robot demonstrations across many labs and robot types.
- **970,000 real-world robot demonstrations**
- Covers 7 different robot types (WidowX, Franka, UR5, Google Robot, etc.)
- Task diversity: grasping, placing, pushing, opening drawers, pouring

OpenVLA was trained on this full dataset with a standard language modeling cross-entropy loss on action tokens.

### Fine-Tuning Protocols

**Full fine-tuning:** Update all 7B parameters — expensive but maximally flexible.

**QLoRA fine-tuning:**
- Quantize the base model to 4-bit (using NF4 quantization from QLoRA)
- Add small LoRA adapters (rank 32) to attention layers
- Only the adapters are trained (~2% of parameters)
- Achieves near full fine-tuning performance on a single A100 GPU in ~1-2 hours

**Quantized inference:** After fine-tuning, quantize to 4-bit or 8-bit for efficient deployment. No loss in task success rate at 8-bit quantization.

---

## Why It Works

### Pre-trained VLM Knowledge

The Llama 2 + DINOv2 + SigLIP backbone is pretrained on internet-scale data. This provides:
- **Strong language grounding:** "red apple" activates visual features for red, apple-shaped objects
- **Spatial understanding:** "to the left of the bowl" parsed correctly via DINOv2's spatial features
- **Instruction following:** Llama 2's language model training means complex task descriptions are parsed naturally

### Action Tokenization Enables Language Model Machinery

By discretizing actions into tokens in Llama's vocabulary, OpenVLA can use the language model's attention mechanism to jointly reason over language instructions, visual observations, and actions. The action generation is just another sequence-to-sequence prediction task.

### Data Scale and Diversity

970k demonstrations across 7 robot types means the model has seen many variations of task-relevant behavior. This breadth enables strong generalization compared to models trained on narrower datasets.

---

## Evaluation

### Benchmark: WidowX (BridgeV2 tasks)
**29 tasks** from the BridgeV2 dataset, evaluated across multiple robot embodiments. Tasks: pick-and-place, stacking, drawer manipulation, object rearrangement.

### Comparison

| Model | Parameters | Task Success Rate (29 tasks) |
|-------|-----------|------------------------------|
| RT-2-X | 55B | 62.5% |
| Octo | ~90M | 50.1% |
| OpenVLA | **7B** | **79.0%** |

**OpenVLA outperforms RT-2-X by +16.5% absolute with 7× fewer parameters.**

### Fine-Tuning Results
**Multi-task setting** (multiple objects, requiring strong language grounding):
- OpenVLA fine-tuned: 74% success
- Diffusion Policy from scratch: 53.6%
- OpenVLA → **+20.4% over Diffusion Policy**

**Language grounding:** OpenVLA correctly identifies target objects from language descriptions in scenarios with multiple distractors — outperforming models trained on narrower data.

**QLoRA fine-tuning efficiency:**
- Full fine-tuning: 8× A100s, ~4 hours
- QLoRA fine-tuning: 1× A100, ~1.5 hours
- Performance difference: <2% on most tasks

### Inference
- Runs at ~6-9 Hz with greedy decoding on a single A100 (without quantization)
- 4-bit quantization: ~6Hz on consumer GPUs (RTX 3090)

---

## Pros

- **Open source, fully released** — weights, code, fine-tuning notebooks; reproducible
- **Outperforms much larger closed models** — beats RT-2-X (55B) by 16.5%
- **Fine-tunable on consumer hardware** — QLoRA brings the barrier to entry way down
- **Strong language grounding** — dual visual encoder + Llama 2 enables precise instruction following
- **Built on standard components** — Llama 2, DINOv2, SigLIP; easily updated when better VLMs appear

## Cons

- **Slow inference** (~6-9 Hz) — autoregressive token generation is inherently slow; too slow for high-frequency control tasks (>25Hz)
- **Action discretization** — binning continuous actions to 256 bins loses precision; fails on tasks requiring <1mm accuracy
- **Fixed context length** — the visual input is a single image (or few frames); limited temporal context compared to video-based approaches
- **7B parameters is still large** — not deployable on edge devices or robots without cloud inference
- **Training cost** — initial training required large-scale GPU clusters; fine-tuning is accessible but initial pretraining is not

---

## OpenVLA as a Foundation

OpenVLA's architecture and training setup have been extended in several ways:
- **VLA-RL** ([[algo-vla-rl]]): Online RL fine-tuning of OpenVLA to surpass offline performance
- **TinyVLA** ([[vision-language-action-models]]): Smaller, faster variant; replaces autoregressive decoding with diffusion decoder
- **SmolVLA** ([[vision-language-action-models]]): Community-focused small VLA; single-GPU training

---

## In This Wiki

OpenVLA: [[vision-language-action-models]], [[llms-for-robotics]].
See also: [[algo-octo]] (another open generalist policy, smaller), [[algo-vla-rl]] (RL fine-tuning of OpenVLA), [[algo-ppo]] (RL component in VLA-RL).
