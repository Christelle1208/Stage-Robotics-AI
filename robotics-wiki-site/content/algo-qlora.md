# Algorithm: QLoRA — Quantized Low-Rank Adaptation

**Paper:** "QLoRA: Efficient Finetuning of Quantized LLMs" — Dettmers, Pagnoni, Holtzman, Zettlemoyer (University of Washington, 2023)
**Category:** Parameter-efficient fine-tuning (PEFT) for large language models
**Related:** [[llms-for-robotics]], [[vision-language-action-models]], [[algo-openvla]]

---

## The Problem It Solves

Fine-tuning a 65-billion-parameter language model in full precision (float32 or bfloat16) requires:
- ~65B × 2 bytes = ~130GB just for model parameters
- Additional memory for gradients (~130GB) and optimizer states (Adam: ~260GB)
- **Total: ~520GB GPU memory** — inaccessible to any individual GPU (even A100s are 80GB)

Standard solutions:
- **Model parallelism** across many GPUs — expensive infrastructure
- **Smaller models** — lose capability
- **Prompt engineering / in-context learning** — no weight updates; limited adaptation

**QLoRA goal:** Fine-tune a 65B model on a **single 48GB GPU** with no performance loss compared to full fine-tuning, by combining quantization with low-rank adaptation.

---

## How It Works

QLoRA combines three independent innovations:

### 1. 4-Bit NormalFloat (NF4) Quantization

**Standard quantization:** Represent float32 weights as int8 or int4 — reduces memory 4-8×, but loses precision.

**The problem with standard int4:** Randomly initialized weights are normally distributed N(0, σ²). Standard int4 quantization uses uniform bins (e.g., -8, -7, ..., 7) — these don't match the weight distribution. Most weights cluster near zero, wasting precision there, while extreme values get coarsely represented.

**NF4 (NormalFloat4):** Uses a quantization grid derived from the **quantiles of the standard normal distribution**:

```
NF4 bins: chosen such that each bin contains exactly 1/16 of the probability mass
         of N(0, 1)
```

The 16 bins are: approximately {-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5} (normalized).

**Why NF4 is optimal:** By construction, equal numbers of weights fall into each bin. No bins are wasted on probability mass where there are no weights. This is **information-theoretically optimal** for normally distributed weights.

**Quantization process:**
```python
# For each weight tensor:
1. Normalize by max absolute value: w_norm = w / max(|w|)
   (scales weight to [-1, 1])
2. Quantize to nearest NF4 bin: q = quantize_nf4(w_norm)
3. Store: q (4 bits) + absmax (16 bits, one per block of 64 values)

# Dequantization:
w_approx = dequantize_nf4(q) * absmax
```

Memory: instead of 32 bits per weight → 4 bits per weight + 16 bits/64 weights overhead = **~4.5 bits/weight effective** (~7× reduction).

### 2. Double Quantization (DQ)

The quantization constants (absmax values stored per block) themselves take up memory. For 64-weight blocks, one absmax float32 = 32/64 = **0.5 bits extra per weight** — significant at scale.

**Double quantization:** Quantize the quantization constants themselves:
```
absmax (float32) → quantize absmax to 8-bit integers
```

Since absmax values tend to be smoothly distributed, 8-bit quantization loses almost no precision. This reduces the overhead from 0.5 bits/weight to 0.127 bits/weight — a **~4× reduction in quantization overhead**.

### 3. Paged Optimizers

NVIDIA GPU memory includes a **unified memory** mechanism: if GPU memory is exhausted, data can automatically page to CPU RAM (with a speed penalty). QLoRA uses this for optimizer states:

- Adam optimizer stores momentum and variance for each parameter: 2× model size in float32 = ~520GB for 65B model
- Paged optimizers: store optimizer states in CPU RAM by default; move to GPU only for the gradient update step
- **Effect:** Memory spikes from optimizer states don't cause OOM (out-of-memory) errors; they spill to CPU RAM temporarily

### 4. LoRA (Low-Rank Adaptation)

LoRA is a prior PEFT method (Hu et al., 2021) that QLoRA builds upon:

**Key insight:** LLM weight updates during fine-tuning have **low intrinsic rank** — the update matrix ΔW can be approximated as a product of two small matrices:

```
ΔW = A · B
where A: (d × r), B: (r × k), with r << min(d, k)
```

**LoRA training:**
- Freeze the pretrained weights W₀
- Add trainable low-rank matrices A, B initialized as (A ~ N(0,1), B = 0)
- Forward pass: h = W₀x + (A·B)x = W₀x + ΔWx
- Backpropagate only through A and B

For rank r=16 and dimension d=4096: ΔW has 4096² = 16.7M parameters; A,B have 2×4096×16 = 131k parameters — **127× fewer trainable parameters**.

**QLoRA = 4-bit quantized W₀ + full-precision LoRA adapters A, B:**
```
h = W_NF4 · x + A · B · x
         ↑ frozen 4-bit    ↑ trainable float32/bfloat16
```

Gradients flow through the 4-bit base model using a **straight-through estimator** (treat quantization as identity for backprop), then update only A and B.

---

## Why It Works

### Why Quantization Doesn't Hurt Fine-Tuning

Standard wisdom: quantization degrades performance (bits of precision lost = worse weights). QLoRA's counter-insight: for *fine-tuning*, what matters is the gradient direction, not perfect weight precision.

- The 4-bit quantized base model provides an approximate gradient signal through the straight-through estimator
- The LoRA adapters (in full precision) capture the task-specific update
- The combination recovers full fine-tuning performance because the adapters compensate for quantization errors

### Why Low Rank is Sufficient

Pre-trained LLMs have already learned rich representations. Fine-tuning changes the model's behavior in a *specific direction* relevant to the new task — this directional change is inherently low-rank. The full-rank ΔW has redundant information; rank 16-64 captures the essential adaptation.

---

## Evaluation

### Benchmark: Vicuna

Vicuna is a chatbot evaluation benchmark comparing models to ChatGPT via pairwise human preference.

**Guanaco** — QLoRA-fine-tuned LLaMA models — serves as the primary evaluation:

| Model | Training | GPU Memory | Training Time | Vicuna Score |
|-------|----------|-----------|--------------|-------------|
| Alpaca-65B (full FT) | 8× A100 (40GB), ~weeks | ~600GB | Very long | — |
| GPT-4 | — | — | — | 100% (reference) |
| ChatGPT | — | — | — | ~97% |
| **Guanaco-65B (QLoRA)** | **1× A100 (48GB), 24h** | **~48GB** | **24 hours** | **99.3%** |
| Guanaco-13B | 1× RTX 3090 (24GB), ~12h | ~24GB | 12 hours | 97.8% |
| Guanaco-7B | 1× RTX 2080 (11GB), ~5h | ~11GB | 5 hours | 95.5% |

**Key result:** QLoRA fine-tunes 65B LLaMA on a *single 48GB GPU in 24 hours* to **99.3% of ChatGPT performance** — making SOTA LLM fine-tuning accessible to individual researchers.

### Scale Experiments

QLoRA was used to fine-tune **>1,000 model variants** across:
- Model sizes: 7B, 13B, 33B, 65B
- Model families: LLaMA, T5
- Instruction datasets: 8 different datasets (FLAN, Alpaca, Dolly, etc.)

This scale of experimentation would have been impossible without QLoRA's efficiency.

### Ablations

| Variant | GPU Memory | Vicuna Score |
|---------|-----------|-------------|
| BFloat16 full FT | ~650GB | 100% (reference) |
| 8-bit + LoRA | ~160GB | ~99% |
| 4-bit (standard int4) + LoRA | ~48GB | ~97.5% |
| **4-bit NF4 + LoRA + DQ** | **~48GB** | **~99.3%** |
| 4-bit without DQ | ~52GB | ~99.2% |

**NF4 vs. standard int4:** +1.8% performance at same memory — NF4's optimality matters.

---

## Relevance to Robotics

### Direct application: OpenVLA fine-tuning

The OpenVLA paper explicitly uses QLoRA for fine-tuning their 7B VLA to new robot setups:
- Full fine-tuning: 8× A100 GPUs, ~4 hours
- QLoRA fine-tuning: 1× A100, ~1.5 hours
- Performance: <2% difference from full fine-tuning on most tasks

This makes VLA fine-tuning accessible to individual robotics labs (one A100 ≈ $10k, vs. 8× A100 ≈ $80k+).

### General principle for VLA development

Any VLA based on an LLM (OpenVLA, SmolVLA, TinyVLA) can use QLoRA to fine-tune to:
- New robot types (different arm kinematics → different action spaces)
- New camera configurations (different visual observations)
- New task domains (kitchen, factory, outdoor)

---

## Pros

- **Dramatic memory reduction** — 65B model on 48GB GPU (8× memory reduction)
- **No performance loss** — matches full fine-tuning performance (99.3% on Vicuna)
- **Fast** — single GPU, 24 hours for 65B (vs. weeks with full FT)
- **Off-the-shelf** — implemented in HuggingFace PEFT library; 3 lines of code to enable
- **Accessible** — individual researchers can fine-tune SOTA models on consumer hardware

## Cons

- **Inference overhead** — 4-bit weights must be dequantized during forward pass; ~2× slower inference than float16 (mitigated by quantized inference frameworks like ExLlama, llama.cpp)
- **Quantization precision loss** — though small, 4-bit is lossy; for safety-critical applications, full precision may be required
- **LoRA rank selection** — rank r is a hyperparameter; too small = insufficient expressivity; too large = defeats the purpose
- **Not all layers benefit** — LoRA is typically applied to attention layers only; other layers (feedforward, embeddings) may need different treatment for specialized domains
- **Gradient signal through quantized weights** — straight-through estimator is an approximation; may limit fine-tuning for tasks very different from pretraining

---

## In This Wiki

QLoRA: [[llms-for-robotics]], [[vision-language-action-models]] (OpenVLA fine-tuning).
Compare with: [[algo-openvla]] (primary use case for QLoRA in robotics), [[algo-octo]] (smaller model; less need for QLoRA), [[algo-vla-rl]] (RL-based alternative to fine-tuning).
