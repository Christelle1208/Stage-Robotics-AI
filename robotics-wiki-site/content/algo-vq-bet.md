# Algorithm: VQ-BeT — Vector-Quantized Behavior Transformer

**Paper:** "Behavior Generation with Latent Actions" — Lee, Wang, Etukuru, Kim, Shafiullah, Pinto (NYU, 2024)
**Category:** Imitation learning — tokenization-based behavior generation
**Related:** [[imitation-learning]], [[algo-act]], [[algo-diffusion-policy]]

---

## The Problem It Solves

**Behavior Transformers (BeT)** — the prior work VQ-BeT extends — addressed multimodal IL by discretizing actions with k-means clustering and using a Transformer to predict clusters. Two limitations:

1. **k-means doesn't scale:** k-means in high-dimensional continuous action spaces (e.g., 14-DoF joint velocities × 100 timesteps) produces poor clusters — the "curse of dimensionality" means all clusters are equidistant.
2. **No gradient information:** k-means produces hard discrete assignments with no gradient, preventing end-to-end learning with the Transformer policy.

The alternative — **Diffusion Policy** — handles multimodality well but is slow at inference (100 denoising steps) and computationally expensive.

**VQ-BeT goal:** Match Diffusion Policy's multimodal expressiveness with BeT's speed, by replacing k-means with **learned vector quantization (VQ)** — a trainable discrete codebook that compresses continuous action sequences into tokens.

---

## How It Works

### Step 1: Hierarchical Vector Quantization (HVQ)

The key contribution is replacing k-means with a **hierarchical VQ** module that tokenizes action sequences:

**VQ-VAE Background:**
A standard VQ-VAE maps continuous vectors to the nearest entry in a learned codebook `{e₁, ..., eK}`:
```
z_q = argmin_{eₖ} ||z - eₖ||₂
```
Training uses a straight-through estimator: gradients flow backward through the quantization step as if it were an identity operation.

**Hierarchical VQ for actions:**
Rather than quantizing a single action, VQ-BeT quantizes a *sequence* of actions (a chunk `a_{t:t+W}`) hierarchically:

1. **Level 1 (coarse):** The action sequence is encoded to a latent `z`, then quantized to a coarse code `c₁` from a small codebook C₁ (e.g., 512 entries). c₁ captures the *mode* (approach from left vs. right, fast vs. slow).
2. **Level 2 (fine):** Given c₁, a residual is encoded and quantized to a fine code `c₂` from a smaller codebook C₂ (e.g., 32 entries). c₂ captures fine-grained details within the mode.

The full representation is `(c₁, c₂)` — a pair of discrete tokens.

**Reconstruction:** A decoder maps `(c₁, c₂)` → continuous action sequence `â_{t:t+W}`.

### Step 2: Training the Transformer Policy

Once the HVQ module is trained, the policy Transformer learns to predict `(c₁, c₂)` from observations:

```
(c̃₁, c̃₂) = Transformer(o_{t-H:t})
â_{t:t+W} = Decoder(c̃₁, c̃₂)
```

Training: cross-entropy on the discrete token predictions + MSE on residual offset from the codebook centroid.

**Why cross-entropy, not MSE?**
Predicting which *mode* (c₁) to select is a classification problem, not a regression. Cross-entropy is the right loss for discrete categorical predictions. The Transformer becomes a classifier over behavioral modes, plus a regressor for fine-grained offsets.

### Step 3: Two-Phase Training

**Phase 1: Train the HVQ module**
- Collect all action sequences from demonstrations
- Train the encoder-codebook-decoder to minimize reconstruction loss + VQ commitment loss
- No policy Transformer needed yet; this is pure unsupervised tokenization

**Phase 2: Train the Transformer policy**
- Freeze the HVQ codebook
- Train the Transformer to predict (c₁, c₂) from observations
- Also train a small offset predictor for continuous fine-tuning within each code

---

## Why It Works

### Gradient Flow Through VQ

k-means has no gradient; VQ-VAE uses the straight-through estimator to route gradients backward. In Phase 2, gradients from the policy loss can flow back into the observation encoder (improving visual features) — something impossible with k-means BeT.

### Hierarchical Captures Multiple Scales

The two-level hierarchy separates *what mode to use* (level 1, few codes) from *how to execute within that mode* (level 2, fine codes). This factorization is more compact and generalizes better than a flat large codebook.

### Speed vs. Diffusion Policy

Diffusion Policy inference: K=100 denoising steps on a U-Net or Transformer → multiple forward passes.
VQ-BeT inference: one Transformer forward pass → classify tokens → decode → done.

This makes VQ-BeT **~5× faster** than Diffusion Policy at inference time.

---

## Evaluation

### Environments (7 total)

**Simulation:**
1. **Push-T:** Push a T-shaped block to target — classical multimodal benchmark
2. **BlockPush:** Push two blocks sequentially — multi-stage
3. **Kitchen:** 4-task sequential kitchen manipulation (same environment as in Relay Policy Learning, Diffusion Policy)
4. **CARLA Autonomous Driving:** Turn decisions at intersections — multimodal (turn left, right, or straight)

**Real robot:**
5. **Bimanual ALOHA tasks:** Bimanual manipulation (same hardware as ACT)

**Partially observable:**
6. **Minigrid (partial obs):** Navigation with partial observability

**Multi-task:**
7. **LIBERO:** Multi-task manipulation benchmark

### Baselines
- BeT (k-means), BC (MSE), IBC (energy-based), Diffusion Policy, ACT, GATO

### Results

| Task | BeT | Diffusion | ACT | VQ-BeT |
|------|-----|-----------|-----|--------|
| Push-T (coverage) | 0.56 | 0.82 | 0.62 | **0.87** |
| BlockPush (task) | 65% | 72% | 64% | **79%** |
| Kitchen (# tasks) | 2.4 | 2.8 | 3.1 | **3.4** |
| CARLA (success) | 78% | — | — | **89%** |

**Key numbers:**
- VQ-BeT improves over Diffusion Policy on most benchmarks
- VQ-BeT matches or exceeds ACT
- **5× faster inference** than Diffusion Policy
- Specifically strong on multi-task and partial-observation settings

### Ablations
- **Flat VQ (no hierarchy):** ~10% worse than hierarchical
- **k-means instead of VQ:** Matches BeT performance (confirms VQ is the key improvement)
- **Larger codebook C₁:** Diminishing returns above 512 codes
- **Offset predictor:** ~5% improvement vs. nearest-centroid only

---

## Pros

- **Fast inference** — 5× faster than Diffusion Policy; suitable for real-time control
- **Multimodal** — hierarchical codebook captures multiple behavioral modes cleanly
- **End-to-end trainable** — straight-through VQ gradients enable full gradient flow
- **Strong across task types** — multi-task, partial observation, long-horizon, driving
- **Principled discretization** — learned VQ codes are semantically meaningful; k-means codes are not

## Cons

- **Two-phase training** — HVQ must be trained before the policy Transformer; adds training complexity
- **Codebook size hyperparameter** — C₁ and C₂ must be chosen; too small = mode collapse; too large = sparse code use
- **Codebook collapse** — a known VQ-VAE failure: some codebook entries never get used. Requires entropy regularization or EMA updates to avoid
- **Reconstruction quality ceiling** — the HVQ's ability to reconstruct actions is a bottleneck; lossy compression loses fine-grained precision
- **Less studied for real robot** — most experiments are simulation; fewer real-robot demos than ACT or Diffusion Policy

---

## Comparison: VQ-BeT vs. ACT vs. Diffusion Policy

| Property | Diffusion Policy | ACT | VQ-BeT |
|----------|-----------------|-----|--------|
| Multimodal | Yes (strong) | Yes (CVAE) | Yes (VQ codes) |
| Inference speed | Slow (100 steps) | Fast (1 pass) | Fast (1 pass, 5× faster than diff.) |
| Training complexity | Medium | Medium | High (2-phase) |
| Chunk prediction | Yes | Yes | Yes |
| Precision | High (continuous) | High (continuous) | Medium (codebook-limited) |
| Best for | Highly multimodal tasks | Fine-grained bimanual | Multi-task, partial obs, speed-critical |

---

## In This Wiki

VQ-BeT: [[imitation-learning]], [[pick-and-place]].
Compare with: [[algo-diffusion-policy]] (slower but often more precise on highly multimodal tasks), [[algo-act]] (CVAE-based; better on fine-grained bimanual).
