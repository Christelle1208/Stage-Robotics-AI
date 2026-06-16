# Algorithm: ACT — Action Chunking with Transformers

**Paper:** "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" — Zhao, Kumar, Levine, Finn (2023)
**Category:** Imitation learning — Transformer-based policy with action chunking
**Related:** [[imitation-learning]], [[grasping-and-manipulation]], [[algo-diffusion-policy]]

---

## The Problem It Solves

**Behavioral cloning** on fine-grained manipulation tasks (inserting a battery, threading a needle, precise assembly) has two fundamental failure modes:

1. **Compounding errors:** The policy predicts one action at a time. Small errors cause the robot to drift to states not seen during training. The policy then makes larger errors. Errors compound exponentially.
2. **Multimodal demonstrations:** Human demonstrations may pause, jitter, or approach the same object from different angles. BC tries to predict the *average* of these, resulting in a policy that hesitates or takes unnatural intermediate paths.

**ACT goal:** Achieve high success on delicate bimanual tasks from a small number of (~10-minute) demonstrations, using a transformer policy that predicts *chunks* of actions at once.

---

## How It Works

### 1. Action Chunking

Instead of predicting one action `a_t` at each timestep, ACT predicts a **chunk** of K future actions `(a_t, a_{t+1}, ..., a_{t+K-1})` jointly:

```
(a_t, ..., a_{t+K-1}) = π_θ(o_{t-H}, ..., o_t)
```

Where H is the observation history length and K is the chunk size (typically K=100 for a policy running at 50Hz = 2 seconds of actions).

**Why this helps with compounding errors:**
- Each prediction covers a longer time horizon, so small per-step errors have less impact
- The policy commits to a coherent trajectory segment, not a reactive single action
- The chunk is replanned every K steps (or every step with temporal ensemble — see below)

**Why this helps with multimodality:**
- A full chunk captures one coherent mode of behavior (e.g., "approach from the left")
- The policy doesn't need to average across modes; it commits to one at each planning step
- The CVAE (see below) explicitly models multiple modes in the latent space

### 2. CVAE for Multimodal Latent Space

ACT uses a **CVAE (Conditional Variational Autoencoder)** architecture:

**Encoder (used only during training):**
```
z = Encoder(o_t, a_t, ..., a_{t+K-1})
  = μ + σ·ε  where ε ~ N(0, I)
```
The encoder maps the current observation + the demonstrated action chunk to a latent variable z. z captures *which mode* of behavior is being executed.

**Decoder (used at test time = the policy):**
```
(a_t, ..., a_{t+K-1}) = Decoder(o_t, z)
```
The decoder maps observation + latent style z to an action chunk.

During training: the CVAE trains end-to-end, with a KL regularization term ensuring z ~ N(0, I).
During inference: sample z ~ N(0, I) → decode to get the action chunk. Or use z = 0 (mean of prior) for deterministic execution.

### 3. Transformer Architecture

Both encoder and decoder use **Transformer attention**:

**Encoder input tokens:**
- CLS token (produces the latent z)
- Tokenized current observation (from ResNet image backbone)
- Tokenized proprioception (joint positions)
- Tokenized action sequence

**Decoder input tokens:**
- Timestep queries (K learnable position embeddings, one per future action)
- Encoded observation features
- Latent z

Cross-attention in the decoder allows each future action to attend to the observation and the latent code. Multi-head self-attention lets actions within the chunk attend to each other (ensuring temporal consistency).

### 4. Temporal Ensemble (at Inference)

Instead of executing a chunk for K steps and then replanning, ACT can replan at *every* timestep and average overlapping predictions:

```
ā_t = Σ_{i=0}^{min(t,K-1)} w_i · a_t^{(t-i)}
```

Where `a_t^{(t-i)}` is the prediction of action at time t, made at time t-i (i steps ago), and w_i are exponential decay weights.

This **temporal ensemble** reduces variance: multiple independent predictions of the same action are averaged, smoothing jitter from the stochastic z.

---

## Why It Works

### Action Chunking Reduces Compounding Error

If each single-step prediction has error probability p, a sequence of T steps has compounding error ~T·p. But if each chunk of K steps has error probability p_chunk, then T/K replanning steps give error ~(T/K)·p_chunk. As long as p_chunk < K·p (the chunk prediction is at most K× worse than single-step), chunking wins.

In practice, a Transformer predicting K steps jointly is *much better* than K independent single-step predictions — it captures temporal dependencies within the chunk.

### CVAE Captures Human Variability

Human demonstrations are inherently multimodal (left approach vs. right approach; grasp from above vs. side). A standard regression loss would fit the average — which is often not a valid trajectory. The CVAE's latent z captures which mode each demonstration uses, so the policy learns all modes rather than averaging them.

### Transformer for Long-Range Dependencies

Fine-grained manipulation often requires "remember what I grasped 50 steps ago to place it correctly now." Self-attention across the K-step chunk window captures these long-range dependencies within a segment.

---

## Hardware: ALOHA

ACT was paired with the **ALOHA (A Low-cost Open-source Hardware System for Bimanual Teleoperation)** platform:
- Two ViperX robot arms (6-DoF each)
- Two Logitech cameras (top-view + wrist view)
- Total cost: ~$20,000 (vs. hundreds of thousands for industrial bimanual systems)
- Teleoperation via two matching ALOHA leader arms

The low cost is crucial: it makes 10-minute demo collection accessible to individual labs.

---

## Evaluation

### Tasks (all bimanual, from ALOHA)

| Task | Description | Difficulty |
|------|-------------|------------|
| Inserting battery | Slot AA battery into remote | High (2mm tolerance) |
| Ziplocking bag | Close a ziploc bag | High (flexible object) |
| Taping phone charger | Stick cable to surface | Medium |
| Slotting dishes | Slide plate into dish rack | Medium |
| Opening cup | Unscrew cup lid | High (rotation) |
| Stacking cups | 3-cup bimanual stack | Medium |

### Data Collection
- ~10-50 demonstrations per task (~10 minutes of teleoperation)
- 50Hz control rate; action space = 14 joint positions (7 per arm)

### Baselines
- **BC + ResNet (flat behavioral cloning):** Attends to image observations; no chunking, no CVAE
- **BC + Transformer (flat):** Transformer policy with single-step prediction
- **GATO-style (autoregressive):** Tokenizes actions

### Results

| Task | BC + ResNet | BC + Transformer | ACT (no ensemble) | ACT |
|------|-------------|-----------------|-------------------|-----|
| Battery insertion | 0% | 4% | 76% | **91%** |
| Bag ziplocking | 0% | 0% | 58% | **82%** |
| Cup stacking | 4% | 0% | 72% | **86%** |
| Average | ~1% | ~2% | ~65% | **84%** |

**Key numbers:** ACT achieves 80-90% success across tasks; baselines fail almost entirely (0-4%).

### Ablations (Success Rates)
- **No chunking (K=1):** ~35% (confirms chunking is the key)
- **No CVAE (no latent z):** ~55% (confirms multimodal modeling helps)
- **No temporal ensemble:** ~65% (ensemble adds ~20% on average)
- **K=10 vs K=100:** K=100 best; shorter chunks don't capture enough trajectory context

---

## Pros

- **State-of-the-art on fine-grained bimanual tasks** — 80-90% on tasks that stump all baselines
- **Data efficient** — 10-50 demos per task (10 minutes of teleoperation)
- **Handles multimodal demonstrations** — CVAE prevents mode-averaging failure
- **Reduces compounding errors** — chunking dramatically improves long-sequence performance
- **Open-source** — ALOHA hardware design + ACT code both publicly available

## Cons

- **Chunk size K is a hyperparameter** — requires tuning; different tasks need different K
- **Not reactive** — pre-committed chunks can't adapt mid-chunk to unexpected perturbations (mitigated by temporal ensemble)
- **Assumes similar observation conditions** — trained on specific cameras/backgrounds; may need domain adaptation for deployment
- **No online improvement** — purely IL; performance capped by demonstration quality and coverage
- **High-dimensional action space** — 14-DoF bimanual is the target; for simpler arms, less benefit from chunking
- **CVAE can over-commit** — sampling z stochastically can produce inconsistent behavior across planning steps (mitigated by temporal ensemble + deterministic z=0)

---

## Results in This Project — ACT & SmolVLA on SO-100 (Dataset_v4)

ACT and [[algo-smolvla|SmolVLA]] were fine-tuned on the same dataset (Dataset_v4, ~111 episodes: 80 Phase 1 + 15 recovery + 16 random-orientation episodes) and evaluated under the same protocol. Full per-condition tables and analysis live in [[results]] and the [[decision-guide]] REX section.

| Algorithm | In-distribution | OOD position | Distractor | Training | Key finding |
|-----------|-----------------|--------------|-----------|----------|------------|
| **ACT** | **83%** @ 0° / **92%** @ 45° | **100%** OOD @ 45° / 50% @ 0° | **75%** (3/4) | 100k steps, 1 GPU | Dataset iteration (v1→v4) drove most of the gain; recovery + orientation episodes were decisive |
| [[algo-smolvla\|SmolVLA]] | **58%** (both orientations) | 50% @ 45° / 25% @ 0° | **0%** (3 near-successes) | 20k steps, 4 GPUs, full fine-tune | Consistent across orientations, but fails entirely with a distractor |

**Takeaways for ACT specifically:**
- **Best overall algorithm on this setup** — outperforms SmolVLA on every tested condition at this data volume (~111 demos).
- **Recovery episodes were the single biggest improvement.** Adding 15 episodes that start from failure states (cube pushed, arm raised, cube dropped) fixed ACT's complete inability to recover, which was the dominant failure mode in Dataset_v2/v3.
- **45° orientation generalizes better than 0°** (100% OOD vs 50% OOD) — the 16 random-orientation training episodes produced richer, more generalizable visual features than expected.
- **Data composition > algorithm choice.** Each dataset iteration (v1→v4) produced a larger jump than any hyperparameter change — see [[decision-guide]] for the full v1→v4 narrative.

---

## In This Wiki

ACT: [[imitation-learning]], [[grasping-and-manipulation]], [[pick-and-place]] (bimanual tasks), [[results]] (SO-100 results dashboard).
Compare with: [[algo-diffusion-policy]] (different approach to multimodal IL; explicit denoising rather than CVAE), [[algo-vq-bet]] (tokenization-based alternative), [[algo-relay-policy]] (hierarchical IL+RL for long-horizon tasks), [[algo-smolvla]] (VLA alternative evaluated on the same dataset).
