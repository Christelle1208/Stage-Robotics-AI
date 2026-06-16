# Algorithm: Diffusion Policy

**Paper:** "Visuomotor Policy Learning via Action Diffusion" — Chi, Feng, Du, Xu, Cousineau, Burchfiel, Song (Columbia / Toyota Research, 2023)
**Category:** Imitation learning — diffusion-based visuomotor policy
**Related:** [[imitation-learning]], [[algo-act]], [[algo-vq-bet]]

---

## The Problem It Solves

Robot behavior learned via imitation must handle several hard properties:
1. **Multimodality:** Human demonstrators solve the same task in different ways (approach left or right; open a door from different angles). Regression-based policies predict the mean — an average that's often physically impossible.
2. **High-dimensional action spaces:** Predicting full arm joint trajectories is a high-dimensional regression problem, prone to mode collapse.
3. **Training stability:** Behavior Transformer (BeT) discretizes actions to handle modes, but k-means clustering scales poorly and loses precision.

**Diffusion Policy goal:** Model the robot's action distribution as a **conditional denoising diffusion process** — generating actions by iteratively refining a noise sample, conditioned on visual observations. This inherits the expressive multimodal modeling of diffusion image generators for robot action prediction.

---

## How It Works

### Background: Denoising Diffusion Probabilistic Models (DDPMs)

DDPMs learn to generate data by training a neural network to reverse a fixed noise process:

**Forward process (adds noise):**
```
q(aᵏ | aᵏ⁻¹) = N(aᵏ; √(1-βₖ)·aᵏ⁻¹, βₖ·I)
```
Over K steps, the original action a⁰ becomes pure Gaussian noise aᴷ.

**Reverse process (removes noise — what we learn):**
```
p_θ(aᵏ⁻¹ | aᵏ, o) = N(aᵏ⁻¹; μ_θ(aᵏ, k, o), σₖ²·I)
```
A neural network learns to predict the noise (or the denoised action) at each step, conditioned on the observation `o`.

**Training objective:** Minimize the noise-prediction error:
```
L(θ) = E_{(o,a)~D, k~[1,K], ε~N(0,I)} [||ε - ε_θ(o, a⁰ + noise, k)||²]
```

**Inference:** Start from random Gaussian noise aᴷ ~ N(0, I), apply the learned denoiser K times to obtain a clean action a⁰.

### Diffusion Policy: Conditioning on Robot Observations

Diffusion Policy conditions the noise predictor on robot observations:

```
ε_θ(o_{t-H:t}, aᵏ_{t:t+W}, k)
```

Where:
- `o_{t-H:t}`: visual + proprioceptive observation history (H steps)
- `aᵏ_{t:t+W}`: the *action sequence* being denoised — a window of W future actions
- `k`: the current diffusion step

The output is a denoised action sequence `a⁰_{t:t+W}`. Like ACT, this predicts a **sequence** (chunk) of future actions, not just one.

### Two Noise-Predictor Architectures

**1. CNN-based (DDPM):**
- The visual observation is encoded with a CNN (ResNet-based)
- The noisy action + diffusion timestep + observation are fed to a 1D U-Net (temporal convolution)
- Simple and effective for fixed-camera setups

**2. Transformer-based (DDIM / time-series diffusion):**
- Observations are tokenized; noisy action sequence is tokenized
- A Transformer processes all tokens jointly via self-attention
- Better for multi-camera setups and longer action horizons
- Uses DDIM (Denoising Diffusion Implicit Models) for fast inference (10× fewer steps)

### Receding Horizon Control

Like ACT's chunking, Diffusion Policy uses a *receding horizon* strategy:
- Predict W future actions at each planning step
- Execute only the first `n < W` actions
- Replan (re-denoise from noise) every n steps

This combines the benefits of multi-step prediction (temporal coherence, error reduction) with responsiveness to new observations.

---

## Why It Works

### Multimodal Distributions

Diffusion models are powerful generative models for complex, multimodal distributions. Because the robot's action distribution can have multiple modes (grasp from left vs. right), a generative model that samples from the full distribution is fundamentally more appropriate than a regression model that predicts the mean.

At each denoising step, the model probabilistically collapses the action distribution onto one mode — the stochastic sampling process naturally selects and commits to one trajectory from the multimodal distribution.

### Score Matching Perspective

Diffusion Policy learns the *score function* ∇ₐ log p(a | o) — the gradient of the action log-probability with respect to the action. Inference traces the gradient toward high-probability action modes. This is a principled way to represent complex distributions without explicitly computing the normalizing constant.

### High-Dimensional Action Spaces

Unlike discrete methods (BeT uses k-means → limited to low-dimensional discretization), diffusion operates in continuous action space of any dimensionality. The 1D U-Net / Transformer naturally scales to 14-DoF joint trajectories.

---

## Evaluation

### Benchmarks (4 suites, 12 tasks total)

**1. Robomimic (simulation):** Lift, Can, Square, Transport — varying difficulty robot arm tasks; also includes multimodal demonstrations (proficient + subproficient mixed)

**2. Push-T (simulation):** Push a T-shaped object to a target configuration — requires strategic contact planning; highly multimodal

**3. BlockPush (simulation):** Push two blocks to specific targets in sequence — multi-stage

**4. Kitchen (simulation):** Complete 4 sequential kitchen tasks — long-horizon (same as in Relay Policy Learning evaluation)

**5. Real robot tasks:** Bimanual cup arrangement, sauce pouring, dish placement — tested on real hardware

### Baselines

| Method | Type | Handles Multimodal? |
|--------|------|---------------------|
| Behavioral Cloning (BC) | Regression | No |
| BeT (Behavior Transformer) | k-means discrete | Partially |
| IBC (Implicit BC) | Energy-based | Yes |
| BCRNN (RNN BC) | Regression | No |
| ACT | CVAE Transformer | Yes |
| **Diffusion Policy** | Diffusion | Yes |

### Results

**Average improvement: +46.9% over prior state-of-the-art** across all 12 tasks.

Selected results:

| Task | BC | IBC | BeT | ACT | Diffusion Policy |
|------|----|-----|-----|-----|-----------------|
| Robomimic-Lift | 92% | 68% | 78% | 98% | **98%** |
| Robomimic-Can | 72% | 12% | 58% | 94% | **95%** |
| Robomimic-Square | 26% | 6% | 30% | 72% | **76%** |
| Robomimic-Transport | 14% | 0% | 4% | 18% | **62%** |
| Push-T | 34% | 45% | 56% | 62% | **82%** |
| Kitchen (4 tasks) | 30% | — | 20% | — | **58%** |

**Key findings:**
- Diffusion Policy is especially dominant on **multimodal and high-precision tasks** (Transport: 62% vs 18% ACT, 14% BC)
- CNN-based and Transformer-based variants are competitive; CNN slightly better in constrained settings, Transformer scales better
- On simple tasks (Lift), all methods converge; Diffusion shines where multimodality matters

### Inference Speed
- DDPM (100 denoising steps): ~10Hz → too slow for reactive control
- DDIM (10 steps): ~25Hz → acceptable for most manipulation tasks
- Consistency Policy / DDIM with 1-5 steps: 50Hz+ → real-time capable (addressed in follow-up work)

---

## Pros

- **Best multimodal modeling** — diffusion naturally captures all modes of demonstration behavior
- **High-dimensional actions** — scales to full arm joint trajectories without discretization
- **Consistent temporal sequences** — generates coherent action chunks, not independent per-step predictions
- **Stable training** — noise-prediction MSE is a well-behaved regression loss (unlike GAN-based generative models)
- **Flexible architectures** — works with CNN, Transformer, U-Net backbones
- **Strong empirical results** — 46.9% average improvement; new SOTA on major manipulation benchmarks

## Cons

- **Slow inference** — K=100 denoising steps means ~10-20ms per action (DDPM); requires DDIM or consistency models for real-time
- **Compute-intensive** — running the full denoising chain at each control step is more expensive than forward-passing a single network
- **Hyperparameters** — noise schedule (β_k), diffusion steps K, and network architecture all require tuning
- **No guarantees on action validity** — unlike constrained optimization methods, denoised actions may slightly violate joint limits (mitigated by clamping)
- **Not online-improvable** — purely IL; needs re-training on new demonstrations to improve; no RL fine-tuning built in (though VLA-RL-style approaches could add this)

---

## Extensions and Follow-ups

- **TinyVLA** (this wiki): uses a diffusion policy decoder for fast inference in a VLA context
- **Consistency Policy** (2024): reduces denoising to 1-2 steps by training a consistency model; achieves real-time inference
- **3D Diffusion Policy**: adapts Diffusion Policy to 3D point cloud observations for better spatial reasoning

---

## In This Wiki

Diffusion Policy: [[imitation-learning]], [[pick-and-place]], [[vision-language-action-models]] (TinyVLA uses diffusion decoder).
Compare with: [[algo-act]] (CVAE alternative; better on fine-grained bimanual), [[algo-vq-bet]] (tokenization alternative; 5× faster inference), [[algo-sql-sac]] (RL alternative when online interaction is available).
