# Decision Guide — Choosing a Robot Learning Approach

This page is a practical tool for picking the right approach given your constraints. It is opinionated by design. See [[overview]] for the conceptual framing of the three families.

---

## Step 1 — Clarify your constraints

Before choosing an algorithm, answer these four questions:

| Question | Answer options |
|----------|---------------|
| **Do you have a simulator?** | Yes (fast, GPU-accelerated) / Yes (slow/basic) / No |
| **Can you define a reward function?** | Yes (dense reward) / Partially (sparse/binary) / No |
| **Do you have demonstrations?** | Yes, many (50+) / Yes, few (10–30) / No |
| **What generalization do you need?** | Same objects/setup / New objects, same task / Multi-task, language-conditioned |

---

## Step 2 — Decision flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│              What's your primary constraint?                    │
└─────────────────────────────────────────────────────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
  Have simulator +        Have demos,           Need language /
  can define reward       no sim needed         multi-task generalization
         │                      │                      │
         ▼                      ▼                      ▼
    ┌────────┐           ┌────────────┐         ┌────────────────┐
    │   RL   │           │    IL      │         │      VLA       │
    └────────┘           └────────────┘         └────────────────┘
         │                      │                      │
    ┌────┴────┐           ┌─────┴─────┐         ┌──────┴──────┐
    │         │           │           │         │             │
    ▼         ▼           ▼           ▼         ▼             ▼
Continuous  Discrete  Few demos   Many demos  Compute     Low compute
  action     action   (10–30)     (50+)       available   constrained
    │         │           │           │         │             │
    ▼         ▼           ▼           ▼         ▼             ▼
  SAC       DQN/PPO    ACT/IL→RL   Diffusion  OpenVLA/    SmolVLA /
                       + AWAC      Policy     π0           TinyVLA
```

---

## Step 3 — Full recommendation table

| Your situation | Recommended approach | Why | Key papers |
|---------------|---------------------|-----|-----------|
| Sim available, reward definable, continuous actions | **SAC** | Off-policy, sample-efficient, auto-tuned entropy | [[algo-sac]] |
| Sim available, reward definable, sparse reward | **SAC + HER** | Hindsight relabeling rescues sparse binary rewards | [[algo-her]] |
| Sim available, long-horizon, composable subtasks | **SQL → SAC** | Composable policy arithmetic | [[algo-sql-sac]] |
| Offline data from previous deployments | **CQL + SAC** | Conservative Q-learning prevents OOD overestimation | [[algo-cql]] |
| 10–30 precise demos, fixed robot setup | **ACT** | Action chunking handles temporal precision, works from few demos | [[algo-act]] |
| 20–100 demos, multimodal behaviors | **Diffusion Policy** | Captures multimodal distributions; +46.9% vs prior SOTA | [[algo-diffusion-policy]] |
| 50+ demos, long-horizon multi-stage | **Diffusion Policy + Mamba2Diff** | Temporal SSM for long sequences | [[imitation-learning]] |
| Have demos but want to exceed demonstrator | **AWAC / HITL-RL** | Bootstraps from demos then improves with RL | [[algo-awac]], [[algo-hitl-rl]] |
| Long-horizon from unstructured human video | **Relay Policy Learning** | Hierarchical IL+RL from unstructured demos | [[algo-relay-policy]] |
| Language-conditioned, many objects, compute available | **OpenVLA (fine-tune)** | 7B open-source, strong language grounding, QLoRA-fine-tunable | [[algo-openvla]] |
| Language-conditioned, limited compute | **SmolVLA** | 450M, community data, async inference, 40% faster than π0 | [[vision-language-action-models]] |
| Cross-embodiment, state-of-the-art frontier | **GR 1.5 / π0** | Multi-embodiment, thinking VLA, 10M+ demos | [[algo-gemini-robotics-15]] |

---

## Step 4 — Family profiles (advantages, limits, training, deployment)

A complete picture of each family: what it's good at, where it breaks, how you actually train and deploy it, and what the practical constraints are.

---

### 🎯 Reinforcement Learning (RL)

#### ✅ Key advantages
- **No demonstrations needed** — only a reward signal and a simulator. The robot discovers strategies from scratch, including ones humans wouldn't think of.
- **Can exceed human performance** — unlike IL, RL is not bounded by demonstrator quality. With enough simulation time, it finds optimal policies.
- **Precise, repeatable control** — RL policies learn exact joint trajectories that optimize the reward, producing very precise and consistent behavior on the trained task.
- **Reward shaping = task decomposition for free** — breaking the task into subtasks with shaped rewards (approach → grasp → place) dramatically accelerates convergence and produces interpretable intermediate behaviors.
- **Strong in simulation** — GPU-parallelized sim (MuJoCo, Isaac Sim) allows millions of training episodes per hour. [[algo-sac|SAC]] on a pick-and-place task converges in hours on a single GPU.

#### ❌ Hard limits
- **Without a simulator** — real-robot RL is extremely slow and risky. Exploration inevitably produces dangerous behaviors. The sim-to-real gap is a second tax on top.
- **Without a reward** — if you can't specify "success" as a scalar number, RL has nothing to optimize. Reward design is often the hardest part and can take as long as the training itself.
- **No generalization** — a [[algo-sac|SAC]] policy trained to pick a red cube from bin A will fail on a green cube, a different bin, or different lighting. RL policies are highly task-specific by nature.

#### 🔧 How to train ([[algo-sac|SAC]] / [[algo-ppo|PPO]])

**1. Choose a simulator**

| Simulator | Best for | Notes |
|-----------|---------|-------|
| MuJoCo / MJX | Contact-rich tasks, fast physics | Open-source; GPU via MJX |
| Isaac Sim (NVIDIA) | Photorealistic, massive parallelism | Good for sim-to-real visual transfer |
| Robosuite | Manipulation benchmarks, easy setup | Built on MuJoCo, standardized tasks |
| PyBullet | Lightweight, accessible | Less accurate contact dynamics |

**2. Define the MDP**
- **State space**: joint positions + velocities, gripper state, object pose (from sim ground truth or estimated from camera). For real transfer, prefer using only observations available on real hardware.
- **Action space**: joint velocity deltas ([[algo-sac|SAC]] default) or end-effector delta pose (more intuitive reward design). End-effector control simplifies reward shaping but adds IK overhead.
- **Episode length**: enough for the task (e.g. 200 steps at 20 Hz = 10 seconds for P&P). Too short = never succeeds; too long = learns to stall.

**3. Design the reward — the most important step**
- **Decompose into subtasks** (approach → grasp → place). Each subtask gets its own shaped reward term. This is the single most impactful choice for convergence speed.
- **Use normalized distance rewards**: `r = 1 - tanh(distance / scale)` — bounded in [0,1], avoids reward scale issues across subtasks.
- **Add success bonus**: binary +1 when task completes. This shapes the end condition clearly.
- **Penalize dangerous actions**: penalize excessive joint torques or end-effector forces to protect hardware.
- **Avoid reward hacking**: if the robot finds a way to maximize reward without completing the task (e.g., dragging the object), add a penalty for that behavior.
- **[[algo-her|HER]] for sparse signals**: if you can only provide binary success/failure, use [[algo-her|Hindsight Experience Replay]] — it turns every failed episode into useful data.

**4. Algorithm choice**
- **[[algo-sac|SAC]]** — default choice for continuous action manipulation. Auto-tunes entropy coefficient α; off-policy so data-efficient; twin critics prevent Q-value overestimation.
- **[[algo-ppo|PPO]]** — better for discrete actions or when sample efficiency matters less than training stability. Simpler to implement.
- **[[algo-cql|CQL]]** — when you have offline data (previous robot runs) you want to reuse before online training.

**5. Training loop**
- Start with 0 exploration steps, let [[algo-sac|SAC]]'s initial random policy fill the replay buffer (typically 1k–10k steps).
- Monitor: reward curves, success rate, entropy (α). A falling entropy with no reward increase = policy collapsed early.
- **Curriculum**: start with the object close to the robot (easy), progressively randomize its position more widely. This is much faster than training on the full distribution from step 0.
- Run **at least 1M steps** for P&P with [[algo-sac|SAC]]. Evaluate every 50k steps on a fixed held-out set of 100 initial configs.

#### 🛡️ Building robustness — RL

The core technique is **domain randomization**: training with varied conditions so the policy doesn't overfit to any specific configuration. Apply these during training, not just at test time.

**Visual robustness**
- **Lighting**: randomize ambient light intensity (±30%), add directional lights, vary shadow direction and intensity. In sim: randomize light position/color each episode.
- **Camera pose**: add small random perturbations to camera position (+/- 2cm) and orientation (+/- 3°) each episode.
- **Background / table texture**: randomize table surface texture (colors, patterns). This prevents the policy from using table color as a cue.
- **Object color/texture**: if using visual observations, randomize object appearance. A policy trained only on a red cube will fail on a blue cube.
- **Object distractors**: add irrelevant objects (different shapes, colors) in the scene that should not be grasped. Forces the policy to be selective based on task instruction or reward signal.

**Spatial robustness**
- **Object position**: sample object starting position uniformly from a region (not a fixed point). Gradually increase the region size as training progresses (curriculum).
- **Object orientation**: randomize in-plane rotation (0–360°) and out-of-plane tilt (±15°) for objects with rotational symmetry. For non-symmetric objects, vary orientation to cover all grasping faces.
- **Goal position**: randomize the placement target location within a valid region.
- **Robot starting pose**: vary initial joint configuration slightly to prevent the policy from memorizing a fixed approach trajectory.

**Dynamics robustness**
- **Object mass/friction**: randomize ±20% around nominal values. This is critical for sim-to-real — real objects have different mass/friction than sim defaults.
- **Joint damping**: vary robot joint damping to simulate motor wear and hardware variation.
- **Latency**: add random observation delays (1–3 timesteps) to simulate sensor/communication latency on real hardware.
- **Gripper compliance**: vary gripper force/stiffness to simulate real pneumatic/servo gripper variation.

**Sim-to-real calibration checklist**
- [ ] Camera intrinsics/extrinsics match real setup
- [ ] Object mass and friction calibrated from real measurements
- [ ] Joint PD gains match real robot controller
- [ ] Contact model validated on sample grasps

#### 🚀 How to deploy
- **Export policy**: save SAC actor network (small MLP or CNN) as ONNX or TorchScript for efficient inference
- **ROS/ROS2 wrapper**: policy node reads joint states + camera frames → outputs joint velocity commands at 20–50 Hz
- **Sim-to-real calibration**: run the same sim environment with real camera feed to check visual alignment before first real run
- **Expect a sim-to-real drop**: typically 10–20 pp. Budget time for real-world fine-tuning ([[algo-cql|CQL]] on real rollouts, or [[algo-hitl-rl|HITL-RL]] for targeted corrections)
- **Safety layer**: add joint limit checks and velocity clipping as a safety wrapper around the policy output

#### ⚠️ Key constraints
| Constraint | Typical value | Notes |
|-----------|--------------|-------|
| Training time (sim) | Hours on 1 GPU | With fast parallelized sim |
| Training time (real) | Days–weeks | Exploration is slow on hardware |
| Data needed | 0 demos | Only a reward function |
| Inference speed | Very fast (<1ms) | Policy is a small MLP or CNN |
| Generalization scope | Single task/setup | Retraining needed for new objects |
| Hardware risk | High (exploration) | Mitigated by sim pretraining |

---

### 🎭 Imitation Learning (IL)

#### ✅ Key advantages
- **No reward design** — the expert demonstrates the task; the policy learns the behavior directly. This eliminates the hardest part of RL (reward engineering).
- **Safe training** — no exploration. The policy only produces actions that look like the demonstrations. Zero risk of unexpected robot behavior during training.
- **Fast convergence from few demos** — [[algo-act|ACT]] achieves 80–90% success from ~10 minutes of demonstrations. [[algo-diffusion-policy|Diffusion Policy]] extracts rich multimodal information from 20–100 demos. No simulator needed.
- **Captures implicit human knowledge** — force modulation, grasp approach angles, timing — things that are easy to demonstrate but nearly impossible to specify as a reward function.
- **Works directly on real hardware** — no sim-to-real gap. Demonstrations are collected on the target robot in the target environment.

#### ❌ Hard limits
- **Can't exceed demonstrator performance** — the policy learns what was shown. Inconsistent or suboptimal demos average into a suboptimal policy.
- **Distribution shift** — as soon as the robot reaches a state not covered by demos (which it will), performance degrades. Errors compound over time steps.
- **Data collection cost** — quality demonstrations require time and setup. 20–50 teleoperated demos via SpaceMouse or kinesthetic teaching take 1–3 hours. Bimanual setups (ALOHA) require expensive hardware.
- **Brittle to visual changes** — different lighting, slightly different camera angle, new object appearance all cause performance drops not seen during training.

#### 🔧 How to train (ACT / Diffusion Policy)

**1. Demo collection protocol — quality over quantity**

The most important investment is *how* you collect demonstrations, not just how many.

- **Teleoperation** (SpaceMouse, ALOHA leader arm): captures both position and velocity intent; best for dexterous tasks. Preferred for P&P and assembly.
- **Kinesthetic teaching** (manually guide the robot): faster to set up, but physical contact with the robot can introduce noise and inconsistency.
- **Human video + retargeting** ([[imitation-learning#EasyMimic|EasyMimic]]): requires no robot during demo collection, but retargeting to robot kinematics introduces error.

**Collection consistency rules** (critical — inconsistent demos are worse than fewer consistent ones):
- Fix camera position and lighting before any recording — do not move either mid-dataset
- Standardize grasp approach: always approach from the same angle, same height
- Standardize speed: too fast → noisy trajectories; too slow → policy learns to be sluggish
- Collect demos across the **full intended operating range** of object positions, not just the center
- Record a minimum of 3–5 demos per condition you care about. 1–2 is not enough for statistical coverage.

**Data filtering**:
- Watch each demo before keeping it. Discard demos with hesitations, unexpected grasps, or failures mid-demo.
- Compute trajectory smoothness score (e.g., jerk) and flag outliers for manual review.

**2. Architecture choice**

| Situation | Architecture | Why |
|-----------|-------------|-----|
| Few demos (10–30), precise task | **[[algo-act\|ACT]]** | β-CVAE latent captures behavioral modes; chunking reduces error compounding |
| Many demos (50+), multimodal grasps | **[[algo-diffusion-policy\|Diffusion Policy]]** | DDPM covers full multimodal distribution; +46.9% vs prior SOTA |
| Long-horizon (5+ steps) | **[[imitation-learning\|Mamba2Diff]]** | SSM handles temporal dependencies across long sequences |
| Fast inference needed (<10ms) | **[[algo-act\|ACT]]** or **[[algo-vq-bet\|VQ-BeT]]** | VQ-BeT is 5× faster than Diffusion Policy |

**3. Training details — [[algo-act|ACT]]**
- **Chunk size (k)**: typically 50–100 timesteps (1–5 seconds at 20 Hz). Larger chunks reduce boundary artifacts but increase latency.
- **β parameter** (KL weight in β-CVAE): controls latent space compression. Higher β = smoother but less expressive. Start at β=10, tune based on rollout smoothness.
- **Training budget**: 50k–200k gradient steps. Use learning rate warmup (1k steps) and cosine decay.
- **Validation**: reserve 10–20% of demos as a held-out set. Monitor reconstruction loss on held-out trajectories, not just training loss.

**3b. Training details — [[algo-diffusion-policy|Diffusion Policy]]**
- **Noise schedule**: [[algo-diffusion-policy|DDPM]] with 100 denoising steps at training; can reduce to 10–25 at inference with DDIM.
- **Action horizon**: predict 16–32 future steps; execute 8. Receding horizon reduces commitment error.
- **Visual encoder**: typically a pre-trained ResNet or ViT. Freeze or fine-tune depending on how different your visual domain is from ImageNet.

**4. Preprocessing**
- Normalize all joint positions to [-1, 1] using dataset min/max
- Normalize images to [0, 1], apply mean/std normalization using dataset statistics
- Align timestamps between camera stream and joint recordings (off-by-one-frame errors are common)

#### 🛡️ Building robustness — IL

For IL, robustness comes primarily from **demo diversity** (you control what the policy sees at training time) and **data augmentation** (synthetic variation applied to existing demos).

**Demo diversity — collect across the full variation space**

This is the most important robustness lever for IL. The policy can only generalize to conditions it has seen during training.

| Variation axis | What to do | Notes |
|---------------|-----------|-------|
| **Object position** | Sample positions on a grid or randomly across the target workspace. At minimum: 3–4 distinct positions covering corners + center | If you only demo center positions, expect failure near edges |
| **Object orientation** | Collect demos at 0°, 45°, 90°, 135° for symmetric objects. For asymmetric objects, cover all valid grasp faces | Even small rotations (15°) can break [[algo-act\|ACT]] if not covered |
| **Lighting conditions** | Collect some demos under different ambient lighting (bright/dim, window vs artificial). Even 5–10 demos per condition helps | Minimal cost; large robustness gain |
| **Object distractors** | Add irrelevant objects to the scene in some demos (10–20% of dataset). The policy should learn to ignore them | A policy never trained with distractors will try to grasp them |
| **Grasp alternatives** | If multiple valid grasps exist (left/right approach), deliberately include both in the dataset | [[algo-diffusion-policy\|Diffusion Policy]] handles this well; [[algo-act\|ACT]] needs both in the β-CVAE latent |
| **Table height / workspace variation** | If deployment height varies slightly, include this variation in demos | Often overlooked; 1cm height difference can break a policy |

**Data augmentation — synthetic variation at training time**

Applied to recorded demos during training, not during collection. These are cheap and effective:

- **Color jitter** (brightness ±20%, contrast ±20%, saturation ±20%, hue ±5%): makes the visual encoder invariant to minor lighting changes. Apply to every image before it enters the network.
- **Random crop** (crop to 90% of image, then resize back): builds spatial invariance within the image frame.
- **Gaussian noise on joint positions** (σ ≈ 0.002 rad): improves robustness to encoder position noise.
- **Random cutout / masking** (mask 10–20% of image randomly): forces the policy to not rely on a single visual cue.
- **Do NOT augment the action trajectory** — only augment the input observations. Augmenting actions introduces inconsistency between observations and actions.

**What augmentation does NOT fix**:
- New object types / appearances not in the dataset
- Completely different lighting (not just ±20% variation)
- Object positions outside the demonstrated range
- Camera pose changes beyond a few degrees

#### 🚀 How to deploy
- **Load policy + preprocessors**: load trained weights, normalization stats, and image preprocessing pipeline together
- **Action chunking ([[algo-act|ACT]])**: predict chunk of k actions → execute at hardware frequency → re-query policy at chunk boundary (or every n steps with temporal ensembling)
- **Temporal ensembling**: query policy every step, take weighted average of overlapping predictions — smoother execution, slightly higher compute
- **Async inference ([[algo-diffusion-policy|Diffusion Policy]])**: run inference at 5–10 Hz on GPU server; robot client executes from action queue at 20–50 Hz
- **Monitor for distribution shift**: if execution deviates significantly from training trajectories (measure joint position error vs expected), flag for human correction or re-demonstration (DAgger, [[algo-hitl-rl|HITL-RL]])
- **Recovery**: when failure detected, return to a known safe state (home position) before retrying

#### ⚠️ Key constraints
| Constraint | Typical value | Notes |
|-----------|--------------|-------|
| Demo collection time | 1–4 hours | For 20–50 demos on a single task |
| Training time | 1–4 hours (consumer GPU) | Fast compared to RL |
| Data needed | 10–100 demos | Quality > quantity |
| Inference speed | Fast (ACT: <10ms) | Diffusion Policy: ~50–200ms per step |
| Generalization scope | Fixed setup | New objects/positions require new demos |
| Hardware risk | Very low | No exploration, pure imitation |

---

### 🌐 Vision-Language-Action Models (VLAs)

#### ✅ Key advantages
- **Semantic generalization** — understands natural language instructions ("pick the mug on the left") without per-instruction training. Inherited from internet-scale pretraining.
- **Novel object generalization** — recognizes and grasps objects not seen during fine-tuning because the visual backbone already understands object categories.
- **Low fine-tuning data requirement** — 50–200 demonstrations can produce a model that generalizes broadly, because the backbone already handles perception and reasoning.
- **Cross-embodiment transfer** — models like [[algo-octo|Octo]], X-VLA, and [[algo-gemini-robotics-15|GR 1.5]] can transfer across different robot hardware with minimal additional data.
- **Language as a free interface** — no task-specific programming. "Sort the objects by color" works if the model understands color, without any task-specific demo for that exact sorting criterion.

#### ❌ Hard limits
- **Geometric precision gap** — VLAs fail on tight tolerance tasks (peg-in-hole, connector insertion, assembly). They predict joint targets from images without modeling contact forces. This is a fundamental gap with current architectures.
- **Inference speed** — 7B models run at 1–6 Hz. Even SmolVLA (450M) may not reach 20 Hz without async inference. High-frequency dexterous control is out of reach without special engineering.
- **Fine-tuning data quality** — bad demonstrations are confidently reproduced. Unlike RL (which can self-correct via reward), IL-based VLA fine-tuning has no self-correction mechanism.
- **Compute cost** — even the smallest production VLAs (450M+) require a GPU for both training and inference. Not deployable on embedded hardware.
- **Black-box behavior** — it's difficult to understand why a VLA fails. The failure mode could be perception, reasoning, or action generation — and they're entangled in a single forward pass.

#### 🔧 How to train / fine-tune (SmolVLA / OpenVLA)

**1. Choose the base model**

| Model | Parameters | Hardware needed | Best for |
|-------|-----------|----------------|---------|
| [[vision-language-action-models\|SmolVLA]] | ~450M | Single consumer GPU (8GB+) | Accessible; community data; SO-100 compatible |
| [[algo-openvla\|OpenVLA]] | 7B | 48GB GPU (or 4×A100 without [[algo-qlora\|QLoRA]]) | Strong language grounding; well-documented |
| [[vision-language-action-models\|π0]] | 3.3B | 24GB GPU (quantized) | Flow matching; large pretrain corpus |
| [[algo-gemini-robotics-15\|GR 1.5]] | — | API only | Multi-embodiment; thinking VLA; closed |

**2. Collect and structure fine-tuning demos**
- **Format**: use LeRobotDataset (tabular joint data + MP4 videos + JSON metadata). This ensures compatibility with all HuggingFace VLA training pipelines.
- **Quantity**: 50–200 demos covers most fixed-task setups. More demos → better generalization, but diminishing returns after ~100 for a single task.
- **Language annotation**: every demo must have a clear, consistent language instruction. Use a fixed vocabulary ("pick the [color] [object] and place it in the [location]"). Inconsistent wording degrades instruction grounding.
- **Camera setup**: follow the model's expected camera configuration (SmolVLA expects top/wrist/side views in a standardized order). Document your camera-to-model mapping.
- **Re-annotation tool**: if instructions are missing or noisy, run a small off-the-shelf VLM (e.g., LLaVA, GPT-4V) on sampled frames to generate standardized task descriptions — same technique used to build SmolVLA's training set.

**3. Fine-tuning with LoRA / [[algo-qlora|QLoRA]]**
- **LoRA rank**: r=8 or r=16 is typically enough for single-task adaptation. Higher rank = more capacity but more parameters to update and higher overfitting risk.
- **Target modules**: attention layers (Q, K, V, O projections). Optionally include the action head fully (no LoRA on the action expert).
- **[[algo-qlora|QLoRA]] (for large models)**: 4-bit NF4 quantization of the frozen backbone + full-precision LoRA adapters. Enables 7B fine-tuning on a single 48GB GPU.
- **Learning rate**: 1e-4 to 5e-5 for LoRA adapters. Use cosine decay with 100-step warmup.
- **Epochs**: 10–50 epochs on a small dataset. Monitor held-out success rate, not just training loss — VLAs can memorize demos without generalizing.
- **Frozen vs full fine-tune**: LoRA (frozen backbone + adapters) is strongly preferred. Full fine-tuning on a small dataset destroys the pretrained language/vision knowledge.

**4. Instruction design**
- Keep instructions **action-grounded** ("pick the red cube and place it in the blue bin") rather than abstract ("sort the objects").
- Include **instruction paraphrases** in the dataset: "pick up the red cube", "grab the red block", "take the red object" all map to the same demo. This significantly improves instruction-level generalization.
- For object differentiation: include color, size, and/or position descriptors ("the small red cube on the left"). The model's pretrained vocabulary covers most common objects and colors.

#### 🛡️ Building robustness — VLA

VLA robustness comes from two sources: **fine-tuning data diversity** (same principles as IL) and **leveraging the pretrained backbone's existing robustness**.

**What the pretrained backbone already handles**
VLAs start with a backbone trained on billions of images from diverse conditions. This means some robustness to lighting changes, object appearance, and background variation is *already built in* — unlike IL policies trained from scratch. The fine-tuning stage mainly needs to teach *how to act on this specific robot* rather than *how to perceive the world*.

**What fine-tuning data still needs to cover**

| Variation axis | How much diversity needed | vs IL |
|---------------|--------------------------|-------|
| Object color/texture | Low — backbone generalizes | IL needs explicit demos |
| Object position | Moderate — cover operating workspace | Same as IL |
| Object orientation | Moderate — especially non-symmetric objects | Same as IL |
| Lighting | Low — pretrained backbone robust | IL needs lighting demos |
| Distractors | Low-moderate — backbone recognizes them | IL more fragile to distractors |
| Instruction paraphrases | **High** — unique to VLAs | Not applicable to IL |
| New object *categories* | **High** — include diverse objects | Not applicable to IL |

**Data augmentation for VLA fine-tuning**
- **Visual augmentation** (same as IL): color jitter, random crop, cutout. Apply to fine-tuning images.
- **Instruction augmentation**: generate 3–5 paraphrases per unique instruction using a language model. Feed all paraphrases during training. This is the most impactful augmentation unique to VLAs.
- **Negative instructions**: include a small fraction of demos where the instruction does NOT match the demonstrated task, labeled as incorrect. Trains the model to reject ill-formed instructions rather than hallucinating a response.

**Failure mode unique to VLAs**: **instruction hallucination** — the model confidently executes a plausible-sounding task that was not requested. This is rare for simple instructions but appears with ambiguous or complex prompts. Mitigation: keep instructions simple and unambiguous, and add a success-detection step after each action.

#### 🚀 How to deploy
- **Policy server + robot client** (LeRobot async inference stack): VLA inference on GPU server → action chunks buffered → robot executes at hardware frequency (20–50 Hz)
- **Quantized inference**: 4-bit quantization allows 7B models on a single 24GB GPU with <5% performance drop
- **Language interface at runtime**: send instruction string; VLA handles grounding without retraining
- **Action chunk execution**: VLAs typically predict 50-step chunks (like ACT). Execute the chunk, then re-query with updated observation.
- **Failure recovery**: add a lightweight success classifier (binary: did the subtask complete?) between action chunks. On failure, either re-issue the instruction or fall back to a recovery primitive (see [[algo-gemini-robotics-15|GR-ER 1.5]] for a production example of this pattern).
- **Inference latency budget**: [[vision-language-action-models|SmolVLA]] at 450M generates a 50-step chunk in ~200ms on a GPU — acceptable for most tasks. For faster loops, reduce chunk size or use async inference.

#### ⚠️ Key constraints
| Constraint | Typical value | Notes |
|-----------|--------------|-------|
| Fine-tuning time | 4–24 hours (1 GPU) | Depends on model size and LoRA rank |
| Data needed | 50–200 fine-tune demos | Pretrained backbone handles the rest |
| Inference speed | 1–6 Hz (7B), 5–15 Hz (450M) | Async inference partially compensates |
| Generalization scope | Broad (language-conditioned) | Best generalization of the 3 families |
| Hardware at inference | GPU required | Min ~8GB VRAM for quantized 7B |
| Hardware risk | Low | Same as IL — pure inference |

---

## Step 5 — How to evaluate — go to [[evaluation-protocol]]

Once trained, all three families should be evaluated under the same standardized conditions to enable fair comparison. Rather than embedding the full protocol here, it lives in a dedicated page:

→ **[[evaluation-protocol]]** — evaluation axes (in-distribution, near-OOD, far-OOD, robustness, perturbation), metrics, and the SO-100-specific test protocol

---

## Step 6 — Hybrid strategies worth knowing

| When | Strategy | Mechanism |
|------|----------|-----------|
| Few demos + want improvement | **IL → RL (AWAC)** | BC initializes policy, RL fine-tunes online | 
| Human-in-the-loop training | **HITL-RL** | Human corrections stored in replay buffer, 1-2h to near-perfect |
| Pre-trained VLA + RL alignment | **VLA-RL** | Online RL fine-tunes frozen VLA; process reward model handles sparse rewards |
| Sim pretraining + real fine-tuning | **CQL pretraining + SAC online** | Offline conservative Q-learning, then switch to online RL |

---

## Step 7 — REX: Return on Experience (SO-100 Pick-and-Place)

This section compiles empirical observations from experiments on the **SO-100** robot arm. Each algorithm entry follows the same structure: setup, what worked, what didn't, surprises, and a revised recommendation informed by reality. This is the section that makes the recommendations above concrete rather than theoretical.

---

### SAC — Soft Actor-Critic

**Task:** Pick-and-place on SO-100 in simulation (MuJoCo)  
**Result:** ✅ **92% overall success (50 episodes)**

#### Setup
- Simulator: MuJoCo (CPU)
- State space: joint positions + gripper state + object/goal positions
- Action space: joint velocity deltas (continuous)
- Reward: 3-subtask decomposition (Reach / Grasp scripted / Place) from Kim et al. 2023 — weighted axis distance penalty + energy penalty
- Object position randomized ±7cm (fixed orientation), fixed box

#### What worked
- **Task decomposition was the decisive lever.** Splitting into 3 subtasks (Reach → scripted Grasp → Place) with per-subtask shaped rewards accelerated convergence dramatically vs. a single sparse signal.
- **Scripting the Grasp** (100-step deterministic gripper close) eliminated one failure mode. The lesson: don't learn what can be made deterministic.
- **Recovery system:** if the cube falls (z ≤ 5cm) during Place, the state machine relaunches Reach. 8 drop recoveries in 50 episodes saved 8 episodes that would have failed.
- **Quantitative results (50 episodes):** Reach 96% (48/50) · Grasp 92% (46/50) · Place 92% (46/50) · Overall 92% (46/50)
- **PPO vs SAC comparison (same env, 50% success threshold):** SAC reaches 50% in 1.58M steps (5h); PPO in 3.17M steps (20 min). SAC is 2× more sample-efficient but slower wall-clock.

#### What didn't work
- **Generalization is hard.** The policy was trained with fixed orientation and ±7cm position randomization. New orientations or larger position ranges require retraining.
- **Reward design took significant iteration.** The energy penalty and per-axis weighting required empirical tuning before convergence was clean.

#### Critical assessment
SAC is the strongest baseline on this task in simulation. The 3-subtask decomposition pattern is directly reusable. The engineering cost is upfront (reward design: ~1–2 days of iteration) but the payoff is high (92% reliable sim policy). Real-hardware transfer remains untested — literature suggests 10–20 pp drop.

> **Revised recommendation:** SAC + 3-subtask decomposition + scripted intermediate steps is the go-to for any simulation P&P task. Add a recovery state machine from day one — it costs little and saves many episodes.

---

### ACT — Action Chunking with Transformers

**Task:** Pick-and-place on SO-100 (real hardware)  
**Result:** ✅ **83% in-distribution / 94% at 45° orientation (Dataset_v4)**

#### Setup
- Dataset: Dataset_v4 (80 Phase 1 episodes + 15 recovery + 16 random orientation = ~111 episodes)
- Training: 100k steps, single GPU
- Cameras: wrist + side
- Teleoperation: ALOHA-style leader/follower with PID gain adjusted (P=4) for smooth trajectories

#### Dataset iteration journey
- **v1 (50 eps, mono camera):** Failed — environment not reproducible between training and eval. Policy replays training trajectory but can't adapt to new cube position. Key lesson: _need a reproducible environment_.
- **v2 (80 eps, Phase 1):** Approaches cube at ~10cm. Understands intention (always goes toward cube) but can't get close enough. Hypothesis: low contrast — blue gripper confuses with black box.
- **v3 (80 eps + colored markers):** Post-its on robot and box improve contrast. Approaches 5–7cm, slightly to the side. OOD: goes to closest known position, can't reach new ones.
- **v4 (v3 + 15 recovery + 16 orientation eps):** Final dataset. Solved orientation and recovery gaps.

#### What worked
- **Recovery episodes were the game changer.** Adding 15 episodes starting from failure situations (cube pushed, arm up, cube dropped) fixed the complete inability to recover seen in v2/v3.
- **Orientation diversity worked.** 16 random orientation episodes made the 45° performance jump to 92–100%.
- **In-distribution performance is strong (83%).** ACT reliably reproduces trained positions.
- **Surprising OOD generalization at 45°:** 4/4 (100%) on unseen positions at 45° — better than at 0°. The varied orientation episodes created richer generalizable features.
- **Distractor robustness (75%):** Handled well — ACT is not confused by a second object.

#### What didn't work
- **OOD at 0° is moderate (50%).** When the cube is at a completely unseen position in the standard orientation, ACT goes to the nearest known position and tries to close the gripper there.
- **Jerky movements** persist at some steps — the PID tuning helped but didn't fully resolve smoothness.

#### Critical assessment
ACT outperformed SmolVLA on all tested conditions with this dataset volume (~111 episodes). The key insight is that **data composition matters more than algorithm choice** — each dataset iteration (v1→v4) drove more improvement than any hyperparameter change.

> **Revised recommendation:** ACT with 80–100 well-structured episodes (varied positions, orientations, AND recovery situations) achieves 83–94% on real hardware without any reward design. The mandatory ingredients are: PID-calibrated teleop hardware, reproducible eval environment, and deliberate coverage of failure modes in the dataset.

---

### SmolVLA — Small Vision-Language-Action Model

**Task:** Pick-and-place on SO-100 (real hardware)  
**Result:** ✅ **58% in-distribution / 0% with distractor (Dataset_v4)**

#### Setup
- Base model: SmolVLA (~450M parameters, HuggingFace, pre-trained on SO-100/SO-101 community datasets)
- Fine-tuning: Dataset_v4, 20k steps, 4 GPUs (AWS g5.12xlarge)
- Same dataset as ACT (Dataset_v4)
- Language instruction: "pick the red cube and place it in the bin"

#### Dataset iteration journey (same as ACT)
- **v2:** Better spatial approach than ACT at same stage (~4-5cm above cube). Understands grasping intention (tries to close gripper when cube centered in wrist camera). But doesn't descend enough. Same contrast hypothesis as ACT.
- **v3 (25k steps, 4 GPUs):** 4 successes out of 16 tested (25%). Recovery completely absent. 45° orientation blocks the policy (keeps wrist straight instead of rotating).
- **v4 (20k steps, 4 GPUs):** Final dataset. Performance improved but key gaps remain.

#### What worked
- **Consistent performance across orientations.** SmolVLA achieves 58% in-distribution at both 0° and 45°, suggesting the pre-trained backbone provides some orientation invariance.
- **Understands the task globally.** Many near-successes (cube above box, dropped at the edge): the policy knows what to do and where to go. The failure is precision, not comprehension.
- **OOD orientation is moderate (50%)** — better than might be expected for a fine-tuned model with limited orientation training data.

#### What didn't work
- **0% with distractor (4 episodes).** Despite 3 near-successes, SmolVLA fails entirely when a second object is in the scene. Counter-intuitive given VLA's reputation for generalization — but this variation was absent from the fine-tuning data (Phase 1 only).
- **Recovery remains weak.** When the first grasp fails, SmolVLA tries to reposition above the cube but consistently misses. More recovery episodes in the dataset would likely fix this.
- **Lower absolute performance than ACT.** 58% vs 83% in-distribution. On this task and this data volume, ACT's specialization beats SmolVLA's generalism.

#### Surprises
- **The distractor result is the most striking finding.** The expectation was that SmolVLA would generalize better thanks to pre-training. Instead, ACT (trained from scratch on the same data) was far more robust to distractors. This suggests fine-tuning on Phase 1 only slightly "overwrites" the pre-training robustness.
- **Near-success rate is a better signal than success rate** for SmolVLA. Many 0%-success conditions actually have 75% near-success, indicating the policy is behaviorally correct but imprecise.

#### Critical assessment
SmolVLA's results are lower than expected on this specific task and data volume. The hypothesis: for a **fixed-setup single task with ~80 episodes**, ACT's direct specialization wins. SmolVLA's advantages (language conditioning, generalization) would likely show at larger data scale or on multi-task settings. The counter-intuitive distractor failure points to a fundamental tradeoff: fine-tuning on narrow data can erode the generalist robustness the pre-trained backbone provides.

> **Revised recommendation:** For a fixed single-task setup with <150 demos, prefer ACT. SmolVLA's generalist advantages need larger or more diverse fine-tuning data to manifest. If adding Phase 2 robustness demos (lighting, distractor), SmolVLA would likely recover its expected generalization advantage. The crossover point is somewhere around 150–200 diverse demos.

---

### Cross-algorithm comparison — SO-100 results

| Algorithm | Sim result | Real ID | Real OOD | Distractor | Data | Key finding |
|-----------|-----------|---------|----------|-----------|------|------------|
| **SAC** | **92%** (50 eps) | N/A (sim only) | N/A | N/A | 0 demos (reward) | 3-subtask decomp + recovery = 92% |
| **ACT** | — | **83%** (0°) / **92%** (45°) | **100%** OOD at 45° | **75%** (3/4) | ~111 demos | Data iteration > algorithm choice |
| **SmolVLA** | — | **58%** (both orient.) | 50% OOD at 45° | **0%** (3 near-success) | ~111 demos | Fine-tuning on narrow data erodes pre-training robustness |

**Key meta-finding:** The dataset iteration v1→v2→v3→v4 drove more performance improvement than any algorithmic change. ACT outperforms SmolVLA at this data volume; the advantage likely reverses with larger/more diverse fine-tuning data.

---

## Related pages
- [[overview]] — conceptual framing of the three families
- [[pick-and-place]] — task-specific synthesis for P&P
- [[reinforcement-learning]] — RL algorithm details
- [[imitation-learning]] — IL algorithm details
- [[vision-language-action-models]] — VLA model details
- [[hybrid-il-rl]] — hybrid IL+RL methods
