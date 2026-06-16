# Imitation Learning

Imitation learning (IL) trains robots by learning from expert demonstrations rather than reward signals. IL sidesteps the reward-design and exploration challenges of RL, but typically lacks mechanisms to improve beyond the demonstrator's performance. Modern methods range from simple behavioral cloning to sophisticated generative models. See also [[hybrid-il-rl]], [[reinforcement-learning]], [[vision-language-action-models]].

→ **Choosing between RL, IL, and VLAs?** See [[decision-guide]].

---

## When to use IL — Critical Synthesis

### ✅ IL genuinely excels at

**Learning precise, dexterous behaviors from few demonstrations.** ACT achieves 80–90% success on delicate bimanual tasks (battery insertion, bag ziplocking) from just ~10 minutes of demonstrations. No reward design, no simulator, no exploration risk. For a fixed robot setup, this is the fastest path from zero to a working policy.

**Tasks that are hard to specify as a reward.** Some behaviors are easy to demonstrate but extremely difficult to define mathematically. Pouring a drink to exactly the right level, folding a cloth smoothly, operating a zipper — these have subtle force and position requirements that human demonstrators encode implicitly. RL would require you to specify all of this in a reward function; IL learns it from the example.

**Safety.** IL never explores. The policy only produces actions that look like the demonstrations. This makes real-robot training much safer than RL, which inherently tries random actions during exploration.

**Data efficiency at small scale.** Modern generative IL methods (ACT, Diffusion Policy) extract enormous information from 10–50 demonstrations by modeling the full distribution of actions, not just the mean. This is a qualitative shift from old BC approaches that averaged over demonstrations.

### ❌ IL fundamentally struggles with

**Distribution shift.** The moment the robot reaches a state not covered by demonstrations — even slightly — the policy degrades. Small errors compound: a 1% deviation at step 1 becomes a 10% deviation at step 10. ACT addresses this with chunking; Diffusion Policy with multimodal coverage; but no pure BC method eliminates it.

**Generalization beyond the demonstration distribution.** If the demonstrator always picked objects from the center of the bin, the policy will struggle when objects are at the edges. If demonstrations used one camera angle, a slightly different angle at test time causes failures. IL policies are brittle to distribution shift in ways that are hard to anticipate.

**Exceeding demonstrator performance.** The policy is bounded above by the quality of the demonstrator. If the human was inconsistent, tired, or suboptimal, the policy will average over those inconsistencies rather than learning the best behavior. The path beyond this limit is [[hybrid-il-rl]].

**Long-horizon tasks with many steps.** Compounding errors are catastrophic over long sequences. A 10-step task with 95% per-step accuracy yields only 60% end-to-end success. Mamba2Diff and hierarchical IL methods address this, but long-horizon IL remains harder than long-horizon RL.

### ⚠️ Common pitfalls

- **Don't collect inconsistent demonstrations.** Quality matters more than quantity. 20 high-quality, consistent demos are worth more than 100 inconsistent ones. Standardize the demonstration protocol (same grasp approach, same speed, same camera conditions).
- **Don't use vanilla BC for dexterous tasks.** The multimodal action distribution (multiple valid grasp approaches) will cause vanilla BC to average to an invalid middle ground. Use Diffusion Policy or ACT which model the full distribution.
- **Don't confuse action chunking with temporal smoothing.** ACT's chunks predict *k* future actions jointly as a coherent trajectory — this is qualitatively different from predicting one action at a time. The CVAE latent variable encodes which "mode" of behavior is being executed.
- **Watch the sim-to-real gap for IL.** Unlike RL where you know the gap exists, IL policies are trained on real demonstrations but can still fail if the test-time visual conditions differ from demo conditions (lighting, camera angle, object appearance).

### 📊 IL in the SO-100 experiments

| Method | Status | Key question |
|--------|--------|-------------|
| ACT | 🔄 Pending | Can 10–30 demos on SO-100 match SAC's 92% sim result? |
| SmolVLA | 🔄 Pending | Does VLA fine-tuning generalize better than ACT to new object positions? |

*The central hypothesis being tested: IL (without reward design, without simulator) can match RL performance on a fixed-setup P&P task, while being deployable on real hardware faster.*

---

## Core Concepts

**Behavioral Cloning (BC):** Supervised learning on (state, action) pairs from demonstrations. Simple and fast but suffers from distribution shift — errors compound when the robot reaches states not seen in training.

**Flow Matching:** A continuous-time generative model that learns a deterministic vector field transporting samples from a simple prior to the data distribution. Supersedes diffusion in recent VLAs (π0, SmolVLA, X-VLA) — generates action chunks in as few as **10 steps** vs. 100 for DDPM, with comparable quality. DMs are a special case of FM. See: [[vision-language-action-models]].

**Goal-Conditioned IL (GCIL):** Conditions the policy on a target goal, allowing the same demonstrations to be relabeled for multiple goal configurations.

**Long-horizon tasks:** A key challenge for IL — demonstrations must cover complex multi-stage behaviors. Addressed by hierarchical methods ([[#Relay-Policy-Learning]]) and diffusion-based temporal modeling ([[#Mamba2Diff]]).

---

## Surveys

### A Survey of Imitation Learning: Algorithms, Recent Developments, and Challenges
**Zare, Kebria, Khosravi, Nahavandi — 2023**

Comprehensive survey of IL for robotics and AI. Covers the full spectrum: behavioral cloning, inverse RL, DAgger, GAIL, goal-conditioned IL, and hybrid methods. Discusses challenges including distribution shift, demonstration quality, and scalability.

*Tags:* survey, behavioral cloning, inverse RL, DAgger, GAIL, 2023

---

### A Survey on Imitation Learning for Contact-Rich Tasks in Robotics
**Tsuji, Kato, Solak, Zhang, Petrič, Nori, Ajoudani — Saitama / IIT / Jožef Stefan / Google DeepMind, 2026**

Focused survey on IL specifically for **contact-rich manipulation** — tasks involving continuous physical interaction (assembly, peg-in-hole, wiping, surgical procedures). 36-page IJRR survey covering the full pipeline from data collection to deployment.

**Teaching methods taxonomy:**
- **Kinesthetic teaching** — hand-guiding the robot directly
- **Teleoperation** — bilateral control transmitting both position and force
- **VR-based teaching** — capturing movements in virtual space
- **Observation methods** — cameras/motion capture observing human demos

**Data modalities for contact tasks:**
- Position/proprioception (baseline)
- Force/torque (essential for insertion, assembly)
- Vision RGB/RGB-D (contextual, but blind to contact forces)
- Tactile (fine-grained contact geometry, slip detection — still mostly research-only)
- EMG signals (muscle activation → stiffness modulation)

**8 IL paradigm categories covered:**
1. Behavior Cloning — ACT, Diffusion Policy, LSTM-based
2. Dynamic Movement Primitives (DMPs) — ProMPs, KMPs, FA-ProDMP
3. Generative methods — VAE, diffusion, Transformers
4. Foundation models — VLMs, VLAs (RT-1, RT-2, RoboFlamingo)
5. Inverse RL / GAIL — reward inference from expert data
6. Multimodal IL — force + vision + tactile integration
7. Offline RL — CQL, Q-Transformer, IQL pre-training + SAC fine-tuning
8. Other — world models, zero/one-shot IL, Riemannian manifolds

**Algorithm selection by sensor modality:**

| Sensor | Recommended approaches | Key advantages | Limitations |
|--------|----------------------|----------------|-------------|
| Force/Torque | BC with force features, force-coupled DMPs, IRL | Direct contact observation | Expensive sensors |
| Vision (RGB/RGB-D) | Visual BC, diffusion policies | Rich semantic info, scalable | Implicit contact, occlusion |
| Tactile | Conditioned BC, hybrid vision-tactile | Fine-grained contact, slip detection | Limited area, costly |
| Proprioception only | DMPs, kinesthetic teaching | Simple, reliable | Limited object generalization |

**Applications:** Industrial (peg-in-hole, insertion, polishing, deburring), household (wiping, cloth manipulation, door/drawer opening), healthcare (surgical bone-grinding, rehabilitation, dressing assistance).

**Core open challenges:**
1. **Hierarchical architectures** — no unified design principle; System 1 (fast/reactive) + System 2 (slow/planning) duality from cognitive science is a promising framework
2. **Multimodal sensing** — tactile adoption still limited to research; hardware reliability and integration unsolved
3. **Sim-to-real gap** — contact dynamics are hard to simulate accurately; domain randomization and differentiable simulation are partial solutions

*Tags:* survey, contact-rich, force feedback, tactile, teaching methods, DMPs, multimodal IL, 2026 | See: [[grasping-and-manipulation]], [[hybrid-il-rl]], [[simulation-and-tools]]

---

## Landmark Papers

### Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT) → [[algo-act]]
**Zhao et al., 2023**

Introduced **Action Chunking with Transformers (ACT)**, a policy architecture that predicts chunks of future actions jointly, reducing compounding errors. Deployed on the **ALOHA** bimanual low-cost hardware platform. Achieves 80-90% success on delicate manipulation tasks (inserting a battery, ziplocking a bag) from only ~10 minutes of demonstrations per task.

*Architecture:* [[algo-act|ACT]] Transformer encoder-decoder | *Hardware:* ALOHA (dual UR5-style arms) | *Tags:* ACT, bimanual, action chunking, low-cost demo collection | See: [[grasping-and-manipulation]]

---

### Visuomotor Policy Learning via Action Diffusion (Diffusion Policy) → [[algo-diffusion-policy]]
**Chi, Feng, Du, Xu, Cousineau, Burchfiel, Song — Columbia/Toyota Research, 2023**

Introduced **Diffusion Policy**: the robot's visuomotor policy is represented as a conditional denoising diffusion process. The policy learns the gradient of the action-score function and iteratively denoises at inference to produce actions. Benchmarked across 12 tasks from 4 manipulation benchmarks; achieves an average **46.9% improvement** over prior SOTA.

Key advantages: gracefully handles multimodal action distributions, high-dimensional action spaces, impressive training stability. Introduces receding horizon control and time-series diffusion transformer.

*Algorithm:* [[algo-diffusion-policy|DDPM]] (denoising diffusion probabilistic model) | *Tags:* diffusion, multimodal actions, visuomotor, 2023

---

### Behavior Generation with Latent Actions (VQ-BeT) → [[algo-vq-bet]]
**Lee, Wang, Etukuru, Kim, Shafiullah, Pinto — 2024**

Introduced **VQ-BeT (Vector-Quantized Behavior Transformer)**, which tokenizes continuous actions using a hierarchical vector quantization module. Extends Behavior Transformers (BeT) — which used k-means clustering — to handle high-dimensional action spaces and long sequences. Tested across 7 environments (manipulation, autonomous driving, robotics). **5x faster inference** than Diffusion Policy while improving mode capture.

*Architecture:* [[algo-vq-bet|VQ-BeT]] Transformer + hierarchical VQ | *Tags:* VQ-BeT, action tokenization, multimodal, 2024

---

### EasyMimic: A Low-Cost Framework for Robot Imitation Learning from Human Videos
**2026**

Converts human demonstration videos into robot trajectories using 3D hand tracking, then fine-tunes a robot policy via co-training. Uses the LeRobot ecosystem. Targets accessibility: demonstrations collected from ordinary human videos without motion capture suits.

*Tags:* human video, 3D hand tracking, co-training, LeRobot, 2026

---

### Data-Driven Planning via Imitation Learning
**2017**

Applies IL to robot planning: the agent learns to adapt its search strategy from expert demonstrations, rather than hand-designing planning heuristics. Early work bridging IL and motion planning.

*Tags:* planning, behavioral cloning, 2017

---

### Mamba2Diff: Enhanced Diffusion for Goal-Conditioned IL in Long-Horizon Tasks
**2026**

Combines diffusion-based policy generation with the Mamba2 state-space model architecture. Introduces the **BDGM (Bi-Directional Goal-conditioned Mamba)** module for improved long-horizon temporal modeling in GCIL settings.

*Algorithm:* [[algo-diffusion-policy|Diffusion]] + Mamba2 SSM | *Tags:* GCIL, long-horizon, state-space models, 2026

---

### CLIPORT: What and Where Pathways for Robotic Manipulation → [[algo-cliport]]
**Shridhar, Manuelli, Fox — CoRL 2021**

A **language-conditioned imitation-learning agent** combining a frozen, pretrained **CLIP** model (semantic "what" pathway) with the **Transporter** network's two-step pick-and-place primitive (spatial "where" pathway). Trained purely from demonstrations via cross-entropy on pick/place affordance heatmaps — no object detectors, segmentation, or pose estimation.

Achieves **>90%** average success on 10 simulated language-conditioned tasks with both streams, vs. 50% (Transporter-only, no language) or 76% (CLIP-only, no spatial precision) for either alone. A single multi-task model for all 10 tasks often **outperforms** per-task models (57% of evaluations). Validated on a real Franka Panda with just 179 image-action pairs across 9 tasks (~55-75% success).

*Architecture:* Two-stream FCN — frozen CLIP ResNet50 + CLIP sentence encoder (semantic) / Transporter ResNet (spatial), fused via tiled Hadamard conditioning | *Tags:* CLIP, language-conditioned, affordance prediction, pick-and-place, multi-task, 2021 | See: [[vision-language-action-models]], [[pick-and-place]]

---

## Comparison: Key IL Methods

| Method | Action Space | Multimodal | Long Horizon | Speed |
|--------|-------------|------------|--------------|-------|
| BC | Continuous | No | Poor | Fast |
| ACT | Continuous (chunked) | Partial | Good | Fast |
| Diffusion Policy | Continuous | Yes | Good | Slow |
| VQ-BeT | Discrete tokens | Yes | Good | 5x faster than Diffusion |
| Mamba2Diff | Continuous | Yes | Very good | — |
| CLIPORT | Pixelwise affordance (pick/place SE(2)) | No (argmax) | Step-by-step via language | Fast (FCN inference) |

---

## Related Topics
- [[hybrid-il-rl]] — combining IL with RL for improvement beyond demonstrations
- [[reinforcement-learning]] — reward-based alternative
- [[vision-language-action-models]] — VLA models that use IL at scale
- [[pick-and-place]] — P&P tasks learned via IL
- [[grasping-and-manipulation]] — bimanual and dexterous manipulation via IL
