# Vision-Language-Action Models (VLAs)

Vision-Language-Action models (VLAs) extend large vision-language models (VLMs) with action prediction heads, enabling robots to be controlled by natural language instructions. They are pretrained on massive internet-scale vision-language data and robot demonstration datasets, then fine-tuned for specific robot setups. VLAs represent the current frontier of generalist robot policies. See also [[world-models]], [[llms-for-robotics]], [[imitation-learning]].

→ **Choosing between RL, IL, and VLAs?** See [[decision-guide]].

---

## When to use VLAs — Critical Synthesis

### ✅ VLAs genuinely excel at

**Semantic generalization.** This is the unique capability that neither RL nor IL has. A fine-tuned OpenVLA can correctly interpret "pick the mug on the left" vs "pick the red cup" without task-specific training per instruction. It understands *what* you mean because it has seen billions of images and text descriptions on the internet. No amount of RL or IL training produces this from scratch.

**Novel object generalization.** VLAs trained on diverse demonstration datasets generalize to objects not seen during fine-tuning — new colors, shapes, textures — because their visual backbone already understands object categories. RL policies and IL policies with limited demos cannot do this.

**Cross-embodiment transfer.** Models like Octo (800k trajectories, 9 platforms), X-VLA (soft prompting), and GR 1.5 (Motion Transfer) can transfer skills across different robot hardware with minimal additional data. RL and IL require retraining from scratch for each new robot.

**Reducing the data burden compared to training from scratch.** A SmolVLA fine-tuned on 50–100 demonstrations can match or exceed a specialized policy trained on thousands of demonstrations, because the backbone already knows how to perceive and reason about the world.

### ❌ VLAs fundamentally struggle with

**Geometric precision.** VLAs are strong at semantic tasks but weak at precise insertion, assembly, or tight tolerance placement. This is because their training signal is mostly image-level supervision — they predict joint configurations without modeling contact forces. Contact-rich tasks remain a major gap (see the [[imitation-learning#contact-rich-survey]] for why this is hard).

**High-frequency control.** Autoregressive token generation at 7B parameters runs at 1–6 Hz. Control tasks requiring >10 Hz feedback (dexterous grasping, dynamic catching, contact-rich assembly) are out of reach without special architectures (async inference, smaller action experts like SmolVLA's ~100M parameter head).

**Data quality sensitivity.** Unlike RL (which explores and corrects), VLAs are pure imitation — they learn exactly what their fine-tuning demos show. Poor demonstrations, inconsistent language annotations, or mismatched camera setups will silently degrade performance in ways that are hard to debug.

**Compute cost.** Even SmolVLA (450M) requires a GPU for training and reasonable inference. A 7B model like OpenVLA requires 4–8x A100 GPUs for fine-tuning (or clever quantization via QLoRA on a single 48GB GPU). This is a real barrier compared to ACT or Diffusion Policy which train on consumer hardware.

### ⚠️ Where the field is heading

VLAs are advancing faster than any other paradigm in this wiki. The trajectory: OpenVLA (7B, open-source, beats RT-2-X 55B) → SmolVLA (450M, community data, 40% faster) → GR 1.5 (multi-embodiment, thinking VLA, agentic system). The key open problems are:
- **Closing the precision gap**: adding contact-rich feedback (force, tactile) to VLA training
- **Inference speed**: async inference stacks and small action experts are promising
- **Evaluation standardization**: most VLAs are evaluated on proprietary setups; LIBERO is emerging as the open benchmark

### 📊 VLAs in the SO-100 experiments

| Model | Status | Result |
|-------|--------|-------------|
| [[algo-smolvla\|SmolVLA]] | ✅ Done (Dataset_v4) | **58%** in-distribution, **0%** with distractor (3 near-successes) — see [[results]] |

**Outcome vs hypothesis:** the original hypothesis was that SmolVLA would generalize better than ACT to new positions/objects thanks to pretraining. In practice, at ~111 fine-tuning demos, **ACT outperformed SmolVLA on every tested condition**, including the distractor case where SmolVLA dropped to 0%. The leading explanation is fine-tuning-strategy-dependent: SmolVLA was fully fine-tuned (no frozen layers), which may have eroded the pretrained backbone's robustness to novel scene elements. See [[algo-smolvla]] for the fine-tuning options analysis and a proposed follow-up with a frozen-backbone strategy.

---

## The VLA Paradigm

The key insight: instead of training a visuomotor policy from scratch, inherit the visual and linguistic understanding of large pretrained models (LLMs + VLMs), then adapt them to predict robot actions. This enables:
- Language-conditioned control ("pick up the red cup")
- Generalization across novel objects, scenes, and instructions
- Fine-tuning to new robots with relatively few demonstrations

The main tradeoff: large models (7B+ parameters) are slow at inference and expensive to train, motivating smaller and more efficient VLAs.

---

## Precursor: CLIPORT → [[algo-cliport]]
**Shridhar, Manuelli, Fox — CoRL 2021**

Before "VLA" was a term, **CLIPORT** demonstrated the core idea that defines the paradigm above: graft a **frozen, internet-pretrained vision-language model (CLIP)** onto a robot-specific action network to get language-conditioned manipulation, without retraining the semantic backbone from scratch. Its two-stream architecture (CLIP "what" pathway + Transporter "where" pathway) achieved >90% on language-conditioned pick-and-place tasks in simulation and was validated on a real Franka Panda with only 179 demonstrations across 9 tasks.

Architecturally it differs sharply from modern VLAs — it predicts dense pixelwise pick/place *affordances* (SE(2)) rather than continuous joint trajectories via autoregressive tokens or flow matching — but the **frozen-pretrained-backbone-for-semantics** idea reappears directly in [[algo-smolvla|SmolVLA]]'s fine-tuning strategy options. See [[algo-cliport]] for full detail.

---

## Generalist Policies

### Octo: An Open-Source Generalist Robot Policy → [[algo-octo]]
**Octo Model Team — UC Berkeley / Stanford / Google DeepMind, 2024**

The first large-scale open-source generalist robot policy. A **Transformer-based** policy trained on **800k trajectories** from the **Open X-Embodiment** dataset — the largest robot manipulation dataset at the time. Accepts language commands or goal images. Fine-tunable to new observation/action spaces within **a few hours** on standard consumer GPUs.

Evaluated across **9 robotic platforms**. Demonstrates Octo as a versatile initialization for diverse robot setups. Includes detailed ablations on architecture and training data choices.

*Architecture:* Transformer | *Data:* Open X-Embodiment, 800k trajectories | *Tags:* generalist, open-source, multi-embodiment, fine-tuning, 2024

---

### OpenVLA: An Open-Source Vision-Language-Action Model → [[algo-openvla]]
**Kim, Pertsch, Karamcheti et al. — Stanford / UC Berkeley / Google, 2024**

A **7B-parameter** open-source VLA trained on **970k real-world robot demonstrations**. Built on **Llama 2** + visual encoder fusing **DINOv2** and **SigLIP** features.

Key results:
- Outperforms **RT-2-X (55B parameters) by 16.5%** on 29 tasks across multiple robot embodiments — with **7x fewer parameters**
- Outperforms Diffusion Policy by 20.4% in multi-task multi-object environments
- Fine-tunable via [[algo-qlora|QLoRA]] on consumer GPUs
- Served efficiently via quantization without loss in task success

*Architecture:* Llama 2 + DINOv2 + SigLIP | *Parameters:* 7B | *Tags:* open-source, fine-tuning, QLoRA, language grounding, 2024 | See: [[llms-for-robotics]]

---

## Generalist Policy Evolution

The generalist VLA landscape has followed a clear trajectory: RT-1 (130k demos, 13 robots) → RT-2 (internet-scale VLM + robot data) → OpenVLA (7B open-source, beats RT-2-X) → **π0** (10M+ demos, flow matching, physical intelligence) → **SmolVLA** (450M, community data, open). The key shift: from autoregressive token prediction to **flow matching** as the action generation backbone.

---

### π0: A Vision-Language-Action Flow Model for General Robot Control
**Black, Brown, Driess et al. — Physical Intelligence, 2024**

The leading open generalist VLA as of 2025. Introduces **flow matching** (instead of diffusion or autoregressive decoding) for continuous action generation, trained on **10M+ trajectories** — the largest robotics pretraining dataset to date.

**Architecture — Mixture of Experts (MoE):**
- **VLM backbone** (pre-trained, e.g. PaliGemma): processes images + language instructions into tokens
- **Action expert** (~dedicated flow matching network): denoises action chunks conditioned on VLM tokens
- **Blockwise causal attention masking**: prevents VLM tokens from attending to action tokens — enables KV caching across denoising steps for faster inference
- **β-CVAE style action chunking**: predicts Ha future actions jointly (same spirit as ACT)

**Training:**
- Flow matching loss over both backbone and action expert jointly
- Timestep τ sampled from Beta(1.5, 1) on [0, s] — emphasizes noisy samples, focuses learning on mean reconstruction
- Pretraining on proprietary π dataset (~91% private) + Open-X + DROID; fine-tuning on narrow high-quality task data

**Key results:**
- 3.3B parameters
- **10 denoising steps** at inference — much faster than diffusion
- Strong cross-embodiment: zero-pads DoF for robots with fewer joints; uses 3 fixed camera views
- Pre-train + fine-tune consistently outperforms training from scratch per task
- Foundation for downstream models: π0-FAST (distilled for speed, used as baseline in VLA-RL benchmark)

*Architecture:* MoE (VLM backbone + flow matching action expert) | *Training data:* 10M+ trajectories | *Tags:* flow matching, generalist, pre-train+adapt, cross-embodiment, 2024 | See: [[llms-for-robotics]]

---

## Efficient VLAs

### SmolVLA: A VLA for Affordable and Efficient Robotics → [[algo-smolvla]]
**Shukor et al. — HuggingFace, 2025**

Addresses the cost barrier of large VLAs like π0 (3.3B params). SmolVLA targets accessible hardware: trains on a **single GPU**, deploys on **consumer GPUs or CPUs**.

**Architecture:**
- **SmolVLM-2 backbone** (SigLIP vision encoder + SmolLM2 language decoder) — compact pre-trained VLM
- **~100M parameter action expert** with interleaved self-attention + cross-attention layers (vs π0's pure self-attention)
- **Flow matching** for action generation (same as π0) — 10 denoising steps at inference
- **450M total parameters** vs π0's 3.3B
- **Async inference stack**: decouples action prediction from execution for higher control rates on modest hardware

**Data:** 450+ community datasets (SO-100/SO-101 platforms), 20k+ trajectories. Re-annotated with a small VLM to fix noisy/missing instructions.

**Results vs π0:**
- **40% faster inference**
- **6× less memory**
- Comparable task success across real-world and simulated benchmarks

Releases all code, pretrained models, and training data (fully open).

*Architecture:* SmolVLM-2 + flow matching action expert | *Parameters:* ~450M | *Tags:* efficient, affordable, community-driven, async inference, flow matching, 2025 | See: [[simulation-and-tools]]

---

### TinyVLA: Towards Fast, Data-Efficient VLA Models for Robotic Manipulation
**Wen et al., 2025**

A compact VLA family that is (1) **faster at inference** and (2) more **data-efficient** than OpenVLA — eliminating the need for a separate pretraining stage. Integrates a **[[algo-diffusion-policy|diffusion policy decoder]]** during fine-tuning for precise robot actions. Initializes the policy backbone with fast multimodal models.

Outperforms OpenVLA in speed and data efficiency while matching or exceeding performance. Strong generalization: novel objects, unseen positions, appearance changes, background variations.

*Architecture:* Compact VLM + diffusion decoder | *Tags:* fast inference, data-efficient, diffusion decoder, 2025

---

## Safety-Aligned VLAs

### SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning → [[algo-safevla]]
**Zhang et al., 2025**

The first systematic safety alignment framework for VLAs. Addresses real-world deployment safety risks (harm to environment, robot, humans). Proposes **ISA (Integrated Safety Approach)**:
1. Model safety requirements systematically
2. Elicit diverse unsafe behaviors actively
3. Constrain VLA policies via safe RL using **CMDP (Constrained Markov Decision Process)** — a min-max optimization perspective
4. Evaluate rigorously via targeted benchmarks

Results: **83.58% reduction in cumulative safety violations** vs. state-of-the-art, while maintaining or improving task success rate (+3.85%). Strong generalization to out-of-distribution perturbations. Evaluated on long-horizon mobile manipulation.

*Algorithm:* CMDP, constrained safe RL | *Tags:* safety, alignment, long-horizon, mobile manipulation, 2025

---

## RL-Enhanced VLAs

### VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning → [[algo-vla-rl]]
**Lu et al. — Tsinghua University, 2025**

Addresses a key limitation of offline-trained VLAs: failure on out-of-distribution states. Introduces **VLA-RL**, an online RL framework for fine-tuning pretrained auto-regressive VLAs.

Key contributions:
- **Trajectory-level RL formulation**: models manipulation as multi-modal multi-turn conversation
- **Vision-language process reward model**: fine-tuned VLM providing pseudo-reward labels on automatically extracted task segments (addresses sparse reward challenge)
- **Implementation findings**: curriculum selection, GPU-balanced vectorized environments, batch decoding, critic warmup

Results: OpenVLA-7B surpasses strongest fine-tuned baseline by **4.5% on 40 tasks** in LIBERO benchmark. Matches commercial model π0-FAST. Exhibits inference scaling laws (better results with more test-time optimization).

*Base model:* [[algo-openvla|OpenVLA-7B]] | *Algorithm:* Online RL + process reward model | *Tags:* RL fine-tuning, sparse rewards, scaling laws, 2025 | See: [[reinforcement-learning]], [[hybrid-il-rl]], [[algo-openvla]]

---

## Thinking VLAs and Agentic Systems

### Gemini Robotics 1.5 — Thinking VLA + Embodied Reasoning Agent → [[algo-gemini-robotics-15]]
**Gemini Robotics Team — Google DeepMind, 2025**

A two-model family: **GR 1.5** (multi-embodiment VLA) and **GR-ER 1.5** (embodied reasoning VLM), combined into a full agentic system. Three core innovations:

1. **Motion Transfer (MT):** New training recipe enabling zero-shot skill transfer across very different robot embodiments (ALOHA bimanual, Bi-arm Franka, Apollo humanoid) from a single checkpoint
2. **Embodied Thinking:** GR 1.5 interleaves natural-language thinking traces with action tokens — "think before act" decomposition boosts multi-step progress by ~50–100% relative
3. **Agentic system:** GR-ER 1.5 as orchestrator (planning, tool use, web search, success detection) + GR 1.5 as action model → ~80% progress on long-horizon tasks vs. ~44% for VLA-only

GR-ER 1.5 achieves SOTA on embodied reasoning benchmarks, outperforming Gemini 2.5 Pro and GPT-5. Reduces agentic planning failure rate from 25.5% (generic LLM orchestrator) to 9%.

*Architecture:* Gemini backbone + Motion Transfer training | *Robots:* ALOHA, Bi-arm Franka, Apollo Humanoid | *Tags:* thinking VLA, multi-embodiment, agentic, motion transfer, embodied reasoning, 2025 | See: [[llms-for-robotics]], [[simulation-and-tools]], [[algo-safevla]]

---

## Cross-Embodiment VLAs

### X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA
**Zheng et al. — Tsinghua University, 2025**

Proposes **soft prompting** for cross-embodiment generalization: separate sets of learnable embeddings (embodiment-specific prompts) are added per data source. The prompts empower the model to exploit heterogeneous, cross-embodiment data effectively. Uses a flow-matching-based architecture with standard Transformer encoders.

0.9B parameter instantiation (**X-VLA-0.9B**) achieves SOTA across **6 simulations and 3 real-world robots**. Excels at flexible dexterity and quick adaptation.

*Architecture:* Flow-matching Transformer + soft prompts | *Parameters:* 0.9B | *Tags:* cross-embodiment, soft prompts, flow-matching, 2025

---

### Information-Theoretic Graph Fusion with VLA Model for Policy Reasoning (GF-VLA)
**2026**

Integrates **scene graphs** with VLA for dual-arm robotic control. Uses information-theoretic graph fusion to reason about object relationships. Achieves **94% grasp success** and **90% task success** on dual-arm manipulation tasks. Enables more structured spatial reasoning than pure VLMs.

*Tags:* scene graphs, dual-arm, graph fusion, information theory, 2026 | See: [[grasping-and-manipulation]]

---

## Molmo Family of VLAs

### Molmo2: Open Weights and Data for VLMs with Video Understanding and Grounding → [[algo-molmo2]]
**2026**

Open-weights VLM with video understanding and visual grounding capabilities. Foundation model in the Molmo ecosystem used by MolmoAct2 and MolmoB0T.

*Tags:* VLM, video understanding, open weights, 2026

---

### MolmoAct2: Action Reasoning Models for Real-World Deployment → [[algo-molmoact2]]
**2026**

Extends [[algo-molmo2|Molmo2]] with action reasoning for real-world robot deployment. Focus on bridging the gap between vision-language understanding and concrete robot actions in uncontrolled settings.

*Tags:* action reasoning, real-world deployment, 2026 | See: [[algo-molmo2]]

---

### MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation → [[algo-molmobot]]
**2026**

Leverages large-scale simulation data (see [[algo-molmospaces|MolmoSpaces]]) to achieve zero-shot manipulation with VLA policies from the Molmo family. Demonstrates that simulation scale can compensate for lack of real-world demos.

*Tags:* zero-shot, large-scale simulation, 2026 | See: [[world-models]], [[simulation-and-tools]], [[algo-molmo2]], [[algo-molmospaces]]

---

## Comparison of VLA Models

| Model | Parameters | Open | Fine-tunable | Key Strength |
|-------|-----------|------|--------------|--------------|
| Octo | ~90M | Yes | Yes (hours) | Multi-platform, 800k demos |
| OpenVLA | 7B | Yes | Yes (QLoRA) | Beats RT-2-X 55B; strong language grounding |
| π0 | 3.3B | Partial | Pre-train+adapt | Flow matching; 10M+ demos; 10-step inference |
| SmolVLA | ~450M | Yes | Single GPU | 40% faster, 6× less memory than π0; community data |
| TinyVLA | <1B | Yes | Fast | Data-efficient; diffusion decoder |
| SafeVLA | — | Yes | CMDP | Safety alignment |
| VLA-RL | 7B (OpenVLA) | Yes | Online RL | Surpasses offline fine-tuning |
| X-VLA | 0.9B | Yes | Soft prompts | Cross-embodiment SOTA |
| GF-VLA | — | — | — | Scene-graph reasoning, dual-arm |
| GR 1.5 | — | No | Motion Transfer | Multi-embodiment; thinking VLA; ~80% agentic progress |

---

## Related Topics
- [[world-models]] — VLA models extended with world models
- [[llms-for-robotics]] — foundation models and LLMs enabling VLAs
- [[imitation-learning]] — VLAs rely on IL at scale (behavior cloning on demonstrations)
- [[hybrid-il-rl]] — VLA-RL combines VLA with online RL
- [[simulation-and-tools]] — evaluation environments for VLAs
