# Wiki Index — Robotics & Robot Learning

A wiki compiled from research papers on robot manipulation, reinforcement learning, imitation learning, and foundation models for robotics. Built in parallel with experiments on the SO-100 robot arm.

**New here?** Start with [[overview]] — it explains the three paradigm families and how to navigate the wiki.  
**Choosing an approach?** Go directly to [[decision-guide]] for the decision flowchart and comparison tables.
**Want the experimental results directly?** Go to [[results]] for the SO-100 results dashboard.

---

## 📊 Wiki Status — At a Glance

An ensemble view of every major page: what it covers, and whether it's a finished synthesis (with this project's own results) or still reference-only / pending data.

| Page | What it covers | Status |
|------|-----------------|--------|
| [[overview]] | Three-paradigm framework, navigation guide, experiment tracker | ✅ Complete |
| [[decision-guide]] | Decision flowchart, per-family training/robustness guides, REX (full narrative results) | ✅ Complete |
| [[evaluation-protocol]] | Evaluation axes (ID/OOD/robustness), SO-100 test matrix, per-algorithm comparison | ✅ Complete — full 210-trial matrix still pending |
| [[results]] | Consolidated SO-100 results dashboard (SAC, ACT, SmolVLA) | ✅ Complete |
| [[pick-and-place]] | P&P synthesis across the 3 families + SO-100 results | ✅ Complete |
| [[reinforcement-learning]] | RL algorithms + critical synthesis ("when to use RL") | ✅ Complete |
| [[imitation-learning]] | IL algorithms + critical synthesis, contact-rich survey | ✅ Complete |
| [[vision-language-action-models]] | VLA models + critical synthesis | ✅ Complete |
| [[algo-smolvla]] | SmolVLA architecture, fine-tuning strategies, project results | ✅ Complete |
| [[algo-act]] | ACT architecture, project results | ✅ Complete |
| [[algo-cliport]] | CLIPORT architecture (CLIP + Transporter), literature only | 🔄 Reference only — no own experimental results |
| [[algo-sac]] | SAC deep-dive | ✅ Complete |
| [[grasping-and-manipulation]] | Grasping methods, contact-rich manipulation | 🔄 Reference only — no own experimental results |
| [[hybrid-il-rl]] | AWAC, Relay Policy, HITL-RL | 🔄 Reference only — not yet tested in this project |
| [[trajectory-planning]] | Motion planning algorithms | 🔄 Reference only |
| [[simulation-and-tools]] | Simulators, LeRobot tooling | 🔄 Reference only |
| [[world-models]] | World/action models | 🔄 Reference only — emerging area, no own experiments |
| [[llms-for-robotics]] | LLM/VLM foundation models for robotics | 🔄 Reference only |

**Legend:** ✅ Complete = has both literature synthesis and (where applicable) this project's own results. 🔄 Reference only = literature coverage exists but no SO-100 experiments have been run yet for this topic.

---

## Navigation & Synthesis Pages

| Page | Purpose |
|------|---------|
| [[overview]] | Entry point — three paradigm families, how to navigate, experimental results |
| [[decision-guide]] | Flowchart + comparison table — which approach for which situation; training workflows, robustness techniques, REX |
| [[evaluation-protocol]] | Standardized evaluation axes (ID, near-OOD, far-OOD, robustness, perturbation) + SO-100 test matrix |
| [[results]] | Results dashboard — consolidated SO-100 results for SAC, ACT, and SmolVLA |

---

## Algorithm Explainer Pages

Detailed pages covering mechanics, intuition, evaluation, results, and pros/cons for each algorithm.

### Reinforcement Learning Algorithms

| Page | Algorithm | Summary |
|------|-----------|---------|
| [[algo-ppo]] | PPO | Clipped surrogate objective; stable on-policy RL; most widely used robot RL algorithm |
| [[algo-her]] | HER | Hindsight relabeling for sparse/binary rewards; enables P&P from scratch |
| [[algo-sql-sac]] | SQL / SAC | Max-entropy RL; composable policies (SQL); stable off-policy actor-critic (SAC) |
| [[algo-sac]] | SAC | Dedicated SAC deep-dive: twin critics, auto-α, squashed Gaussian; 16700 HalfCheetah score |
| [[algo-dqn]] | DQN | Deep Q-network for discrete action P&P; replay buffer + target network |
| [[algo-cql]] | CQL | Conservative Q-Learning for offline RL; avoids OOD action overestimation |

### Hybrid IL + RL Algorithms

| Page | Algorithm | Summary |
|------|-----------|---------|
| [[algo-awac]] | AWAC / AW-Opt | Advantage-weighted regression; offline demos → online RL without catastrophic forgetting |
| [[algo-relay-policy]] | Relay Policy Learning | Hierarchical goal-conditioned IL+RL for long-horizon tasks from unstructured demos |
| [[algo-hitl-rl]] | HITL-RL | Human-in-the-loop RL with interactive corrections; near-perfect dexterity in 1-2.5h |

### Imitation Learning Algorithms

| Page | Algorithm | Summary |
|------|-----------|---------|
| [[algo-act]] | ACT | Action Chunking with Transformers + CVAE; 80-90% bimanual success from 10-min demos |
| [[algo-diffusion-policy]] | Diffusion Policy | DDPM-based visuomotor policy; +46.9% avg over SOTA; handles multimodal distributions |
| [[algo-vq-bet]] | VQ-BeT | Hierarchical vector-quantized action tokens; 5× faster than Diffusion Policy |
| [[algo-cliport]] | CLIPORT | Frozen CLIP (semantic) + Transporter (spatial) two-stream IL; >90% on language-conditioned pick-and-place; precursor to VLAs |

### VLA Models and Generalist Policies

| Page | Algorithm | Summary |
|------|-----------|---------|
| [[algo-smolvla]] | SmolVLA | ~450M VLA; SmolVLM-2 + flow-matching action expert; fine-tuning strategies (frozen backbone vs full fine-tune) and SO-100 project results |
| [[algo-openvla]] | OpenVLA | 7B open-source VLA; Llama2+DINOv2+SigLIP; beats RT-2-X (55B) by 16.5% |
| [[algo-octo]] | Octo | ~90M generalist policy; 800k demos; modular fine-tuning to new robots in hours |
| [[algo-vla-rl]] | VLA-RL | Online RL fine-tuning of OpenVLA; +4.5% on LIBERO; process reward model; inference scaling |
| [[algo-safevla]] | SafeVLA | CMDP safety alignment; 83.58% safety violation reduction; OOD generalization |
| [[algo-molmo2]] | Molmo2 | Open-weights VLM; video understanding + visual grounding; Molmo ecosystem foundation |
| [[algo-molmoact2]] | MolmoAct2 | Explicit reasoning chains before action tokens; real-world deployment focus |
| [[algo-molmobot]] | MolmoB0T | Zero-shot manipulation via simulation scale; >70% zero-shot success |
| [[algo-gemini-robotics-15]] | Gemini Robotics 1.5 | Thinking VLA + Embodied Reasoning Agent; Motion Transfer for zero-shot cross-embodiment transfer; ~80% agentic long-horizon progress |

### Motion Planning Algorithms

| Page | Algorithm | Summary |
|------|-----------|---------|
| [[algo-namr-rrt]] | NAMR-RRT | Neural adaptive multi-directional RRT; dynamic environments; 9× faster replanning |
| [[algo-stst]] | STST S-Curve | Jerk-bounded S-curve trajectory + forbidden-sphere avoidance; industrial standard |

### Tools and Enabling Techniques

| Page | Algorithm | Summary |
|------|-----------|---------|
| [[algo-qlora]] | QLoRA | 4-bit NF4 + LoRA; fine-tune 65B LLM on single 48GB GPU; enables VLA fine-tuning |
| [[algo-molmospaces]] | MolmoSpaces | 230k environments, 130k objects, 42M grasps; sim-to-real R=0.96; simulator-agnostic |

---

## Topic Pages

| Page | Summary |
|------|---------|
| [[reinforcement-learning]] | Core RL algorithms (PPO, SAC, DQN, SQL, MAPPO, CQL, HER) and their application to robot manipulation |
| [[imitation-learning]] | Learning from demonstrations: behavioral cloning, ACT, Diffusion Policy, VQ-BeT, Mamba2Diff, CLIPORT |
| [[hybrid-il-rl]] | Methods combining IL bootstrapping with RL fine-tuning: AW-Opt, Relay Policy Learning, HITL-RL |
| [[vision-language-action-models]] | VLA models: OpenVLA, SmolVLA, TinyVLA, SafeVLA, VLA-RL, X-VLA, GF-VLA, Octo, Molmo family |
| [[world-models]] | World and action models: GigaWorld-Policy, World Action Models (zero-shot), visual predictive models |
| [[pick-and-place]] | P&P as applied task — surveys, industrial, logistics, cluttered environments, long-horizon variants |
| [[grasping-and-manipulation]] | 6-DoF grasping (TSDF, Transformer), bimanual (ACT/ALOHA), dexterous manipulation, robot platforms |
| [[simulation-and-tools]] | Simulators and benchmarks: MuJoCo Playground, Isaac Sim, MolmoSpaces, MARTENS |
| [[trajectory-planning]] | Motion planning: STST S-curve, NAMR-RRT neural RRT, palletizing trajectory review |
| [[llms-for-robotics]] | Foundation models: LLM robot survey, Cosmos-Reason1, QLoRA, Molmo2, reward design |

---

## Source Summaries by Topic

### Surveys and Overviews

| Source | Year | Summary |
|--------|------|---------|
| A Survey of Imitation Learning: Algorithms, Recent Developments, and Challenges | 2023 | Comprehensive IL survey: BC, inverse RL, DAgger, GAIL, goal-conditioned IL, challenges in robotics |
| A Survey on Imitation Learning for Contact-Rich Tasks in Robotics | 2026 | IL survey focused on physical interaction tasks (assembly, insertion, wiping, surgical); teaching methods taxonomy; data modalities; algorithm selection by sensor type |
| A Survey of Robot Intelligence with Large Language Models | 2024 | LLM/VLM survey across 5 categories: reward design, low-level control, planning, manipulation, scene understanding |
| Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes | 2025 | Modern DRL survey focused on real-world deployment challenges and successes |
| Reinforcement Learning for Pick and Place Operations in Robotics: A Survey | 2021 | RL algorithms for P&P: MDP formulation, policy optimization, pose estimation |
| Trajectory Planning for Robotic Manipulators in Automated Palletizing: A Comprehensive Review | 2025 | Survey of trajectory optimization algorithms for palletizing P&P robots |

---

### Core RL Algorithms

| Source | Year | Summary |
|--------|------|---------|
| Proximal Policy Optimization Algorithms | 2017 | PPO: clipped surrogate objective for stable, sample-efficient on-policy RL |
| Hindsight Experience Replay (HER) | 2018 | HER: relabels failed episodes as successes to learn from sparse rewards |
| Multi-Goal Reinforcement Learning: Challenging Robotics Environments | 2018 | OpenAI Fetch arm + Shadow Hand benchmarks with sparse binary rewards |
| Composable Deep Reinforcement Learning for Robotic Manipulation | 2018 | SQL (Soft Q-Learning): max-entropy RL with composable policy arithmetic |
| Actionable Models: Unsupervised Offline RL of Robotic Skills | 2021 | Goal-conditioned offline Q-learning with hindsight relabeling; no online interaction needed |
| A Reinforcement Learning-Based Framework for Robot Manipulation Skill Acquisition | 2020 | MAPPO + OCM reward (Euclidean + Pearson) for 3 UR5 manipulation tasks |
| Reinforcement Learning for Collaborative Robots Pick-and-Place Applications: A Case Study | 2022 | RL + computer vision for cobots in human-shared industrial workspaces |
| Reinforcement Learning for Robot Manipulation Using CQL/SAC | 2025 | CQL+SAC for HRC P&P: 100% sim, 80% real success; CQL stabilizes offline data |
| The Task Decomposition and Reward-System RL for Pick-and-Place | 2023 | SAC with subtask decomposition and axis-based reward shaping; 93.2% success in MuJoCo |
| Simulated and Real Robotic Reach, Grasp, and Pick-and-Place Using RL | 2023 | PPO+SAC on Franka Panda; combined RL and traditional controls for sim-to-real |
| MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control | 2026 | Open-source RL framework for motion imitation from reference trajectories |

---

### Imitation Learning

| Source | Year | Summary |
|--------|------|---------|
| Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT/ALOHA) | 2023 | ACT (action chunking transformers) on ALOHA dual-arm; 80-90% on delicate tasks from 10-min demos |
| Visuomotor Policy Learning via Action Diffusion (Diffusion Policy) | 2023 | Diffusion Policy: DDPM for robot actions; 46.9% avg improvement; handles multimodal distributions |
| Behavior Generation with Latent Actions (VQ-BeT) | 2024 | VQ-BeT: hierarchical vector quantization for action tokens; 5× faster than Diffusion Policy |
| EasyMimic: A Low-Cost Framework for Robot IL from Human Videos | 2026 | 3D hand tracking from video → robot co-training via LeRobot; no motion capture needed |
| Data-driven Planning via Imitation Learning | 2017 | IL for robot planning; learns search strategy adaptation from expert demonstrations |
| Mamba2Diff: Enhanced Diffusion for Goal-Conditioned IL in Long-Horizon Tasks | 2026 | Diffusion + Mamba2 SSM + BDGM module for GCIL long-horizon action modeling |
| CLIPORT: What and Where Pathways for Robotic Manipulation | 2021 | Frozen CLIP (semantic) + Transporter (spatial) two-stream IL; >90% on language-conditioned pick-and-place; 179-demo real-robot transfer |

---

### Hybrid IL + RL

| Source | Year | Summary |
|--------|------|---------|
| AW-Opt: Learning Robotic Skills with Imitation and Reinforcement at Scale | 2021 | AWAC + QT-Opt hybrid; positive filtering; scales IL+RL to real robot tasks |
| Relay Policy Learning: Solving Long-Horizon Tasks via IL and RL | 2019 | Hierarchical goal-conditioned policies from unstructured demos, fine-tuned with RL; kitchen env |
| Precise and Dexterous Manipulation via Human-in-the-Loop RL (HITL-RL) | 2024 | Human corrections + RL; near-perfect dexterity in 1-2.5h; 2× better than pure IL |
| Hybrid Robot Learning for Automatic Robot Motion Planning in Manufacturing | 2025 | IL+RL for industrial motion planning at GE Aerospace |

---

### Vision-Language-Action Models

| Source | Year | Summary |
|--------|------|---------|
| Robot Learning: A Tutorial | 2025 | HuggingFace/Oxford tutorial covering RL → BC → generalist VLAs; introduces LeRobot ecosystem, π0 and SmolVLA architectures, flow matching for actions, HIL-SERL, RLPD |
| Octo: An Open-Source Generalist Robot Policy | 2024 | Transformer trained on 800k Open X-Embodiment trajectories; 9 platforms; fine-tunable in hours |
| OpenVLA: An Open-Source Vision-Language-Action Model | 2024 | 7B VLA on 970k demos; Llama2+DINOv2+SigLIP; outperforms RT-2-X (55B) by 16.5% |
| SmolVLA: A VLA for Affordable and Efficient Robotics | 2025 | Single-GPU VLA; async inference; matches VLAs 10× larger |
| TinyVLA: Fast, Data-Efficient VLA Models | 2025 | No pretraining needed; diffusion decoder; faster and more data-efficient than OpenVLA |
| SafeVLA: Towards Safety Alignment of VLA via Constrained Learning | 2025 | CMDP safety alignment; 83.58% cost reduction; robust to out-of-distribution perturbations |
| VLA-RL: Towards Masterful Robotic Manipulation with Scalable RL | 2025 | Online RL fine-tuning of OpenVLA-7B; +4.5% on LIBERO; process reward model; inference scaling |
| X-VLA: Soft-Prompted Transformer as Cross-Embodiment VLA | 2025 | Soft prompts for cross-embodiment; 0.9B; SOTA across 6 sims and 3 real robots |
| Information-Theoretic Graph Fusion with VLA for Policy Reasoning (GF-VLA) | 2026 | Scene graph + VLA for dual-arm; 94% grasp success; 90% task success |
| MolmoAct2: Action Reasoning Models for Real-World Deployment | 2026 | Molmo-family action reasoning for real-world robot deployment |
| Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer | 2025 | Two-model family (GR 1.5 VLA + GR-ER 1.5 VLM); Motion Transfer for zero-shot cross-embodiment; Thinking VLA with reasoning traces; ~80% agentic task progress vs 44% VLA-only |

---

### World Models and Generalist Policies

| Source | Year | Summary |
|--------|------|---------|
| GigaWorld-Policy: An Efficient Action-Centered World-Action Model | 2026 | Action-centered WAM; 9× faster than Motus; 7% higher task success |
| World Action Models are Zero-Shot Policies | 2026 | World model trained at scale → zero-shot robot policies via planning |
| MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation | 2026 | Simulation scale enables zero-shot manipulation in Molmo ecosystem |
| Learning High-Level Robotic Manipulation Actions with Visual Predictive Model | 2024 | Visual predictive model + action decomposer for high-level P&P planning |
| Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning | 2026 | NVIDIA foundation model for physical common sense and embodied reasoning |

---

### Pick-and-Place Applications

| Source | Year | Summary |
|--------|------|---------|
| Automated Trajectory Planner of Industrial Robot for Pick-and-Place Task | 2013 | STST S-curve trajectory + forbidden-sphere avoidance for PUMA 560 |
| Development of a Methodology to Improve Multi-Robot Pick & Place Applications | 2016 | Multi-robot P&P scheduling simulation; validated experimentally |
| Pick and Place Operations in Logistics Using a Mobile Manipulator + DRL | 2019 | DRL mobile manipulator for warehouse P&P; end-to-end learned policy |
| Deep RL Applied to a Robotic Pick-and-Place Application | 2021 | DQN + MobileNet (84% success); ROS Cobot |
| Learning Pick to Place Objects Using Self-Supervised Learning | 2021 | DQN on UR5 with RGB-D; minimal compute required |
| Prehensile and Non-Prehensile P&P in Clutter Using DRL | 2023 | DQN + DenseNet-121 + FCN; push then grasp in cluttered environments |
| Intelligent Pick-and-Place System Using MobileNet | 2023 | MobileNet CNN for real-time object recognition in P&P |
| Monocular Camera-Based Robotic P&P in Fusion Applications | 2023 | Monocular camera + forward kinematics; DRL; fusion facility application |
| Reinforcement Learning for Collaborative Robots P&P | 2022 | RL + CV for cobots in human-shared workspaces |

---

### Grasping

| Source | Year | Summary |
|--------|------|---------|
| Bio-Inspired Affordance Learning for 6-DoF Robotic Grasping | 2025 | Transformer + TSDF input for 6-DoF grasp prediction; outperforms VGN; Franka Panda |
| Vision-Based Robotic Object Grasping — A DRL Approach | 2023 | SAC + YOLO for 6-DOF industrial grasping with variety of objects |

---

### Simulation and Tools

| Source | Year | Summary |
|--------|------|---------|
| MuJoCo Playground | 2025 | Open-source GPU-accelerated sim framework; minutes-to-policy; zero-shot sim-to-real |
| An Easy-to-Use DRL Library for AI Mobile Robots in Isaac Sim | 2022 | OpenAI Gym + SB3 API on Isaac Sim for mobile robot DRL |
| MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation | 2026 | 230k envs, 130k objects, 42M grasps; sim-to-real R=0.96; simulator-agnostic |
| In-Simulation Testing of DL Vision Models in Robotic Manipulators (MARTENS) | 2024 | Evolutionary adversarial testing in Isaac Sim; 25-50% more failures detected |

---

### Trajectory Planning

| Source | Year | Summary |
|--------|------|---------|
| Automated Trajectory Planner of Industrial Robot for Pick-and-Place Task | 2013 | STST S-curve + forbidden-sphere for PUMA 560 P&P |
| Trajectory Planning for Robotic Manipulators in Automated Palletizing: A Review | 2025 | Survey of placement + minimum-time trajectory algorithms for palletizing |
| NAMR-RRT: Neural Adaptive Motion Planning for Mobile Robots | 2025 | Neural heuristic + multi-directional RRT for dynamic environments |

---

### LLMs and Foundation Models for Robotics

| Source | Year | Summary |
|--------|------|---------|
| A Survey of Robot Intelligence with Large Language Models | 2024 | LLM survey: reward design, control, planning, manipulation, scene understanding |
| Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning | 2026 | NVIDIA physical common sense foundation model for embodied reasoning |
| QLoRA: Efficient Finetuning of Quantized LLMs | 2023 | 4-bit quantization + LoRA; fine-tune 65B model on single 48GB GPU |
| Molmo2: Open Weights and Data for VLMs | 2026 | Open-weights VLM with video understanding and grounding; Molmo ecosystem foundation |
