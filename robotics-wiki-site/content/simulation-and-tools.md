# Simulation and Tools

Robot learning research depends heavily on simulation environments for safe, scalable policy training. Simulators allow millions of interactions without hardware wear or safety risks. Key challenges: sim-to-real transfer (policies trained in sim must generalize to real robots), and benchmark diversity (does the policy generalize across tasks and robot types?). See also [[reinforcement-learning]], [[pick-and-place]], [[trajectory-planning]].

---

## Physics Simulators

### MuJoCo Playground
**Zakka, Tabanpour, Liao et al. — UC Berkeley / Google DeepMind / University of Toronto, 2025**

A fully open-source framework for robot learning built on **MJX** (MuJoCo's JAX-based GPU backend). Key features:
- `pip install playground` — immediate access
- Train policies in **minutes on a single GPU** via massive parallelization
- Supports diverse platforms: quadrupeds, humanoids, dexterous hands, robotic arms
- Enables **zero-shot sim-to-real transfer** from both state and pixel inputs
- Integrated stack: physics engine + batch renderer + training environments

*Tags:* MuJoCo, MJX, GPU training, sim-to-real, open-source, 2025

---

### An Easy-to-Use Deep Reinforcement Learning Library for AI Mobile Robots in Isaac Sim
**2022**

A DRL library built on **NVIDIA Isaac Sim** with **OpenAI Gym** API + **Stable Baselines 3 (SB3)** integration. Targets AI mobile robots (navigation + manipulation). Designed for accessibility: researchers familiar with Gym can immediately use Isaac Sim's GPU-accelerated simulation without writing low-level sim code.

*Simulator:* Isaac Sim | *Framework:* OpenAI Gym + Stable Baselines 3 | *Tags:* Isaac Sim, mobile robots, SB3, 2022

---

## Large-Scale Benchmarks

### MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation → [[algo-molmospaces]]
**Kim, Pumacay, Rayyan et al. — University of Washington, 2026**

The most comprehensive simulation ecosystem in this collection:
- **230,000+ diverse indoor environments** (handcrafted + procedurally generated multi-room houses)
- **130,000 richly annotated object assets**, including 48k manipulable objects with **42 million stable grasps**
- **Simulator-agnostic**: supports MuJoCo, Isaac, and ManiSkill
- Full embodied task spectrum: static/mobile manipulation, navigation, long-horizon multi-room tasks
- **MolmoSpaces-Bench**: 8 tasks with strong **sim-to-real correlation (R=0.96, ρ=0.98)**

Key findings: newer zero-shot policies outperform older ones; identified key sensitivities to prompt phrasing, initial joint positions, and camera occlusion.

*Tags:* benchmark, 230k environments, sim-to-real, open ecosystem, 2026

---

### Multi-Goal Reinforcement Learning: Challenging Robotics Environments → [[algo-her]]
**Plappert et al. — OpenAI, 2018**

Introduced a standard set of challenging continuous-control benchmark tasks integrated with OpenAI Gym:
- Fetch robotic arm: pushing, sliding, pick-and-place (sparse binary rewards)
- Shadow Dexterous Hand: in-hand object manipulation

These environments became standard benchmarks for [[algo-her|HER]] and goal-conditioned RL research.

*Environments:* Fetch P&P, Shadow Hand | *Tags:* benchmark, OpenAI Gym, sparse rewards | See: [[reinforcement-learning]], [[algo-her]]

---

## Testing and Validation

### In-Simulation Testing of Deep Learning Vision Models in Autonomous Robotic Manipulators (MARTENS)
**2024**

Introduces **MARTENS** — a framework for adversarially testing deep learning vision models in Isaac Sim before real deployment. Uses **evolutionary search** to find hard failure cases:
- Detects **25-50% more failures** than standard testing approaches
- Identifies edge cases in perception pipelines that would cause manipulation failures
- Reduces risk of real-world deployment failures

*Simulator:* Isaac Sim | *Tags:* adversarial testing, evolutionary search, vision models, 2024

---

## Robot Operating System (ROS)

ROS appears across multiple papers as the middleware for connecting perception, planning, and control on real hardware. Papers using ROS: Deep RL P&P (DQN+MobileNet), Sim+Real P&P (PPO/SAC + Franka Panda). ROS bridges the simulation-to-real gap by providing a consistent API.

---

## Common Simulation Environments Referenced

| Simulator | Key Papers | Key Strength |
|-----------|-----------|--------------|
| MuJoCo | MuJoCo Playground, Task Decomp RL, Composable DRL | Contact-rich physics; open-source |
| Isaac Sim (NVIDIA) | Isaac Sim DRL lib, MARTENS | GPU-accelerated; photorealistic |
| ManiSkill | MolmoSpaces | Dexterous manipulation focus |
| Gazebo | Various | ROS integration |
| PyBullet | Various | Lightweight; OpenAI Gym compatible |
| Robosuite | Task Decomp RL | Manipulation tasks on standard robots |
| OpenAI Gym | Multi-Goal RL, Isaac Sim lib | Standard API for RL environments |

---

## Sim-to-Real Transfer

A recurring challenge. Papers addressing it:
- **MuJoCo Playground**: zero-shot sim-to-real from state and pixel inputs
- **[[algo-molmospaces|MolmoSpaces]]**: R=0.96 correlation between sim and real performance
- **MARTENS**: adversarial sim testing to catch real-world failures early
- **Simulated and Real P&P**: validated [[algo-ppo|PPO]]/[[algo-sac|SAC]] transfer to Franka Panda real hardware
- **[[algo-cql|CQL]]/[[algo-sac|SAC]] HRC**: 100% sim → 80% real transfer

Sim-to-real gap arises from: visual appearance differences, contact dynamics, latency, sensor noise, and unmodeled environmental variation.

---

## Open-Source Robotics Tooling

### LeRobot — End-to-End Open-Source Robot Learning (HuggingFace)
**Capuano, Pascal, Zouitine, Wolf, Aractingi — University of Oxford / HuggingFace, 2025**

HuggingFace's vertically integrated open-source library for the full robot learning stack. The de facto standard toolkit for accessible robot learning research.

**LeRobotDataset format:**
- Unified multi-modal dataset format: tabular data (joint states, actions) + MP4 video streams + JSON metadata
- Native windowing API for action chunking and observation history
- Streaming mode for large datasets without local storage
- Supports SO-100, SO-101, ALOHA-2, humanoid arms, and simulation-based datasets

**Supported algorithms (all in PyTorch):**
- [[algo-act|ACT]] — action chunking with Transformers
- [[algo-diffusion-policy|Diffusion Policy]] — DDPM-based visuomotor policy
- [[algo-vq-bet|VQ-BeT]] — vector-quantized behavior transformer
- **π0** — flow matching VLA (Physical Intelligence)
- **SmolVLA** — compact VLA with async inference
- **HIL-SERL** — human-in-the-loop RL (same as [[algo-hitl-rl|HITL-RL]])
- **TD-MPC** — model-based RL

**Async inference stack:** Decouples policy server (runs on GPU, generates action chunks) from robot client (executes actions at control frequency). Enables deploying large VLAs on modest hardware by buffering predicted actions.

*Tags:* open-source, tooling, LeRobotDataset, async inference, 2025 | See: [[imitation-learning]], [[vision-language-action-models]], [[reinforcement-learning]]

---

## Related Topics
- [[reinforcement-learning]] — RL training relies on simulated environments
- [[pick-and-place]] — P&P is the primary task evaluated in simulators
- [[grasping-and-manipulation]] — grasping benchmarks (Fetch, Shadow Hand, Franka)
- [[trajectory-planning]] — trajectory algorithms validated in simulation
- [[world-models]] — world models learn environmental dynamics like simulators
- [[imitation-learning]] — LeRobot is the primary tooling for IL research
