# Pick-and-Place

Pick-and-place (P&P) is the canonical robot manipulation task: a robot arm grasps an object from one location and moves it to another. It appears in nearly every paper in this collection. P&P is studied as both an end goal (industrial automation, logistics) and a benchmark task for evaluating manipulation policies. See also [[grasping-and-manipulation]], [[reinforcement-learning]], [[trajectory-planning]].

→ **Not sure which approach to use?** See [[decision-guide]] for the full flowchart.

---

## Synthesis — Which approach for pick-and-place?

P&P is deceptively simple as a description but covers an enormous range of difficulty depending on context. The right approach depends on three key factors: **what generalization you need**, **whether a simulator is available**, and **how much demonstration data you can collect**.

### The three families applied to P&P

#### RL for P&P
RL is the natural fit when you have a simulator and can define a reward (e.g., distance to target + grasp success signal). The results in this collection are strong: SAC with task decomposition reaches **93.2%** (Kim et al., 2023) and **92% on SO-100** in simulation. CQL+SAC transfers to real hardware at 80%.

**RL works well for P&P when:**
- The task is fixed (same objects, same bin layout, same robot)
- A physics simulator is available (MuJoCo, Isaac Sim, Robosuite)
- You can decompose the task into subtasks with shaped rewards (approach → grasp → place)
- You need precise control (exact placement coordinates)

**RL struggles with P&P when:**
- Objects vary (new shapes, new materials, new positions at test time)
- No simulator is available — real-robot RL exploration damages hardware
- The scene is cluttered and contact dynamics are hard to simulate accurately
- You need to generalize to language instructions ("put the red mug next to the bowl")

**Key insight from the literature:** reward shaping matters enormously. Papers that decompose P&P into 3 subtasks (approach, grasp, place) with per-subtask rewards consistently outperform those with a single sparse success signal. HER is the go-to fix for sparse binary rewards without reward engineering.

#### IL for P&P
IL removes the reward design bottleneck. With 10–50 demonstrations via teleoperation (ALOHA, SpaceMouse), ACT and Diffusion Policy can achieve high performance on a fixed P&P setup. ACT reaches 80–90% on delicate bimanual tasks from ~10 minutes of demos.

**IL works well for P&P when:**
- The setup is fixed (same robot, same camera, same objects in similar positions)
- You can collect quality demonstrations via kinesthetic teaching or teleoperation
- You want to skip reward design entirely
- The task is dexterous and hard to specify as a reward (e.g., inserting a battery)

**IL struggles with P&P when:**
- Objects or positions vary significantly at test time (distribution shift)
- You can only collect a handful of inconsistent demos
- The task is long-horizon (compounding errors become severe after ~10 steps)
- You need to react to unexpected perturbations mid-task

**Key insight from the literature:** ACT's action chunking (predicting k future actions jointly) is specifically designed to fight the compounding error problem in P&P. Diffusion Policy handles multimodal grasps better (e.g., left-hand vs right-hand approach). For long-horizon P&P sequences, Mamba2Diff's temporal SSM outperforms both.

**Affordance-based P&P with language → [[algo-cliport]]:** CLIPORT reformulates P&P as a 2-step `(T_pick, T_place)` affordance-prediction problem, conditioned on a frozen CLIP model for language/semantics. This sidesteps reward design *and* per-object retraining — "pack the blue pen" vs "pack the red pen" is handled by the same model via language conditioning, achieving >90% on seen language-conditioned tasks with as few as 100 demos. The tradeoff: the SE(2) pick/place primitive doesn't extend to dexterous 6-DOF manipulation the way ACT/Diffusion Policy's continuous action spaces do.

#### VLA for P&P
VLAs add semantic understanding: a fine-tuned SmolVLA or OpenVLA can pick "the object on the left" or "the red container" without any task-specific reward or dedicated demonstration per object. This is the direction the field is moving for real-world deployment.

**VLAs work well for P&P when:**
- The task requires language conditioning ("pick the blue cube and put it in the red bin")
- Objects or scenes vary — generalization is the core requirement
- You have modest fine-tuning data (50–200 demos) but want broad coverage
- Inference speed is not critical (VLAs typically run at 1–6 Hz)

**VLAs struggle with P&P when:**
- Geometric precision is needed (peg-in-hole, tight insertion tolerances)
- High-frequency control is required (>10 Hz) — use async inference stacks
- You have zero demonstrations (VLAs still need some fine-tuning)
- Compute is very limited (though SmolVLA at 450M runs on consumer GPUs)

---

### Practical recommendation for a new P&P project

| Your situation | Start with | Then consider |
|---------------|-----------|---------------|
| Sim available, fixed setup, want strong baseline fast | SAC (task-decomposed reward) | Add CQL for offline stability if deploying to real |
| Fixed setup, 20+ demos available, no sim needed | ACT | Fine-tune with AWAC if you can collect more online data |
| Variable objects, need language instructions | SmolVLA fine-tune | OpenVLA if you want stronger language grounding |
| Long-horizon sequence (5+ steps) | Relay Policy (IL→RL hierarchy) | Or Diffusion Policy + Mamba2Diff |
| Cluttered environment | DQN + non-prehensile moves (push + grasp) | Or HITL-RL for real-world robustness |

---

### Experimental results — SO-100

| Method | Environment | ID result | OOD result | Distractor | Notes |
|--------|-------------|-----------|-----------|-----------|-------|
| SAC (task-decomposed) | Simulation | **92%** (50 eps) | N/A | N/A | Reach 96% · Grasp 92% · Place 92% · 8 drop recoveries |
| ACT (Dataset_v4, 100k steps) | Real hardware | **83%** @ 0° / **92%** @ 45° | 100% OOD @ 45° / 50% @ 0° | **75%** (3/4) | 80 novel positions: 80%. Best overall algorithm on this setup. |
| SmolVLA (Dataset_v4, 20k steps) | Real hardware | **58%** (both orient.) | 50% OOD @ 45° / 25% @ 0° | **0%** (3 near-success) | Many near-successes: precision issue, not comprehension |

**Key finding:** ACT outperforms SmolVLA on all conditions with ~111 episodes of training data. Dataset iteration (v1→v4) was the primary driver of improvement — more impactful than algorithm choice. SmolVLA's distractor failure (0% success despite 75% near-success) is the most counter-intuitive result: fine-tuning on narrow Phase 1 data erodes the pre-trained backbone's generalization.

**PPO vs SAC (same sim environment, 50% success threshold):**
- PPO: 3.17M steps / ~20 min training
- SAC: 1.58M steps / ~5h training  
→ SAC is 2× more sample-efficient but slower wall-clock per step.

---

## P&P Task Taxonomy

| Variant | Key Challenge | Representative Papers |
|---------|--------------|----------------------|
| Tabletop bin-to-bin | Clean perception, reliable grasp | DQN P&P, Self-supervised P&P |
| Cluttered environment | Object separation, non-prehensile moves | Prehensile+Non-Prehensile |
| Multi-robot | Scheduling, coordination | Multi-robot P&P |
| Logistics/mobile | Base mobility + arm coordination | Logistics DRL, Mobile Manipulator |
| Palletizing | Trajectory optimization, heavy loads | Trajectory Palletizing Survey |
| Long-horizon | Multi-stage, subtask chaining | Relay Policy, Task Decomp |
| Bimanual | Dual-arm coordination | ACT/ALOHA, GF-VLA |

---

## Surveys

### Reinforcement Learning for Pick and Place Operations in Robotics: A Survey
**Lobbezoo, Qian, Kwon — University of Waterloo, 2021**

Comprehensive RL survey for P&P: MDP formulation, environment setup, policy optimization (on-policy, off-policy), pose estimation for grasping, and simulation/real-world gaps. Covers [[algo-dqn|DQN]], [[algo-ppo|PPO]], DDPG, and related algorithms. Also discusses IL approaches.

*Tags:* survey, RL algorithms, MDP, pose estimation, sim-to-real

---

### Trajectory Planning for Robotic Manipulators in Automated Palletizing: A Comprehensive Review
**Romero et al. — University of Los Andes, 2025**

Reviews trajectory design and optimization for palletizing robots. Covers placement planning, minimum-time path planning for P&P of heterogeneous boxes onto mixed pallets in automated cells. Synthesizes state-of-the-art algorithms across recent years.

*Tags:* survey, palletizing, trajectory optimization, path planning | See: [[trajectory-planning]]

---

### Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes
**2025**

Covers DRL applications across manipulation, locomotion, and navigation with focus on real-world deployment challenges: sim-to-real transfer, safety, sample efficiency.

*Tags:* survey, real-world DRL, sim-to-real | See: [[reinforcement-learning]]

---

## Industrial & Applied P&P

### Automated Trajectory Planner of Industrial Robot for Pick-and-Place Task → [[algo-stst]]
**2013**

Classic trajectory planning approach for industrial P&P. Uses **STST (S-curve with trapezoidal speed profile)** for smooth, jerk-bounded motion. A **forbidden-sphere** method avoids obstacles. Demonstrated on the **PUMA 560** robot arm. Foundational reference for smooth trajectory generation.

*Algorithm:* [[algo-stst|STST S-curve trajectory]], forbidden-sphere | *Robot:* PUMA 560 | *Tags:* industrial, trajectory planning, jerk-bounded, 2013 | See: [[trajectory-planning]]

---

### Development of a Methodology to Improve Multi-Robot Pick & Place Applications: From Simulation to Experimentation
**2016**

Addresses multi-robot P&P in industrial settings. Studies scheduling strategies and robot coordination in simulation, then validates in physical experiments. Focuses on throughput optimization when multiple robot arms share a workspace.

*Tags:* multi-robot, scheduling, simulation-to-real, 2016

---

### Pick and Place Operations in Logistics Using a Mobile Manipulator Controlled with DRL
**Iriondo et al. — Fundación Tekniker, 2019**

Applies DRL to a mobile manipulator for warehouse/logistics P&P. The robot learns to navigate to objects and pick them up without manual programming. Eliminates hand-coded task scripts via end-to-end learned policy.

*Algorithm:* DRL | *Tags:* logistics, mobile manipulation, warehouse, 2019 | See: [[reinforcement-learning]]

---

### Intelligent Pick-and-Place System Using MobileNet
**2023**

Uses MobileNet (a lightweight CNN) for object recognition in a P&P robotic arm system. Trades off recognition accuracy for inference speed suitable for real-time control.

*Tags:* MobileNet, CNN, object recognition, 2023

---

### Monocular Camera-Based Robotic Pick-and-Place in Fusion Applications
**Yin, Wu, Li et al. — ASIPP, 2023**

End-to-end DRL P&P using only a monocular camera and forward kinematics (no 3D sensor). Designed for nuclear fusion facility environments where 3D sensors may not be viable. Fully data-driven.

*Algorithm:* DRL | *Tags:* monocular camera, fusion, data-driven, 2023

---

## Deep RL for P&P

### Deep Reinforcement Learning Applied to a Robotic Pick-and-Place Application → [[algo-dqn]]
**2021**

Systematic comparison of CNN backbones for DRL-based P&P. MobileNet achieves the best performance at **84% success rate**. Uses DQN + depth camera on a ROS-controlled Cobot.

*Algorithm:* [[algo-dqn|DQN]] + MobileNet CNN | *Tags:* DQN, depth camera, ROS, Cobot, 2021

---

### Learning Pick to Place Objects Using Self-Supervised Learning with Minimal Training Resources → [[algo-dqn]]
**2021**

DQN on a UR5 robot with RGB-D input. Demonstrates that effective P&P policies can be learned with minimal computational resources using self-supervised learning for perception.

*Algorithm:* [[algo-dqn|DQN]] | *Robot:* UR5 | *Tags:* self-supervised, RGB-D, resource-efficient, 2021

---

### Reinforcement Learning for Collaborative Robots Pick-and-Place Applications: A Case Study
**Gomes et al. — Hanze University, 2022**

RL + computer vision for collaborative robots (cobots) performing P&P in human-shared workspaces. Emphasizes safe coexistence with workers.

*Tags:* cobots, human-robot collaboration, computer vision, 2022

---

### Reinforcement Learning for Robot Manipulation Using CQL/SAC → [[algo-cql]], [[algo-sac]]
**Husakovic et al., 2025**

CQL + SAC for P&P in human-robot collaboration (HRC). 100% success in simulation, 80% on real hardware. CQL addresses offline data quality; SAC handles exploration.

*Algorithm:* [[algo-cql|CQL]] + [[algo-sac|SAC]] | *Tags:* HRC, offline RL, transfer to real, 2025 | See: [[reinforcement-learning]]

---

### Prehensile and Non-Prehensile Robotic Pick-and-Place of Objects in Clutter Using DRL → [[algo-dqn]]
**Imtiaz, Qiao, Lee — Technological University of the Shannon, 2023**

Framework for P&P in cluttered environments where objects may need to be pushed aside (non-prehensile) before grasping. MDP with three actions: grasp, push (non-prehensile), and place. Uses DQN + DenseNet-121 + fully convolutional networks.

*Algorithm:* [[algo-dqn|DQN]] + DenseNet-121 | *Tags:* clutter, non-prehensile, MDP, 2023

---

### The Task Decomposition and Reward-System-Based RL for Pick-and-Place → [[algo-sac]]
**Kim, Kwon, Park, Kwon, 2023**

Decomposes P&P into three subtasks (approach object, grasp, reach place position). Each subtask has a dedicated reward function with axis-based weight tuning. SAC achieves 93.2% average success in MuJoCo/Robosuite.

*Algorithm:* [[algo-sac|SAC]], task decomposition | *Tags:* reward shaping, MuJoCo, Robosuite, 93.2% success, 2023

---

### Simulated and Real Robotic Reach, Grasp, and Pick-and-Place Using RL + Traditional Controls → [[algo-ppo]], [[algo-sac]]
**Lobbezoo and Kwon, 2023**

Combines PPO/SAC with traditional robot control on a Franka Panda. Demonstrates sim-to-real transfer for reach, grasp, and P&P. Uses ROS integration.

*Algorithm:* [[algo-ppo|PPO]], [[algo-sac|SAC]] | *Robot:* Franka Panda | *Tags:* sim-to-real, ROS, 2023 | See: [[simulation-and-tools]]

---

### Hybrid Robot Learning for Automatic Robot Motion Planning in Manufacturing
**GE Aerospace, 2025**

Hybrid IL+RL for P&P motion planning in manufacturing. Demonstrates that combining demonstrations with RL fine-tuning yields more robust and deployable industrial solutions.

*Tags:* manufacturing, GE Aerospace, hybrid IL+RL, 2025 | See: [[hybrid-il-rl]]

---

## Long-Horizon and Multi-Stage P&P

### Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning → [[algo-relay-policy]]
**Gupta et al. — Google/Berkeley, 2019**

Goal-conditioned hierarchical policies for multi-stage kitchen manipulation. Bootstrapped from unstructured demonstrations, then improved via RL. Addresses the challenge of long action sequences that exceed demonstration coverage.

*Tags:* long-horizon, hierarchical, kitchen env | See: [[hybrid-il-rl]]

---

### Learning High-Level Robotic Manipulation Actions with Visual Predictive Model
**2024**

Decomposes high-level P&P commands using a visual predictive model + action decomposer. The model plans at a semantic level and translates to primitive P&P actions.

*Tags:* visual predictive, high-level planning | See: [[world-models]]

---

## Related Topics
- [[grasping-and-manipulation]] — detailed grasping methods
- [[reinforcement-learning]] — RL algorithms used for P&P
- [[imitation-learning]] — IL for P&P (ACT, Diffusion Policy)
- [[hybrid-il-rl]] — hybrid methods for P&P
- [[trajectory-planning]] — motion planning for P&P
- [[simulation-and-tools]] — simulators used for P&P training
- [[vision-language-action-models]] — language-conditioned P&P policies
