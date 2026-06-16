# Algorithm: DQN — Deep Q-Network

**Origin:** Mnih et al. (DeepMind, 2013/2015) — "Playing Atari with Deep Reinforcement Learning"
**Category:** Off-policy, model-free, value-based RL (discrete actions)
**Related:** [[reinforcement-learning]], [[algo-ppo]], [[algo-sql-sac]], [[algo-her]]

---

## The Problem It Solves

Classic Q-learning maintains a table Q(s,a) mapping every (state, action) pair to a value. For robot manipulation with image observations, the state space is enormous (millions of pixels) — a lookup table is infeasible.

DQN replaces the table with a **deep neural network** that approximates Q(s, a; θ) for all actions a given state s. This enables Q-learning to scale to high-dimensional observations like images.

---

## How It Works

### Q-Learning Foundation

Q-values represent expected cumulative reward when taking action a in state s and following policy π thereafter:

```
Q*(s, a) = r(s, a) + γ · max_{a'} Q*(s', a')
```

The optimal policy is simply: `π*(s) = argmax_a Q*(s, a)`.

Training: repeatedly update Q toward the Bellman target:

```
Loss = (Q(s, a; θ) - [r + γ · max_{a'} Q(s', a'; θ)])²
```

### DQN's Key Additions

Naively applying Q-learning with neural networks diverges. DQN adds two techniques that make it stable:

**1. Experience Replay**

Instead of updating on (s, a, r, s') immediately, store all transitions in a **replay buffer** and sample random mini-batches for training. This breaks temporal correlations in sequential data (which violate the i.i.d. assumption of SGD) and allows each transition to be reused many times.

```python
replay_buffer = deque(maxlen=100000)

# During interaction:
replay_buffer.append((s, a, r, s', done))

# During training:
batch = random.sample(replay_buffer, batch_size=32)
# Compute loss and update θ
```

**2. Target Network**

The Bellman target `r + γ · max Q(s', a'; θ)` depends on the same network being updated. This creates a moving target problem (updating θ changes the target immediately, causing oscillations).

DQN maintains a **separate target network** with parameters `θ⁻` that is updated slowly:

```
target = r + γ · max_{a'} Q(s', a'; θ⁻)    # uses frozen θ⁻
loss = (Q(s, a; θ) - target)²               # updates θ
# Every C steps: θ⁻ ← θ (hard copy)
```

### Architecture (for image inputs)

```
Input: stacked frames (4 × 84 × 84 grayscale)
→ Conv(32, 8×8, stride=4) → ReLU
→ Conv(64, 4×4, stride=2) → ReLU
→ Conv(64, 3×3, stride=1) → ReLU
→ Flatten → Dense(512) → ReLU
→ Dense(|A|)  [one output per discrete action]
```

For robotics P&P papers, image encoders are swapped for task-specific CNNs (MobileNet, DenseNet-121, etc.)

### Action Selection: ε-Greedy

During training, DQN uses ε-greedy exploration: with probability ε choose a random action; with probability 1-ε choose the argmax action. ε is annealed from 1.0 (fully random) to 0.1 or 0.01 (nearly greedy) over training.

---

## Why It Works

- **Experience replay** solves the i.i.d. assumption: by shuffling past transitions, sequential correlations are broken and the network trains on diverse data
- **Target network** solves the moving target problem: by freezing the target for C steps, the loss has a stable reference point
- **Function approximation** with CNNs generalizes across visually similar states rather than memorizing each one

These three ingredients together (neural Q-function + replay buffer + target network) are the core innovation.

---

## Evaluation

### In the DQN Papers (DeepMind)
- 49 Atari games as benchmark
- Results: superhuman performance on ~20 games; previously no RL method had worked on Atari from raw pixels
- Compared to linear function approximation and prior deep RL methods

### In Robotics Papers (this wiki)

**Deep RL Applied to a Robotic Pick-and-Place Application (2021):**
- **Task:** Pick-and-place with a Cobot arm, depth camera, ROS
- **Input:** RGB + depth images
- **CNN backbone:** Tested multiple (MobileNet, ResNet, custom); MobileNet best
- **Result:** **84% task success rate** with DQN + MobileNet
- **Evaluated on:** Real hardware, 100 P&P trials

**Learning Pick to Place with Self-Supervised Learning (2021):**
- **Task:** P&P with UR5 arm, RGB-D
- **Input:** RGB-D images
- **Goal:** Minimal training resources
- **Result:** Effective P&P from minimal data; no quantitative breakthrough but demonstrates resource efficiency

**Prehensile and Non-Prehensile P&P in Clutter (2023):**
- **Task:** P&P in cluttered environments; must push objects aside
- **CNN backbone:** DenseNet-121 + fully convolutional network
- **Actions:** Grasp, push (non-prehensile), place — all discrete
- **Result:** Successfully learns combined prehensile/non-prehensile strategy

---

## Pros

- **Discrete actions** — natural fit for environments with finite action sets (directions, discrete grasps)
- **Sample reuse** — replay buffer allows each transition to be used many times
- **Works from raw pixels** — CNN encoder handles image inputs directly
- **Well-understood** — extensive body of extensions (Double DQN, Dueling DQN, PER, Rainbow)
- **Simple objective** — one network, one loss function

## Cons

- **Discrete actions only** — cannot directly output continuous joint angles or velocities (requires discretization, which loses precision)
- **Q-value overestimation** — standard DQN overestimates Q-values (the `max` operator is biased upward); fixed by Double DQN
- **Exploration** — ε-greedy is crude; doesn't maintain diverse behavior like SAC's entropy
- **Not sample efficient** vs. SAC for continuous control
- **Large replay buffers** needed for stability; memory-intensive with image observations
- **Slow convergence** in sparse reward settings (needs HER or shaped rewards)

---

## DQN Variants

| Variant | Fix | Used In |
|---------|-----|---------|
| **Double DQN** | Overestimation bias (separate select/evaluate networks) | Robotics papers |
| **Dueling DQN** | Separates state value V(s) and advantage A(s,a) | Game playing |
| **PER** (Prioritized Experience Replay) | Samples important transitions more often | Various |
| **Rainbow** | Combines 6 DQN improvements | Atari SOTA |
| **C51** | Distributional Q-learning (predicts full return distribution) | Various |

---

## In This Wiki

DQN is used in: [[pick-and-place]] (DQN P&P applications), [[grasping-and-manipulation]] (clutter grasping).

For continuous action spaces, prefer: [[algo-sql-sac]] (SAC), [[algo-ppo]] (PPO).
For sparse rewards, combine with: [[algo-her]] (HER).
For offline data, use: [[algo-cql]] (CQL).
