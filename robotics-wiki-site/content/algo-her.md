# Algorithm: HER — Hindsight Experience Replay

**Paper:** "Hindsight Experience Replay" — Andrychowicz et al. (OpenAI, 2018)
**Category:** Off-policy RL technique for sparse/binary rewards
**Related:** [[reinforcement-learning]], [[algo-ppo]], [[algo-dqn]]

---

## The Problem It Solves

Sparse and binary rewards are the norm in robotics: the robot either succeeds (reward = 1) or fails (reward = 0). With a sparse reward:
- Early in training, the agent almost never reaches the goal by chance
- The vast majority of episodes give reward = 0
- Standard RL algorithms receive no gradient signal and fail to learn

**Example:** Pick-and-place with binary success. If the robot randomly moves its arm, it will almost never accidentally place the object on the goal position. Thousands of episodes pass with zero reward; no learning happens.

**HER goal:** Make every failed episode useful by changing what "goal" we're measuring success against — retroactively, in hindsight.

---

## How It Works

### Core Insight: Relabeling

When an episode fails (the robot didn't reach goal g), HER observes: "But the robot *did* reach some other position g' at the end of the episode." HER relabels the episode as if g' *was* the goal, making the episode a "success" for the goal g'.

This relabeled experience is stored in the replay buffer and used for training alongside the original (failed) experience.

### Step-by-Step

```
Standard RL loop:
1. Start episode with goal g
2. Execute policy, collect trajectory: (s₀, a₀, r₀, s₁), (s₁, a₁, r₁, s₂), ...
3. Compute rewards r_t = [s_{t+1} reaches g ? 1 : 0]
4. Store in replay buffer, sample for Q-learning / actor-critic updates

HER addition (after step 2):
5. For each episode, sample k additional goals g' from the trajectory
   (e.g., g' = achieved position at the last timestep, or random timesteps)
6. Recompute rewards for each transition using g': r_t' = [s_{t+1} reaches g' ? 1 : 0]
7. Store relabeled transitions in the replay buffer too
8. Train on the mix of original + relabeled experiences
```

### Goal Sampling Strategies

HER proposes four strategies for picking the relabeled goals g':
- **final:** Use the state achieved at the last timestep of the episode (simplest, usually best)
- **future:** Sample random future states from the same episode (produces more diverse goals)
- **episode:** Sample random states from the same episode
- **random:** Sample random states from any episode in the buffer (least effective)

`future` strategy: for each transition at timestep t, sample k goals from states {s_{t+1}, ..., s_T} in the same episode. This ensures relabeled goals are *reachable* from the current state.

### Goal Representation

Goals are represented as desired end-effector positions or object positions, encoded as vectors. The policy receives (observation, goal) as input — it is goal-conditioned. The reward function is:

```python
def compute_reward(achieved_goal, desired_goal, threshold=0.05):
    distance = ||achieved_goal - desired_goal||
    return 0.0 if distance < threshold else -1.0
```

(Or binary 0/1 formulation — both work.)

### Base Algorithm

HER is a *wrapper* — it sits on top of any off-policy RL algorithm. The original paper uses **DDPG (Deep Deterministic Policy Gradient)**. HER works equally with DQN, SAC, or TD3.

---

## Why It Works

### Curriculum Through Relabeling

By relabeling goals, HER effectively creates a **curriculum** of easy → hard goals:
- Early in training, the agent achieves goals near where it already is — easy
- As the agent improves, achieved goals become farther and more varied
- The full original goal eventually becomes achievable because the agent has been practicing on nearby subgoals

### Multi-Goal Generalization

HER forces the policy to generalize across goals (it sees thousands of different goals during training). This generalization is what allows the policy to eventually succeed at the original goal — the policy doesn't memorize one trajectory, it learns a general reaching strategy.

### No Reward Engineering

The critical advantage: HER works with raw binary rewards (success/fail). No need to craft shaped reward functions that guide the robot toward the goal. The hindsight relabeling provides the gradient signal that shaped rewards would otherwise provide.

---

## Evaluation

### Environments (from paper)
All environments use sparse binary rewards and are integrated into OpenAI Gym:

1. **Pushing:** Slide a puck to a goal position on a table (Fetch arm, 3D goals)
2. **Sliding:** Flick a puck so it slides to a distant goal (friction/momentum)
3. **Pick-and-Place:** Lift object from table and move to a 3D goal position (includes mid-air goals)

Also tested on simpler toy tasks (bit-flipping, robot arm reaching) to isolate the mechanism.

### Baselines Compared
- **DDPG** (no HER) — fails on all three tasks
- **DDPG + shaped reward** — requires manually engineered reward; HER matches without any engineering
- **Ablations:** different goal sampling strategies (final vs. future vs. random)

### Results
| Task | DDPG (sparse) | DDPG + HER | DDPG + shaped reward |
|------|--------------|-----------|----------------------|
| Pushing | 0% | ~100% | ~100% |
| Sliding | 0% | ~75% | ~80% |
| Pick-and-Place | 0% | ~85% | ~90% |

**Key result:** DDPG alone completely fails on all three tasks with sparse rewards. HER brings it to competitive with shaped-reward methods — without any manual reward design.

---

## Pros

- **Works with binary/sparse rewards** — eliminates the need for careful reward shaping
- **Algorithm-agnostic** — plugs into any off-policy method (DDPG, DQN, SAC)
- **Sample efficient** — each trajectory generates k+1 training samples (1 original + k relabeled)
- **Encourages goal-conditioned generalization** — policy learns to reach any goal, not just one
- **Simple to implement** — just modify the replay buffer sampling

## Cons

- **Only works for goal-conditioned tasks** — requires a well-defined "achieved goal" at each timestep; doesn't apply to all RL tasks
- **Off-policy only** — cannot be applied to on-policy algorithms like PPO (data must be reusable)
- **Threshold sensitivity** — the success threshold ε must be set carefully; too strict = never relabeled successes; too loose = policy learns sloppy goals
- **Doesn't help with non-sparse rewards** — if rewards are already dense, HER adds little
- **Multi-step dependencies** — if the goal requires an exact sequence of actions (not just reaching a state), relabeling is less effective

---

## Connection to Other Methods

HER is foundational for several later works in this wiki:

- **Actionable Models** ([[reinforcement-learning]]): extends HER to offline RL — relabels unlabeled behavioral data without any reward at all
- **Multi-Goal RL** ([[simulation-and-tools]]): the benchmark environments were designed to be used with HER
- **Relay Policy Learning** ([[hybrid-il-rl]]): relabels demonstration data similarly — any state in a trajectory becomes a valid subgoal

---

## In This Wiki

HER is directly referenced in: [[reinforcement-learning]], [[pick-and-place]], [[simulation-and-tools]].

Compare with: [[algo-ppo]] (on-policy; different exploration mechanism), [[algo-cql]] (offline RL; relabeling for offline settings), [[algo-sql-sac]] (max-entropy complement for exploration).
