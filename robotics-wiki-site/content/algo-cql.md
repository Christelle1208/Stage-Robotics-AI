# Algorithm: CQL — Conservative Q-Learning

**Paper:** "Conservative Q-Learning for Offline Reinforcement Learning" — Kumar et al. (NeurIPS 2020)
**Category:** Offline RL (model-free, value-based)
**Related:** [[reinforcement-learning]], [[algo-sql-sac]], [[algo-dqn]], [[algo-awac]]

---

## The Problem It Solves

**Offline RL** (also called batch RL): learn a policy from a *fixed dataset* of previously collected transitions, without any online interaction with the environment.

Why is this hard? Standard Q-learning applied to offline data severely overestimates Q-values for **out-of-distribution (OOD) actions** — actions the data collection policy never took. Because the dataset never shows what happens when these OOD actions are taken, the Q-function has no signal to accurately evaluate them. Worse, the learned policy tends to *exploit* these inflated estimates, selecting OOD actions at test time and performing poorly.

**CQL goal:** Train a Q-function that is *conservative* — it systematically underestimates Q-values for OOD actions, making the learned policy stick to actions that appear in the dataset.

---

## How It Works

### The Conservative Objective

Standard Q-learning loss (from the dataset D):

```
L_standard(Q) = E_{(s,a,r,s')~D} [(Q(s,a) - (r + γ·max Q(s',a')))²]
```

CQL adds a **regularization term** that penalizes high Q-values for actions not in the dataset:

```
L_CQL(Q) = α · E_{s~D} [log Σ_a exp(Q(s,a)) - E_{a~D}[Q(s,a)]]
          + standard Bellman error
```

Breaking this down:
- `log Σ_a exp(Q(s,a))` = **log-sum-exp over all actions**: soft-maximum of Q-values. Maximizing this over the network parameters means the network tries to increase Q-values. *CQL minimizes this*, pushing all Q-values down.
- `E_{a~D}[Q(s,a)]` = **average Q-value for dataset actions**. CQL maximizes this, pushing dataset Q-values up.
- **Net effect:** Q-values for dataset actions are pushed up; Q-values for all other actions are pushed down. OOD actions get low Q-values; in-distribution actions get accurate ones.

### The Conservative Lower Bound

Theoretically, CQL produces a *lower bound* on the true Q-value of the behavioral policy (the policy that collected the data). This means:
- The learned policy will only prefer an action over the behavioral policy if it's *genuinely* better (not just due to overestimation)
- The gap between CQL estimates and true values is bounded and controlled by α

### CQL + SAC (in practice)

In the papers (HRC robot manipulation, 2025), CQL is combined with SAC:

```
Total loss = SAC policy + SAC critic + CQL regularization term
```

SAC provides the entropy-maximizing actor-critic framework; CQL adds conservatism to the critic. The combined algorithm can also fine-tune offline-learned policies with limited online data.

---

## Why It Works

### Distribution Shift is the Core Problem

When you train on dataset D and then act in the environment, the states you visit are different from those in D (because your learned policy is different from the data-collection policy). This distribution shift causes Q-value errors to compound.

CQL counteracts this by pessimistically evaluating OOD actions. Even when the policy generalizes to new states, it avoids extrapolating Q-values into unseen regions — it stays near the behavioral support.

### Comparison to Standard Offline RL

| Method | OOD Handling | How |
|--------|-------------|-----|
| Behavioral Cloning | Avoids OOD (just copies) | No Q-learning at all |
| Standard Q-learning | None — exploits OOD Q-values | Overestimates badly |
| CQL | Penalizes OOD Q-values | Regularization |
| IQL (Implicit Q-Learning) | Avoids OOD via in-sample max | Different technique |

---

## Evaluation

### Original CQL Paper
**Benchmarks:** D4RL offline RL benchmark (offline datasets from: MuJoCo locomotion, Adroit hand manipulation, Kitchen manipulation, AntMaze navigation)

**Datasets:** Multiple quality levels:
- `random`: data from a random policy
- `medium`: data from a partially-trained policy
- `expert`: data from an expert policy
- `medium-expert`: mix
- `medium-replay`: replay buffer from medium policy training

**Results (selected, normalized scores):**

| Task | BC | BCQ | BEAR | CQL |
|------|----|-----|------|-----|
| Hopper-medium | 52.5 | 54.5 | 52.1 | **58.5** |
| HalfCheetah-medium | 42.6 | 42.2 | 40.1 | **44.0** |
| Kitchen-mixed | 47.5 | 38.2 | 47.2 | **51.0** |
| AntMaze-medium | 0 | 0 | 0 | **61.0** |

CQL particularly shines on **AntMaze** (long-horizon navigation requiring stitching together trajectory segments) — where simpler methods fail completely.

### In Robotics Papers (this wiki)

**Reinforcement Learning for Robot Manipulation Using CQL/SAC (2025):**
- **Task:** Pick-and-place in human-robot collaboration (HRC) scenarios
- **Setup:** CQL for offline pre-training from suboptimal data; SAC for online fine-tuning
- **Evaluation:** 
  - 100 simulation trials + 20 real hardware trials
  - Real Cobot arm; varied object positions and sizes
- **Results:** **100% success in simulation; 80% success on real hardware**
- **Key finding:** CQL prevents Q-value overestimation in scenarios with limited, suboptimal HRC data — important because safe data collection in HRC means the data is sparse and suboptimal

---

## Pros

- **Works from fixed offline data** — no environment interaction needed during training
- **Strong theoretical guarantees** — provably lower bounds on policy value under the data distribution
- **Handles suboptimal data** — unlike behavioral cloning, can outperform the data-collection policy by combining good behaviors from different parts of the dataset
- **Compatible with SAC** — plugs in as an additional loss term; simple to combine
- **Scales to complex tasks** — works with image inputs, high-dimensional state spaces

## Cons

- **Hyperparameter α** — the conservatism coefficient must be tuned; too high = overly conservative (policy barely moves from behavioral policy); too low = doesn't fix overestimation
- **Dataset coverage required** — if the dataset doesn't cover goal-relevant states at all, CQL can't bridge the gap (unlike online RL which can explore)
- **Computational cost** — the log-sum-exp over all actions requires sampling many actions per state; expensive for large action spaces
- **May be *too* conservative** — on tasks where the dataset contains bad behavior mixed with good, CQL may not improve sufficiently over behavioral cloning
- **Doesn't actively explore** — all information must come from the dataset; misses novel strategies the data never showed

---

## Connection to Other Methods

- **CQL → IQL:** IQL (Implicit Q-Learning) is a later offline RL method that avoids OOD actions differently — by replacing the `max` in Bellman targets with an expectile regression that stays in-distribution. Less conservative, often more practical.
- **CQL + SAC = bridge to online:** Pre-train offline with CQL, then fine-tune online with SAC. The CQL initialization prevents early catastrophic online failure.
- **Actionable Models** (this wiki): a goal-conditioned offline RL method with different conservatism mechanism (hindsight relabeling rather than Q-penalization).

---

## In This Wiki

CQL is used in: [[reinforcement-learning]] (CQL/SAC paper), [[pick-and-place]] (HRC P&P applications).

See also: [[algo-sql-sac]] (SAC as the online companion), [[algo-dqn]] (the offline overestimation problem CQL fixes), [[algo-awac]] (another hybrid offline→online method).
