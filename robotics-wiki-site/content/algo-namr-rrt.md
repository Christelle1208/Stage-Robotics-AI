# Algorithm: NAMR-RRT — Neural Adaptive Motion Planning

**Paper:** "NAMR-RRT: Neural Adaptive Motion Planning for Mobile Robots in Dynamic Environments" — Sun, Xia, Xie, Li, Wang (2025)
**Category:** Sampling-based motion planning with neural heuristics
**Related:** [[trajectory-planning]], [[simulation-and-tools]]

---

## Background: RRT and Its Limitations

**RRT (Rapidly-exploring Random Tree)** is a classic sampling-based motion planning algorithm:

```
RRT algorithm:
1. Start with tree T = {q_init}
2. Loop:
   a. Sample random configuration q_rand from free space
   b. Find nearest node q_near in T
   c. Extend T from q_near toward q_rand by step size δ → q_new
   d. If q_new is collision-free: add to T
   e. If q_new is near the goal: return path
```

**Strengths:** Probabilistically complete (will eventually find a path if one exists); handles high-dimensional configuration spaces.

**Weaknesses:**
- **Uniform random sampling** explores irrelevant regions. In a cluttered environment, most of the space is *not* on any useful path — uniform sampling wastes computation.
- **Bi-directional / multi-directional extensions** improve efficiency but still lack focused guidance.
- **Dynamic environments** with moving obstacles require replanning; static heuristics become stale.

**NAMR-RRT goal:** Guide the RRT's sampling process with a neural network that learns *where useful paths are likely to exist*, and adapt this guidance in real-time as the environment changes.

---

## How NAMR-RRT Works

### Core Architecture

NAMR-RRT replaces uniform random sampling with **neural network-generated heuristic regions**:

```
Neural heuristic generator: (current state, obstacle map, goal) → heuristic region H
Sampling distribution: bias sampling toward H instead of uniform over free space
```

The algorithm maintains multiple "growth directions" simultaneously (multi-directional), each guided by the neural heuristic.

### 1. Neural Heuristic Region Generation

The neural network is trained to predict the **promising regions of configuration space** — regions where useful tree branches are likely:

**Input:**
- Current robot position and goal position
- Obstacle map (occupancy grid or point cloud representation of environment)
- Current tree T structure (optional, for iterative refinement)

**Output:**
- A probability distribution over configuration space: `P(q | current state, goal, obstacles)`
- High probability = q is likely on or near an optimal path
- This defines the "heuristic region" H

**Architecture:** Typically a convolutional neural network (for grid-based environments) or a PointNet-style network (for point cloud environments). The network predicts whether each configuration point is promising.

**Training:** Supervised on solved planning problems:
```
For each training problem (start, goal, obstacles):
  Run A* or optimal planner → get optimal path
  Label configurations near the optimal path as positive
  Label others as negative
  Train network to predict labels
```

### 2. Adaptive Sampling Strategy

NAMR-RRT's key innovation: the heuristic region and sampling rates **update dynamically during planning**:

**Static neural RRT (prior work):** Sample from a fixed neural heuristic computed once at the start.

**NAMR-RRT (dynamic):**
```
Loop:
1. Compute initial heuristic H₀ from neural network
2. Grow tree T using heuristic-biased sampling
3. Every K tree extensions:
   a. Re-query neural network with updated tree state
   b. Get updated heuristic region H_k
   c. Adjust sampling distribution toward H_k
4. Adapt sampling rate: if tree grows well in H → increase bias; if not → reduce bias and explore more
```

The **adaptive sampling rate** is crucial: if the heuristic region is rich with valid paths, sample heavily from it; if it's been exhausted or is incorrect, fall back to broader exploration. This is controlled by a success rate monitor:

```python
if tree_growth_rate_in_H > threshold:
    increase_heuristic_bias()   # heuristic is good
else:
    decrease_heuristic_bias()   # heuristic is not useful here, explore wider
```

### 3. Multi-Directional Growth

NAMR-RRT grows trees in multiple directions simultaneously:
- A tree from the **start** (standard)
- A tree from the **goal** (backward search)
- Additional "seed" trees from predicted intermediate waypoints

The neural heuristic predicts likely waypoints in complex environments (e.g., doorways, narrow passages). Trees grown from these waypoints can then connect to the start and goal trees, solving long-horizon paths through constrained environments.

### 4. Dynamic Environment Handling

When obstacles move (pedestrians, other robots):
1. Obstacle map is updated in real-time
2. Neural heuristic is re-queried with the new obstacle map
3. Tree branches that pass through newly occupied regions are pruned
4. New branches are grown toward the updated heuristic regions

This allows **online replanning** without starting from scratch — the tree structure from prior planning is partially preserved and updated.

---

## Why It Works

### Focused Exploration

Uniform RRT wastes samples in irrelevant regions (walls, far from goal). The neural heuristic focuses sampling on the "narrow corridor" of configuration space near optimal paths. In complex environments (many obstacles, long corridors), this reduction in wasted samples translates to dramatically faster planning.

### Adaptivity Prevents Heuristic Failure

A fixed neural heuristic (as in prior work) can fail if the heuristic is wrong for a particular scenario (e.g., predicts a path through a region that's actually blocked). The adaptive sampling rate detects when the heuristic isn't productive and reduces reliance on it — preventing the algorithm from getting "stuck" following a bad heuristic.

### Multi-Directionality Handles Long Paths

For long-distance navigation (room-to-room), pure start-to-goal trees require exponentially more samples as distance increases. Growing from multiple directions (including predicted intermediate waypoints) splits the problem into manageable sub-problems.

---

## Evaluation

### Environments

**Simulation:**
- **2D navigation grid:** Rooms with doorways, cluttered corridors, dynamic pedestrians
- **3D robot simulation:** Mobile robot (TurtleBot-style) navigating in Gazebo environments
- Dynamic pedestrians modeled as moving obstacles

**Real-world:**
- Physical mobile robot (wheeled) in a **shopping mall corridor** with moving people
- **Urban outdoor environment** with dynamic traffic and pedestrians
- Evaluated on 100 planning scenarios per environment

### Baselines

| Method | Type |
|--------|------|
| RRT | Classic uniform sampling |
| RRT* | Asymptotically optimal variant |
| Bi-RRT | Bidirectional extension |
| NeurAR (neural RRT, fixed heuristic) | Neural heuristic, non-adaptive |
| Informed RRT* | Ellipsoidal heuristic (not neural) |
| **NAMR-RRT** | Neural adaptive multi-directional |

### Metrics
- **Planning time:** Time to find a valid path (lower is better)
- **Path length:** Length of the found path (lower is better)
- **Success rate:** Fraction of scenarios where a path is found within time limit (higher is better)
- **Replanning time:** Time to replan when an obstacle moves (lower is better)

### Results

**Selected results (2D cluttered environment, 100 scenarios):**

| Method | Planning Time | Path Length | Success Rate |
|--------|--------------|-------------|-------------|
| RRT | 4.2s | 12.8m | 78% |
| Bi-RRT | 2.8s | 11.4m | 84% |
| NeurAR | 1.9s | 10.8m | 88% |
| **NAMR-RRT** | **1.1s** | **9.8m** | **96%** |

**Key improvements over NeurAR (non-adaptive neural):**
- ~40% faster planning time
- ~9% shorter paths
- ~8% higher success rate

**Dynamic environment (moving obstacles):**
- NAMR-RRT replanning: ~0.4s average
- RRT from scratch: ~3.8s average
- NAMR-RRT is ~9× faster at replanning

**Real-world mall deployment:**
- Tested over 1 week with 200 autonomous navigation runs
- NAMR-RRT: 94% success rate, no collisions with pedestrians
- RRT: 71% success rate (frequent replanning failures in dense crowds)

---

## Pros

- **Significantly faster than standard RRT** — focused sampling dramatically reduces planning time
- **Adaptive** — detects when heuristic is wrong and adjusts; more robust than fixed neural heuristics
- **Dynamic environment support** — online replanning without full restart
- **Multi-directional growth** — handles long-distance navigation efficiently
- **Real-world validated** — tested in shopping mall and urban environments, not just simulation

## Cons

- **Neural network requires training** — needs a dataset of solved planning problems; not applicable out-of-the-box to new environments
- **Neural inference cost** — re-querying the network every K steps adds overhead; on slow hardware, may not be faster than simple RRT
- **Heuristic quality caps performance** — if the training distribution doesn't match the deployment environment, heuristics may be consistently wrong (distribution shift)
- **Hyperparameters** — K (re-query frequency), success rate threshold, heuristic bias level all require tuning
- **Mobile robots only** — evaluated on holonomic/differential-drive mobile robots; extension to high-DoF arm trajectory planning is non-trivial

---

## Comparison: NAMR-RRT vs. Other Planning Methods

| Method | Handles Dynamics | Neural Guidance | Adaptive | Multi-directional |
|--------|-----------------|-----------------|---------|-------------------|
| RRT | No (replan) | No | No | No |
| RRT* | No (replan) | No | No | No |
| Bi-RRT | No (replan) | No | No | Yes |
| NeurAR | Limited | Yes (fixed) | No | No |
| **NAMR-RRT** | **Yes (online)** | **Yes (adaptive)** | **Yes** | **Yes** |

---

## In This Wiki

NAMR-RRT: [[trajectory-planning]], [[simulation-and-tools]].
Compare with: [[algo-stst]] (classical industrial trajectory planning for arms), [[reinforcement-learning]] (RL as a full alternative to planning for dynamic environments).
