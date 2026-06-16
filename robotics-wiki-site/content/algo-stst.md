# Algorithm: STST — S-Curve Trajectory Planning

**Paper:** "Automated Trajectory Planner of Industrial Robot for Pick-and-Place Task" (2013)
**Category:** Classical industrial trajectory planning (kinematics-based)
**Related:** [[trajectory-planning]], [[pick-and-place]]

---

## The Problem It Solves

Industrial robots performing P&P tasks need to move their end-effector smoothly between pick and place positions. Two naive approaches fail:

1. **Point-to-point trapezoidal velocity:** Velocity jumps from 0 → max → 0 instantaneously at segment boundaries → **jerky motion** → mechanical vibration, reduced precision, increased wear.
2. **Unconstrained polynomial interpolation:** Smooth mathematically but doesn't respect joint limits, workspace boundaries, or obstacle-free constraints.

**STST goal:** Generate smooth, jerk-bounded, collision-free trajectories for industrial P&P, with guaranteed maximum velocity and acceleration.

---

## How It Works

### 1. Path Planning: Forbidden-Sphere Obstacle Avoidance

Before generating the time trajectory, a geometric **path** (sequence of waypoints) must be found that avoids obstacles.

**Forbidden-sphere method:**
- Model each obstacle (machine, support structure, neighboring robot) as a **sphere** in Cartesian space
- The sphere is defined by its center (obstacle centroid) and radius (obstacle bounding radius + safety margin)
- The path is constrained to stay outside all spheres

**Path construction:**
```
Given waypoints p₀ (pick) → p₁ (lift) → p₂ (transport) → p₃ (descend) → p₄ (place):
For each segment pᵢ → pᵢ₊₁:
  Check if straight line intersects any forbidden sphere
  If yes: deflect the path around the sphere by computing the tangent point
  Connect deflected waypoints with smooth Bezier curves
```

The result is a geometric path in Cartesian space guaranteed to avoid all modeled obstacles.

### 2. Kinematic Inversion: Cartesian Path → Joint Space

The Cartesian path is converted to joint angles via **inverse kinematics (IK)**:
```
for each waypoint p in Cartesian path:
    q = IK(p)    # compute joint configuration
```

For the PUMA 560 (6-DoF), a closed-form analytic IK solution exists. The method selects among multiple IK solutions by choosing the one closest to the previous joint configuration (to avoid sudden joint flips).

### 3. Time Parameterization: STST S-Curve Velocity Profile

Given the joint-space waypoints `q₀, q₁, ..., qₙ`, STST computes a **time parameterization** that determines how fast to traverse each segment.

**Trapezoidal (traditional) profile:** Piecewise linear velocity — instantaneous acceleration changes → infinite jerk → vibration.

**S-curve (STST) profile:** Seven-phase motion with continuous jerk:

```
Phase 1: Acceleration increases from 0 → a_max  (jerk = j_max)
Phase 2: Constant acceleration a_max             (jerk = 0)
Phase 3: Acceleration decreases a_max → 0        (jerk = -j_max)
Phase 4: Constant velocity v_max                 (jerk = 0)
Phase 5: Deceleration increases 0 → -a_max      (jerk = -j_max)
Phase 6: Constant deceleration -a_max            (jerk = 0)
Phase 7: Deceleration decreases -a_max → 0       (jerk = j_max)
```

The name **STST (Start-T-S-Top-S-T-Stop)** reflects these 7 phases where T=trapezoidal segment and S=S-curve transition.

**Mathematical formulation for Phase 1 (0 ≤ t ≤ t₁):**
```
jerk:         j(t) = j_max                     (constant)
acceleration: a(t) = j_max · t
velocity:     v(t) = j_max · t² / 2
position:     q(t) = j_max · t³ / 6
```

Each subsequent phase is computed by integrating from the end of the previous phase.

**Constraints respected:**
- `|v(t)| ≤ v_max` — maximum joint velocity
- `|a(t)| ≤ a_max` — maximum joint acceleration
- `|j(t)| ≤ j_max` — maximum jerk (new constraint vs. trapezoidal)
- Boundary conditions: v(0) = v(T) = 0 (start and stop)

**Minimum time computation:**
The STST profile is parameterized to find the *minimum-time* trajectory that satisfies all constraints. The algorithm solves for t₁, t₂, ..., t₇ (phase durations) given the segment distance and constraints.

### 4. Synchronization Across Joints

For multi-DoF robots, each joint needs to move a different distance. To keep the end-effector on the planned Cartesian path, all joints must start and stop simultaneously:

- Compute STST profiles independently for each joint
- Find the joint requiring the most time (the **bottleneck joint**)
- Scale all other joints' profiles to match the bottleneck time while staying within their constraints

---

## Why It Works

### Jerk Boundedness = Smoothness

The S-curve guarantees `j(t) ≤ j_max` everywhere. This has real physical benefits:
- **Vibration reduction:** Infinite jerk (trapezoidal profile) excites structural resonances in the robot arm. Bounded jerk keeps frequency content low.
- **Reduced wear:** Smooth acceleration changes reduce stress on joints, gearboxes, and motors.
- **Precision:** Vibration causes the end-effector to oscillate around the target position. Less vibration → settles faster → shorter cycle time.

### Forbidden-Sphere = Simple But Effective Collision Avoidance

For structured industrial environments (fixed machines, known obstacles), sphere approximations are conservative but sufficient. The planning is deterministic and runs in milliseconds — critical for online P&P planning.

---

## Evaluation

### Robot
**PUMA 560** — a classic 6-DoF industrial robot arm, widely used as a benchmark platform in robotics research.

### Task
Automated P&P in a simulated industrial cell with several obstacles (support structures, neighboring machines modeled as spheres).

### Comparison
Compared to:
- **Linear interpolation (no trajectory planning):** Joint angles interpolated linearly → jerky, can't handle velocity/acceleration limits
- **Trapezoidal profiles:** Standard in industrial practice; no jerk limiting

### Results

| Metric | Trapezoidal | STST S-Curve |
|--------|------------|--------------|
| Max jerk | Unbounded | j_max (bounded) |
| End-effector vibration | High | Low |
| Cycle time | Baseline | +5-15% (longer, to respect jerk limit) |
| Path accuracy | ±2-5mm | ±0.5-1mm |
| Obstacle avoidance | Not guaranteed | Guaranteed (spheres) |

**Key tradeoff:** STST trajectories take slightly longer (5-15% more time) than trapezoidal profiles to execute, but the improved precision and reduced vibration outweigh the time cost in precision P&P applications.

---

## Pros

- **Smooth motion** — bounded jerk eliminates vibration; essential for precise P&P
- **Deterministic** — given constraints, trajectory is computed analytically; no sampling uncertainty
- **Fast computation** — milliseconds to compute; suitable for online replanning
- **Guarantees constraint satisfaction** — velocity, acceleration, and jerk limits are guaranteed
- **Obstacle-free path** — forbidden-sphere method provides collision avoidance
- **Proven in industry** — variants of S-curve profiles are standard in industrial robot controllers (Siemens, FANUC, ABB)

## Cons

- **Assumes known, static obstacles** — forbidden-sphere method requires a pre-built map of fixed obstacles; can't handle dynamic objects
- **Conservative obstacle representation** — spheres over-approximate real obstacle shapes; may reject valid paths that are geometrically free
- **No closed-loop adaptation** — trajectory is computed once and executed open-loop; if the object has moved or there's an error, no correction occurs
- **Not for learning environments** — purely geometric; doesn't improve with experience; requires manual modeling of the environment
- **Joint space interpolation issues** — straight-line paths in joint space are not straight lines in Cartesian space; the end-effector follows a curved path (may be undesirable in tight spaces)

---

## Comparison: STST vs. RL-Based Control

| Property | STST | RL (PPO/SAC) |
|----------|------|--------------|
| Trajectory type | Explicit, pre-planned | Implicit (policy → action) |
| Smoothness | Guaranteed (jerk-bounded) | Not guaranteed |
| Obstacle handling | Spheres (known, static) | Learned (dynamic) |
| Adaptability | None | Online (with sufficient training) |
| Computation time | Milliseconds | Milliseconds (policy inference) |
| Training data | None (analytical) | Many episodes |
| Optimality guarantee | Minimum-time within constraints | No guarantee |
| Best for | Industrial cells, fixed setups | Dynamic, uncertain environments |

---

## In This Wiki

STST: [[trajectory-planning]], [[pick-and-place]].
Compare with: [[algo-namr-rrt]] (neural motion planning for dynamic environments), [[reinforcement-learning]] (RL as an alternative for adaptive control), [[algo-ppo]] / [[algo-sql-sac]] (RL algorithms that implicitly learn motion control).
