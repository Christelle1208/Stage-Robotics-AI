# Trajectory Planning and Motion Planning

Trajectory planning determines the time-parameterized path a robot's joints or end-effector follows to move from one configuration to another. Unlike RL-based control (which produces torques or velocities at each timestep), trajectory planning typically generates an explicit smooth path upfront, then executes it with a controller. See also [[pick-and-place]], [[simulation-and-tools]].

---

## Classic Trajectory Planning

### Automated Trajectory Planner of Industrial Robot for Pick-and-Place Task → [[algo-stst]]
**2013**

A foundational paper for industrial P&P trajectory planning.

**STST (S-Curve with Trapezoidal Speed profile):** Generates smooth, jerk-bounded trajectories with S-shaped velocity profiles rather than trapezoidal ones. This avoids velocity discontinuities at the start/end of each motion phase, reducing mechanical wear and vibration — critical for precision industrial applications.

**Forbidden-Sphere obstacle avoidance:** Represents obstacles as spheres in configuration space. The planner avoids trajectories that pass through these forbidden regions.

**Platform:** PUMA 560 (a classic 6-DoF industrial robot arm).

This work predates learning-based approaches but remains relevant as the baseline for smooth motion generation in manufacturing.

*Algorithm:* [[algo-stst|STST S-curve]], forbidden-sphere | *Robot:* PUMA 560 | *Tags:* industrial, jerk-bounded, S-curve, obstacle avoidance, 2013

---

### Trajectory Planning for Robotic Manipulators in Automated Palletizing: A Comprehensive Review
**Romero et al. — University of Los Andes, 2025**

Survey of trajectory planning for **palletizing robots** — a specific P&P application where a robot places boxes onto a pallet in an optimized configuration. Covers:

- **Placement planning:** Determining the 3D layout of heterogeneous boxes on a mixed pallet (a combinatorial optimization problem)
- **Minimum-time path planning:** Given the placement sequence, find the fastest collision-free trajectory between pick positions and place positions
- **Collision-free mixed pallet construction:** Ensuring boxes don't overlap and the pallet is structurally stable

Reviews algorithms proposed over recent years and synthesizes their tradeoffs. Targeted at end-of-line packaging automation.

*Tags:* survey, palletizing, path planning, minimum-time, collision-free, 2025

---

## Neural / Learning-Based Motion Planning

### NAMR-RRT: Neural Adaptive Motion Planning for Mobile Robots in Dynamic Environments → [[algo-namr-rrt]]
**Sun, Xia, Xie, Li, Wang, 2025**

Extends the classic **RRT (Rapidly-exploring Random Tree)** algorithm with neural network-generated heuristic regions:

- Neural network learns where to focus the random tree exploration (away from obstacles, toward the goal)
- **Multi-directional search** with adaptive heuristic regions and sampling rates that update during planning
- Continuously refines the search region as the environment evolves — adapting to moving obstacles

Compared to bi-directional or fixed-heuristic neural RRT variants, NAMR-RRT improves planning efficiency, reduces trajectory length, and achieves higher success rates in dynamic crowded environments (urban areas, shopping malls).

Results validated in both simulation and real-world robot navigation.

*Algorithm:* [[algo-namr-rrt|Neural + RRT]], adaptive heuristic regions | *Tags:* motion planning, dynamic environments, mobile robots, RRT, neural heuristic, 2025

---

## Connection to Pick-and-Place

Trajectory planning is typically the *last mile* in a P&P system:
1. **Perception**: detect and localize the object
2. **Grasp planning**: determine the 6-DoF grasp pose (see [[grasping-and-manipulation]])
3. **Trajectory planning**: generate a smooth collision-free path from current configuration to grasp pose, then to place pose
4. **Execution**: follow the trajectory with a controller

Learning-based methods (RL, IL, VLAs) sometimes subsume trajectory planning inside the policy. Classical trajectory planning remains preferred where smooth motion, predictability, and safety guarantees are required.

---

## Trajectory Planning vs. RL Control

| Approach | Trajectory | Online Adaptation | Guarantees | Best For |
|----------|-----------|-------------------|------------|----------|
| Classical ([[algo-stst\|STST]], RRT) | Explicit, pre-computed | No | Path constraints | Industrial, deterministic envs |
| RL ([[algo-ppo\|PPO]], [[algo-sac\|SAC]]) | Implicit (policy → action) | Yes | None | Dynamic, uncertain envs |
| Neural RRT ([[algo-namr-rrt\|NAMR-RRT]]) | Explicit, guided by NN | Partial | Path constraints | Dynamic environments |
| IL ([[algo-diffusion-policy\|Diffusion Policy]]) | Implicit | No | None | From demonstrations |

---

## Related Topics
- [[pick-and-place]] — trajectory planning applied to P&P tasks
- [[grasping-and-manipulation]] — grasp pose planning precedes trajectory execution
- [[reinforcement-learning]] — RL as an alternative to explicit trajectory planning
- [[simulation-and-tools]] — trajectory algorithms validated in simulation
