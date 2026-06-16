# Evaluation Protocol — Robot Learning Policies

Evaluating a robot learning policy is not just about measuring success rate on the training distribution. A policy that achieves 95% in-distribution success may completely fail when an object is 5cm further away, or when a lamp is turned on. This page defines a standardized evaluation protocol that measures performance across multiple axes to give an honest picture of what a policy *actually* does.

This protocol is designed to be algorithm-agnostic — it applies equally to SAC, ACT, SmolVLA, and any other approach, enabling fair cross-algorithm comparison.

→ See [[decision-guide]] for training guidance. See [[overview]] for the three paradigm families.

---

## Why a standardized protocol matters

Without a protocol, results are incomparable:
- "90% success" means nothing if it was measured on 10 trials in ideal conditions
- A policy that achieves 85% with fixed objects but 20% with varied positions is not as good as one that achieves 75% on both
- Generalization to new objects is the key advantage of VLAs — but it only shows up if you test for it

The goal is not just a single number, but a **profile** across multiple axes.

---

## Evaluation axes

### Axis 1 — In-Distribution (ID)
> *Does the policy work on the exact conditions it was trained/tuned for?*

This is the baseline. If the policy can't pass this, nothing else matters.

**Conditions:** exact training distribution — same objects, same positions (sampled from training range), same lighting, same camera setup.

**What to measure:** success rate, execution time, trajectory smoothness (optional).

**Number of trials:** minimum 50. Use 100+ for publication-quality results. Stratify across the training position range (don't just test the center).

---

### Axis 2 — Near Out-of-Distribution (near-OOD)
> *Does the policy generalize to small, realistic variations it wasn't trained on?*

These are the variations that will inevitably appear in real deployment even if you try to replicate training conditions exactly.

**Conditions to test:**

| Variation | How to test | Expected impact |
|-----------|------------|----------------|
| Object position ± 3–5cm beyond training range | Move object to edge/corner positions outside training distribution | Moderate drop for IL; large drop for RL |
| Object orientation ± 30° beyond training range | Rotate object to angles not covered in training | Large drop for ACT if not in demos |
| Slight lighting change (+/- 30% ambient) | Change room lighting or time of day | Moderate for IL; low for VLA (pretrained backbone) |
| One distractor object added | Place one irrelevant object in the workspace | Large drop for IL without distractor demos |
| Camera position shift (1–2cm) | Move camera slightly from trained position | Small-moderate; depends on visual encoding |

**Number of trials:** 30 per condition (5–6 conditions = 150–180 trials total).

---

### Axis 3 — Far Out-of-Distribution (far-OOD)
> *What does the policy do when it encounters conditions very different from training?*

This tests the limits of generalization. Expected to be lower; the goal is to understand *how* the policy fails, not just whether it does.

**Conditions to test:**

| Variation | How to test | What failure reveals |
|-----------|------------|---------------------|
| New object color/texture | Replace trained object with same shape, different color | Visual generalization. RL: fails. IL: depends on augmentation. VLA: often robust |
| New object shape (same category) | Replace cube with cylinder or irregular shape | Shape generalization. All methods struggle |
| Multiple distractors (3–5) | Add several irrelevant objects of varied types | Attentional robustness |
| Strong lighting change | Test under spotlight/shadow vs training lighting | Visual robustness under extreme conditions |
| Language instruction paraphrase (VLA only) | "Pick up the X" → "Grab the X" / "Take the X" | Instruction-level generalization |
| New object category (VLA only) | Replace with semantically related but untrained object | Cross-category generalization via pretrained knowledge |

**Number of trials:** 20 per condition. Focus on *characterizing failure modes*, not just success rate.

---

### Axis 4 — Robustness to perturbations
> *Does the policy recover when something unexpected happens mid-execution?*

Real deployments involve unexpected events. A policy that achieves 95% in clean conditions but cannot recover from a light perturbation is less deployable than a 80% policy that recovers gracefully.

**Perturbation types to test:**

| Perturbation | Method | Recovery criterion |
|-------------|--------|-------------------|
| **Object slip** (grasp disturbed mid-air) | Gently push object during grasp phase | Does the robot re-attempt grasp? |
| **Object displacement** (repositioned mid-task) | Move object 5cm while robot is approaching | Does the robot track the new position? |
| **Temporary occlusion** | Block camera view for 1 second | Does the policy resume on vision recovery? |
| **Human intervention** | Briefly hold the robot arm during execution | Does it resume smoothly? |
| **Missed grasp** (object not picked up) | Let gripper close on empty air | Does the robot detect failure and retry? |

**Protocol:**
1. Run 20 trials per perturbation type
2. For each trial, record: (a) did the robot detect the perturbation?, (b) did it recover?, (c) what was the final task success?
3. Report: perturbation recovery rate separately from task success rate

**What to expect by family:**
- **RL**: typically low recovery — the policy was trained to succeed, not to detect/recover from failures
- **IL (ACT/Diffusion Policy)**: low recovery — BC has no feedback mechanism beyond replanning from new observations
- **VLA (with thinking/agentic features)**: best recovery — models like GR 1.5 use language traces to detect subtask failure and adapt

---

### Axis 5 — Consistency and reliability
> *Is the policy consistently good, or does it have high variance?*

A policy with 80% mean success but ±30% standard deviation across runs is less reliable than a 75% policy with ±5% deviation.

**What to measure:**
- **Success rate per object position**: break down success by spatial region (near/far, left/right of workspace). Reveals spatial blind spots.
- **Success rate vs episode index**: does performance drop across a session (fatigue effects, actuator heating, visual drift)?
- **Failure mode classification**: categorize failures into: (a) grasping failure, (b) placement failure, (c) approach failure, (d) perception failure, (e) timeout. The distribution tells you where to focus improvement.

---

## Standard metrics

| Metric | Definition | When to use |
|--------|-----------|-------------|
| **Success rate** | Fraction of trials where task completes fully | Primary metric for all evaluations |
| **Progress score** | Fraction of task stages completed (0–1) | Better than binary for long-horizon tasks |
| **Recovery rate** | Fraction of perturbation trials where policy recovers | Axis 4 specifically |
| **Execution time** | Mean time to task completion in successful trials | Efficiency comparison |
| **Trajectory smoothness** | Mean jerk across execution | Optional; proxy for control quality |
| **Failure mode breakdown** | % of failures attributable to each failure type | Diagnostic metric for improvement |

---

## Protocol for the SO-100 pick-and-place experiments

This section operationalizes the axes above for the specific experimental setup.

### Test configuration

| Parameter | Value |
|-----------|-------|
| Robot | SO-100 (6-DoF) |
| Task | Pick-and-place: grasp object from bin A, place in bin B |
| Objects | *[To be specified: object types, sizes, colors]* |
| Camera setup | *[To be specified: top camera only / top + wrist?]* |
| Episode timeout | *[To be specified: e.g., 60 seconds]* |
| Success criterion | Object placed within 5cm of target position |

### Test matrix (minimum required)

| Axis | Conditions | Trials per condition | Total trials |
|------|-----------|---------------------|-------------|
| ID | Training distribution | 50 | 50 |
| Near-OOD: position | Object at ±5cm from training boundary | 30 | 30 |
| Near-OOD: orientation | Object rotated ±45° | 30 | 30 |
| Near-OOD: lighting | Ambient light ±30% | 30 | 30 |
| Near-OOD: distractor | 1 irrelevant object added | 30 | 30 |
| Far-OOD: new color | Same object, different color | 20 | 20 |
| Perturbation: slip | Object pushed during grasp | 20 | 20 |
| **Total** | | | **~210 trials** |

### Per-algorithm comparison table (Dataset_v4 results)

| Algorithm | ID (0°) | ID (45°) | OOD positions (0°) | OOD positions (45°) | Novel random pos. | Distractor |
|-----------|---------|---------|-------------------|--------------------|--------------------|-----------|
| SAC (sim) | 92% | N/A | N/A | N/A | N/A | N/A |
| **ACT** | **83%** (10/12) | **92%** (11/12) | 50% (2/4) | **100%** (4/4) | **80%** (4/5) | **75%** (3/4) |
| SmolVLA | 58% (7/12) | 58% (7/12) | 25% (1/4) | 50% (2/4) | 60% (3/5) | 0% (0/4)* |

*SmolVLA: 3 near-successes with distractor (cube above box but dropped on edge). Training data: ~111 episodes (Dataset_v4). Lighting and color generalization not yet tested — Phase 2 of dataset pending.

---

## Practical tips for running evaluations

**Randomize trial order** — don't run all SAC trials then all ACT trials. Interleave them so any session-level effects (lighting drift, robot wear) are distributed across algorithms.

**Reset completely between trials** — return robot to home position, reset object to initial position, close gripper fully. Partial resets are a common source of evaluation bias.

**Record every trial** — always record video. Failures often look surprising on video and reveal issues you'd miss from success rate alone.

**Report confidence intervals** — with N=50 trials at 80% success rate, the 95% CI is ±11 pp. With N=100 it's ±8 pp. Don't report results from fewer than 30 trials as meaningful.

**Separate training evaluator from training designer** — if possible, have someone who didn't tune the algorithm run the evaluation. Unconscious bias in trial setup (e.g., placing objects in easier positions) is a real confound.

---

## Related pages
- [[decision-guide]] — how to choose and train each approach
- [[pick-and-place]] — task-specific synthesis including SO-100 results
- [[reinforcement-learning]], [[imitation-learning]], [[vision-language-action-models]] — per-paradigm details
