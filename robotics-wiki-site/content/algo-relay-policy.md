# Algorithm: Relay Policy Learning

**Paper:** "Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning" — Gupta, Kumar, Lynch, Levine, Hausman (Google/Berkeley, 2019)
**Category:** Hierarchical imitation learning + RL for long-horizon tasks
**Related:** [[hybrid-il-rl]], [[imitation-learning]], [[reinforcement-learning]]

---

## The Problem It Solves

**Long-horizon manipulation tasks** (e.g., "open the microwave, put something in it, turn on the stove, boil water") require executing dozens of precise, dependent subtasks in sequence. Two classic approaches fail here:

1. **End-to-end RL from scratch:** The reward is so sparse (only at task completion) that random exploration almost never reaches the goal. Even with HER, chaining many subtasks is intractable.
2. **Behavioral cloning on full demos:** Requires demonstrations of *every specific task sequence*. Impractical for large task spaces; doesn't generalize to novel orderings.

**Relay Policy Learning (RPL) goal:** Solve long-horizon tasks using **unstructured, unsegmented demonstrations** of *semantically meaningful behaviors* (not full task sequences), then use RL to chain them together.

---

## How It Works

### Two-Phase Approach

**Phase 1 — Imitation:** Learn a hierarchical policy from unsegmented demonstrations.
**Phase 2 — RL:** Fine-tune the hierarchy to complete the full long-horizon task.

### Phase 1: Goal-Conditioned Hierarchical Policy from Relay Data

**Key insight:** Any trajectory of states `s₀, s₁, ..., sₙ` can be *relabeled* such that any intermediate state `sₖ` is treated as a "subgoal" for the segment `s₀ → sₖ`, and any later state `sⱼ` is treated as the goal for `sₖ → sⱼ`. No segmentation labels needed.

The hierarchy consists of two levels:

**High-level policy h(g_sub | s, g_task):** Given the current state and task goal, proposes the *next subgoal* to pursue.

**Low-level policy l(a | s, g_sub):** Given the current state and the proposed subgoal, executes primitive actions to reach g_sub.

The low-level policy acts for a **fixed window of W timesteps** regardless of whether the subgoal is achieved — this is the "relay" structure. After W steps, the high-level policy proposes a new subgoal.

**Training the hierarchy with relay data relabeling:**

For each demonstration trajectory `(s₀, a₀, s₁, ..., sₙ)`:

*Low-level training:*
```
For each t:
  sample a future subgoal g_sub = s_{t+W} (W steps ahead)
  add (s_t, a_t, g_sub) to low-level dataset
```
The low-level policy learns: "given I'm at s_t and want to reach s_{t+W}, what should I do?"

*High-level training:*
```
For each t in {0, W, 2W, ...}:
  subgoal = s_{t+W}
  task_goal = s_n (end of trajectory)
  add (s_t, subgoal, task_goal) to high-level dataset
```
The high-level policy learns: "when I want to reach s_n and I'm at s_t, a good next subgoal is s_{t+W}."

Both levels are trained with **behavioral cloning** (supervised learning on (observation, action) pairs from the relabeled data).

### Phase 2: RL Fine-Tuning

After Phase 1, the hierarchy already produces reasonable behaviors. RL then improves it:

- **Low-level RL:** The low-level policy is fine-tuned with an intrinsic reward: reaching its assigned subgoal within W steps.
- **High-level RL:** The high-level policy is fine-tuned with the task reward: whether the full task is completed.

The two levels are fine-tuned alternately (while the other level is kept fixed), avoiding the instability of simultaneous hierarchical RL.

### Why Unstructured Demos Are Sufficient

The relay relabeling means:
- The demonstrations only need to *cover* the relevant behaviors, not demonstrate specific task sequences
- e.g., if demonstrations show "opening microwave," "putting in food," "turning on stove" in any order and independently, RPL can learn to chain them
- Demonstrations don't need to be labeled with subtask boundaries

---

## Why It Works

### The Fixed-Window Structure

Fixed window W (rather than "act until goal achieved") gives two benefits:
1. Subgoal relabeling is clean — the high-level policy always commits for exactly W steps
2. Avoids the **termination problem** in hierarchical RL: the low-level doesn't need a termination signal

### Relay Relabeling Creates Dense Learning Signal

Even without task reward, the low-level gets dense supervision from relabeled trajectories (every W-step window is a training example). This is similar to HER but applied to hierarchical policies.

### RL From a Good Starting Point

By the time RL begins, the hierarchy already captures the key behaviors. RL doesn't need to explore from scratch — it only needs to refine timing, coordination, and failure recovery.

---

## Evaluation

### Environment
**FetchKitchen** — a multi-stage kitchen manipulation task in MuJoCo:
- Open microwave, turn burner knob, slide cabinet, move kettle
- 4+ sequential subtasks; sparse terminal reward (all tasks done = success)
- Long horizon: 400+ timesteps per episode

### Demonstrations
- Unstructured: demonstrations of individual kitchen behaviors (not full sequences)
- No segmentation labels required

### Baselines
- **Flat BC:** Behavioral cloning directly on full trajectories
- **Flat RL (SAC/PPO):** RL from scratch on the full task
- **HIRO:** Prior hierarchical RL method (online only)
- **GAIL:** Imitation learning with adversarial objectives

### Results
- **RPL substantially outperforms** all baselines on 4+ stage kitchen tasks
- Flat BC and flat RL both fail: BC doesn't generalize to different orderings; RL can't explore long-horizon tasks
- RL fine-tuning phase is critical: Phase 1 alone achieves ~20-40% success; Phase 2 brings it to ~60-80%
- Key: demonstrates that **unstructured demos + RL** can crack tasks that neither approach alone can

### Ablations
- **No RL phase:** Performance drops significantly (policy can't recover from failures)
- **Fixed vs. variable window W:** Fixed window is simpler and performs comparably to variable
- **Demo source matters:** Structured task-specific demos help more than random data

---

## Pros

- **Works with unstructured demos** — no need to record/label specific task sequences
- **Long-horizon capable** — hierarchical structure breaks temporal credit assignment into manageable chunks
- **Demo-bootstrapped RL** — avoids exploration from scratch; RL fine-tuning converges faster
- **Flexible demo source** — any behaviors covering the relevant subtasks are usable
- **Theoretically motivated** — relay relabeling creates a dense learning signal that standard IL lacks

## Cons

- **Fixed window W is a hyperparameter** — needs to be matched to the timescale of subtasks; too short = high-level replans too often; too long = low-level can't reach distant subgoals
- **Two-phase training complexity** — Phase 1 (IL) then Phase 2 (RL) alternation requires careful orchestration; double the implementation effort of flat approaches
- **Hierarchical instability** — high-level and low-level policies interact; errors at one level compound at the other
- **Demo quality still matters** — if demonstrations don't cover all necessary behaviors, RL must discover them from scratch
- **Long-horizon RL still hard** — Phase 2 RL on long tasks can be slow; requires shaped rewards or HER for intermediate milestones

---

## In This Wiki

Relay Policy Learning: [[hybrid-il-rl]], [[pick-and-place]] (long-horizon P&P tasks).
Compare with: [[algo-awac]] (flat hybrid IL+RL), [[algo-hitl-rl]] (human corrections vs. demo relabeling), [[algo-act]] (flat IL that addresses long-horizon via action chunking).
