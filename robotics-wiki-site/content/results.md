# Results — SO-100 Pick-and-Place Experiments

This page is the **single dashboard for all experimental results** produced during this project. It consolidates everything: simulation RL results, real-hardware IL results, real-hardware VLA results, and the dataset iteration that connects them.

→ For *methodology* (how trials are structured, what each axis measures): [[evaluation-protocol]]
→ For *full narrative* (what worked, what didn't, surprises, revised recommendations): [[decision-guide]] → Step 7 (REX)
→ For *task-level synthesis* (which family to pick for P&P and why): [[pick-and-place]]
→ For *algorithm-specific detail*: [[algo-sac]], [[algo-act]], [[algo-smolvla]]

---

## Headline summary

| Algorithm | Environment | Overall result | Status |
|-----------|-------------|-----------------|--------|
| **SAC** (3-subtask + recovery) | Simulation (MuJoCo) | **92%** (46/50 episodes) | ✅ Done |
| **ACT** (Dataset_v4, 100k steps) | Real hardware (SO-100) | **83%** ID @ 0° / **92%** ID @ 45° | ✅ Done |
| **SmolVLA** (Dataset_v4, 20k steps) | Real hardware (SO-100) | **58%** ID (both orientations) | ✅ Done |

**Meta-finding:** dataset iteration (v1 → v4) was the single biggest driver of performance — bigger than any algorithm or hyperparameter choice. ACT outperforms SmolVLA at this data volume (~111 episodes); the gap is expected to narrow or reverse with more diverse fine-tuning data (see [[decision-guide]] for the reasoning).

---

## 1. SAC — Simulation (MuJoCo)

**Setup:** 3-subtask decomposition (Reach → scripted Grasp → Place), object position randomized ±7cm, fixed orientation, drop-recovery state machine (relaunches Reach if cube falls below z=5cm during Place).

| Subtask | Success rate |
|---------|-------------|
| Reach | 96% (48/50) |
| Grasp | 92% (46/50) |
| Place | 92% (46/50) |
| **Overall** | **92% (46/50)** |
| Drop recoveries | 8 / 50 episodes (all recovered successfully) |

### PPO vs SAC (same environment, 50% success threshold)

| Algorithm | Steps to 50% success | Wall-clock training time |
|-----------|----------------------|---------------------------|
| PPO | 3.17M steps | ~20 minutes |
| SAC | 1.58M steps | ~5 hours |

**Reading:** SAC is **2× more sample-efficient** (half the steps) but **much slower wall-clock** per step — PPO's faster simulation throughput per step wins on raw training time despite needing more steps.

→ Full discussion: [[decision-guide]] REX → SAC, [[reinforcement-learning]] "When to use RL" synthesis.

---

## 2. Dataset iteration — v1 → v4 (shared by ACT and SmolVLA)

Both ACT and SmolVLA were trained on the same evolving dataset. This iteration is the backbone of the meta-finding above.

| Version | Episodes | What changed | Outcome |
|---------|----------|--------------|---------|
| **v1** | 50 (mono camera) | Initial collection | ❌ Failed — non-reproducible environment between training and eval |
| **v2** | 80 (Phase 1) | Reproducible setup, structured by zone/orientation | Both policies approach the cube (~5–10cm) but stop short — low visual contrast hypothesis |
| **v3** | 80 + colored post-it markers | Added contrast markers on robot + box | Approach improves to ~5–7cm; OOD positions still fail (goes to nearest known position) |
| **v4** | v3 + 15 recovery + 16 random-orientation eps (~111 total) | Added failure-recovery episodes and orientation diversity | Recovery and orientation gaps both close — final results below |

→ Full per-version diagnosis: [[decision-guide]] REX → ACT / SmolVLA sections.

---

## 3. ACT — Real hardware (Dataset_v4, 100k steps, 1 GPU)

| Condition | Result |
|-----------|--------|
| In-distribution @ 0° | **83%** (10/12) |
| In-distribution @ 45° | **92%** (11/12) |
| OOD position @ 0° | 50% (2/4) |
| OOD position @ 45° | **100%** (4/4) |
| Novel random positions | 80% (4/5) |
| With distractor | **75%** (3/4) |

**Key findings:**
- Recovery episodes (added in v4) were the single biggest jump — fixed ACT's prior inability to recover from a failed grasp.
- 45° orientation generalizes *better* than 0° (100% OOD vs 50% OOD) — the random-orientation training episodes produced unexpectedly transferable features.
- Distractor robustness is strong (75%) — ACT is not confused by a second object in the scene.

→ Full discussion: [[algo-act]] "Results in This Project", [[decision-guide]] REX → ACT.

---

## 4. SmolVLA — Real hardware (Dataset_v4, 20k steps, 4 GPUs, full fine-tuning)

| Condition | Result |
|-----------|--------|
| In-distribution @ 0° | 58% (7/12) |
| In-distribution @ 45° | 58% (7/12) |
| OOD position @ 0° | 25% (1/4) |
| OOD position @ 45° | 50% (2/4) |
| Novel random positions | 60% (3/5, +1 near-success) |
| With distractor | **0%** (0/4, +3 near-successes) |

**Key findings:**
- Consistent across orientations (58%/58%) — plausibly a pretraining benefit, since ACT needed explicit orientation diversity to reach similar consistency.
- **Near-success rate is the more informative signal** — many "failures" were the cube placed above the bin but dropped on the edge. The model understands the task; precision is the gap.
- **0% with distractor despite 75% near-success** is the most counter-intuitive result of the project. Leading hypothesis: full fine-tuning (no frozen layers) eroded the pretrained backbone's robustness to novel scene elements. See [[algo-smolvla]] "Fine-Tuning SmolVLA — The Options" for the full reasoning and a proposed follow-up experiment (frozen-backbone fine-tuning + distractor demos).

→ Full discussion: [[algo-smolvla]] "Results in This Project", [[decision-guide]] REX → SmolVLA.

---

## 5. Cross-algorithm comparison

| Algorithm | Sim result | Real ID | Real OOD | Distractor | Fine-tune data | Key finding |
|-----------|-----------|---------|----------|-----------|------|------------|
| **SAC** | **92%** (50 eps) | N/A (sim only) | N/A | N/A | 0 demos (reward-based) | 3-subtask decomposition + recovery state machine = 92% |
| **ACT** | — | **83%** (0°) / **92%** (45°) | **100%** OOD @ 45° | **75%** (3/4) | ~111 demos | Data iteration > algorithm choice; best overall on this setup |
| **SmolVLA** | — | **58%** (both orientations) | 50% OOD @ 45° | **0%** (3 near-successes) | ~111 demos | Fine-tuning strategy (full vs frozen) may be the cause of the distractor gap |

---

## What's not yet measured

Per [[evaluation-protocol]], the full SO-100 test matrix (~210 trials across ID, near-OOD, far-OOD, and perturbation axes) has **not yet been completed**. Specifically still pending:

- Lighting variation (±30% ambient) — Axis 2, not yet tested for any algorithm
- Far-OOD: new object color/texture, new object shape — Axis 3
- Perturbation/recovery trials (object slip, mid-task displacement, occlusion) — Axis 4, beyond the drop-recovery already built into the SAC reward
- SAC sim-to-real transfer — currently simulation-only
- SmolVLA with a frozen-backbone fine-tuning strategy (Option A/B in [[algo-smolvla]]) — proposed follow-up to test the distractor hypothesis

---

## Related pages
- [[decision-guide]] — full REX narrative, training/robustness guidance
- [[evaluation-protocol]] — evaluation methodology and test matrix
- [[pick-and-place]] — task-level synthesis
- [[overview]] — three-paradigm framework and experiment tracker
- [[algo-sac]], [[algo-act]], [[algo-smolvla]] — per-algorithm detail
