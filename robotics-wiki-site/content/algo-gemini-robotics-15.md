# System: Gemini Robotics 1.5 — Thinking VLA + Embodied Reasoning Agent

**Paper:** "Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer" — Gemini Robotics Team, Google DeepMind, 2025
**Category:** Multi-embodiment VLA + Embodied Reasoning VLM + Agentic System
**Related:** [[vision-language-action-models]], [[llms-for-robotics]], [[world-models]], [[grasping-and-manipulation]], [[simulation-and-tools]]

---

## Problem Being Solved

Prior VLA models (including the original Gemini Robotics) face three compounding limitations:

1. **Single-embodiment fragility** — each robot requires its own trained checkpoint; skills cannot transfer across different hardware
2. **Blind action execution** — models map instructions directly to actions without explicit reasoning, limiting performance on multi-step tasks and making failures opaque
3. **Flat orchestration** — complex, long-horizon tasks (tool use, web search, memory, error recovery) require a planner that understands the physical world, not just a generic LLM

Gemini Robotics 1.5 addresses all three with a two-model family: **GR 1.5** (the VLA) and **GR-ER 1.5** (the Embodied Reasoning VLM), combined into a unified agentic system.

---

## Architecture

### The Two-Model Family

```
┌─────────────────────────────────────────────────────────┐
│                  GR 1.5 Agentic System                  │
│                                                         │
│  User ──► GR-ER 1.5 (Orchestrator / VLM)               │
│                │   • World understanding                 │
│                │   • Long-horizon planning               │
│                │   • Tool use (web search, code...)     │
│                │   • Success detection                   │
│                ▼                                         │
│           GR 1.5 (Action Model / VLA)                   │
│                │   • Open-vocabulary instructions       │
│                │   • Thinking traces (optional)          │
│                │   • Multi-embodiment control           │
│                ▼                                         │
│              Robot actions                               │
└─────────────────────────────────────────────────────────┘
```

Both models inherit Gemini's multimodal world knowledge. GR-ER 1.5 retains the full Gemini capability stack (reasoning, tool use, code, video); GR 1.5 specializes in translating mid- and short-horizon instructions into physical robot actions.

---

## Key Innovation 1: Motion Transfer (MT)

The central training innovation enabling multi-embodiment generalization.

**The challenge:** Different robots have incompatible action spaces, kinematics, and morphology. Naively mixing data from ALOHA (bimanual tabletop), Bi-arm Franka (industrial arms), and Apollo (humanoid) yields interference rather than transfer.

**Motion Transfer solution:** A new architecture and training recipe that:
- Builds a **unified representation of motion and physical interaction** shared across embodiments
- Aligns action spaces so that the model extracts cross-embodiment commonalities (reaching, grasping, placing) from embodiment-specific trajectories
- Enables **zero-shot skill transfer**: a skill learned on ALOHA can be executed on a Franka or Apollo without additional training data for that embodiment

**Ablation confirmation (Fig. 4):**

| Training configuration | ALOHA task gen. | Franka visual gen. |
|------------------------|----------------|-------------------|
| Single embodiment, no MT | 0.60 | 0.38 |
| Multi-embodiment, no MT | 0.65 | 0.61 |
| **Multi-embodiment + MT (GR 1.5)** | **0.81** | **0.77** |

Multi-embodiment data helps, but MT is required to fully exploit it.

---

## Key Innovation 2: Embodied Thinking (Thinking VLA)

GR 1.5 can "think before acting": it generates **natural-language thinking traces** interleaved with its action tokens, decomposing complex instructions into explicit reasoning steps before executing them.

**Two-stage decomposition:**
1. **Thinking stage:** Convert complex instruction → sequence of specific short-horizon steps (e.g., "sort the clothes" → "move gripper left toward the red shirt")
2. **Action stage:** Map each short-horizon step → low-level robot actions

**Three benefits beyond raw performance:**
- **Interpretability:** Thinking traces expose the robot's planned actions to human observers, building trust
- **Situational awareness:** The model implicitly detects subtask completion and transitions automatically (e.g., "pick up ball" → "put ball in bag" once grasped — without an explicit success detector)
- **Recovery behaviors:** When an object slips from one hand, the next thinking trace immediately proposes recovery with the other hand

**Performance on multi-step benchmark (Fig. 6):**

| Mode | ALOHA | Bi-arm Franka | Humanoid |
|------|-------|--------------|----------|
| GR 1.5 (no thinking) | 0.26 | 0.55 | 0.51 |
| **GR 1.5 (thinking ON)** | **0.55** | **0.60** | **0.67** |

Thinking yields roughly +50–100% relative improvement on multi-step tasks.

---

## Key Innovation 3: GR-ER 1.5 — Embodied Reasoning Model

GR-ER 1.5 is a frontier VLM specialized for embodied reasoning: visuo-spatial-temporal understanding of the physical world needed for robot orchestration.

**Core capabilities:**
- **Complex pointing:** Localizing precise object parts, predicting manipulation trajectories, reasoning about physical/semantic constraints (e.g., "point to an object lighter than 10 pounds")
- **Progress understanding:** Estimating percentage task completion, multi-view success detection, video frame temporal ordering
- **Tool use:** Web search, code execution, memory — enabling tasks like "sort trash" (requires knowing recycling rules via web search) or "nut allergy" (checking ingredient databases)
- **Inference-time scaling:** Performance improves with more thinking tokens; GR-ER 1.5 scales better than Gemini 2.5 Flash on embodied tasks

**Embodied reasoning benchmark vs. frontier models (Fig. 8):**

GR-ER 1.5 expands the Pareto frontier of embodied reasoning vs. generality, outperforming Gemini 2.5 Pro, Gemini 2.5 Flash, GPT-5, and GPT-5-mini on spatial/embodied benchmarks while retaining comparable general-purpose capability.

**Complex pointing (Fig. 10, average across 5 benchmarks):**

| Model | Avg. Pointing |
|-------|--------------|
| **GR-ER 1.5** | **52.6** |
| Gemini Robotics-ER | 49.1 |
| Gemini 2.5 Pro | 39.7 |
| GPT-5 | 30.8 |
| GPT-5-mini | 27.1 |

---

## Evaluation

### Setup
- **Benchmark:** 230 tasks across ALOHA, Bi-arm Franka, Apollo Humanoid
- **Protocol:** A/B/n interleaved real-robot testing (reduces cross-robot variance); MuJoCo simulation used for >90% of development episodes (strong rank consistency with real hardware confirmed)
- **Metric:** Progress score (continuous, finer-grained than binary success rate); success rate reported in appendix

### Short-Horizon Generalization (Fig. 3)

GR 1.5 vs. Gemini Robotics vs. Gemini Robotics On-Device (GRoD):

| Axis | GR 1.5 (ALOHA) | Gemini Robotics | GRoD |
|------|----------------|----------------|------|
| In-distribution | 0.83 | 0.81 | 0.78 |
| Instruction gen. | 0.76 | 0.70 | 0.66 |
| Action gen. | 0.54 | 0.53 | 0.67 |
| Visual gen. | 0.81 | 0.73 | 0.83 |
| Task gen. | 0.70 | 0.44 | 0.41 |

**Task generalization is the hardest axis** (requires all three: visual robustness + instruction understanding + motion adaptation). GR 1.5 shows the largest gain here (+26 pp over GRoD on ALOHA).

### Long-Horizon Agentic Benchmark (Fig. 17, Table 1)

8 complex tasks (tool use, memory, 9-step planning, dexterous manipulation):

| System | Progress score | Planning failures | Total failures |
|--------|---------------|------------------|----------------|
| GR 1.5 (Thinking ON) | ~44% | — | — |
| Gemini 2.5 Flash + GR 1.5 | ~60% | 25.5% | 44.5% |
| **GR-ER 1.5 + GR 1.5 (full agent)** | **~80%** | **9%** | **22%** |

Using GR-ER 1.5 as orchestrator (vs. Gemini 2.5 Flash) cuts planning failures by **64%** and halves total failure rate.

---

## Why It Works

**Motion Transfer** succeeds because it treats motion as a universal physical language. While action tokens differ across embodiments (7-DoF arm vs. 22-DoF humanoid), the underlying physics of manipulation (contact forces, object affordances, spatial trajectories) are shared. MT extracts this invariant physical structure.

**Embodied Thinking** works for the same reason chain-of-thought helps language reasoning: breaking a hard problem (complex instruction → low-level action) into two easier problems (instruction → language plan, language plan → primitive action). The VLM backbone excels at the first step; the action model excels at the second.

**GR-ER 1.5 as orchestrator** outperforms a generic Gemini model because embodied reasoning is a distinct capability — a model may excel at math and code but fail at spatial pointing and progress estimation. GR-ER 1.5 is specialized for exactly what orchestration needs.

---

## Pros

- **True multi-embodiment generalization** — single checkpoint controls ALOHA, Franka, and a humanoid; zero-shot cross-embodiment skill transfer demonstrated
- **"Think before act" interpretability** — thinking traces make the robot's intent legible to humans; critical for trust in deployment
- **Strong agentic performance** — ~80% progress on complex long-horizon tasks (tool use, memory, planning) vs. ~44% for VLA alone
- **SOTA embodied reasoning** — GR-ER 1.5 outperforms GPT-5 and Gemini 2.5 Pro on spatial/embodied benchmarks
- **Safety-first architecture** — multi-layered safety (semantic, dialogue, physical); ASIMOV-2.0 benchmark for adversarial safety evaluation
- **Simulation-augmented development** — >90% of dev in MuJoCo with verified rank consistency; dramatically accelerates iteration

## Cons

- **Closed and proprietary** — no open weights, no publicly available code; researchers cannot reproduce or build on GR 1.5 directly (contrast: [[algo-openvla]], [[algo-octo]])
- **Inference latency** — GR-ER 1.5 used as real-time success detector runs at 5Hz; stale predictions are flagged as a limitation; thinking mode adds further latency
- **Humanoid data gap** — MT is less effective when the embodiment gap is large (humanoid vs. arm); requires more humanoid-specific data to close
- **Evaluation breadth unclear** — 230 tasks is substantial but all in-house; no evaluation on standardized open benchmarks (LIBERO, BridgeV2) makes comparison with [[algo-openvla]] or [[algo-vla-rl]] indirect
- **Commercial availability unknown** — deployment path for external researchers and companies not specified in the report

---

## In This Wiki

**Category:** [[vision-language-action-models]] — VLA family; compare with [[algo-openvla]] (open-source, 7B), [[algo-octo]] (modular generalist), [[algo-vla-rl]] (RL fine-tuning), [[algo-safevla]] (safety alignment)

**Cross-embodiment:** shares the soft-prompting goal of [[algo-openvla]] and X-VLA, but uses Motion Transfer (architecture-level) rather than prompt-level adaptation

**Thinking VLA:** directly comparable to [[algo-molmoact2]] (Molmo family's action reasoning) and MolmoAct2's `[ACT]` token boundary — both interleave reasoning with actions, but GR 1.5 uses continuous natural-language traces rather than discrete token boundaries

**Safety:** the ASIMOV-2.0 benchmark and multi-layered safety approach complements [[algo-safevla]]'s CMDP framework — both address VLA safety but from different angles (reasoning-based vs. constrained-RL)

**Simulation:** uses [[simulation-and-tools|MuJoCo]] for >90% of development; achieves rank consistency with real hardware (similar claim to [[algo-molmospaces]]'s R=0.96 sim-to-real correlation)

**Agentic system:** the GR-ER 1.5 orchestrator pattern mirrors the LLM→VLA pipeline described in [[llms-for-robotics]], but uses a robotics-specialized model rather than a general LLM
