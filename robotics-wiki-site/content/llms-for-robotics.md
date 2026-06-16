# Large Language Models and Foundation Models for Robotics

Large language models (LLMs) and vision-language models (VLMs) are transforming robot intelligence. Rather than hand-coding task logic, robots can leverage internet-scale pretrained models for planning, reasoning, reward design, and action generation. This page covers survey-level overviews, foundational models, and enabling technologies. See also [[vision-language-action-models]], [[world-models]].

---

## Surveys

### A Survey of Robot Intelligence with Large Language Models
**Jeong, Lee, Kim, Shin — Dong-A University / Pukyong National University, 2024**

Comprehensive survey of how LLMs and VLMs boost robot intelligence across five categories:

1. **Reward design in RL**: LLMs generate reward functions automatically (e.g., Eureka model for automating reward function design)
2. **Low-level control**: VLMs map visual observations to motor commands
3. **High-level planning**: LLMs decompose complex instructions into executable subtasks
4. **Manipulation**: RT-2-style models integrate visual data, language, and robot actions in VLA models
5. **Scene understanding**: VLMs describe scenes in natural language and ground objects

Key systems covered: Eureka (automated reward design), RT-2 (VLA), AutoRT (LLM-driven task generation + policy execution).

*Tags:* survey, LLMs, VLMs, reward design, planning, manipulation, scene understanding

---

## Foundation Models for Embodied Reasoning

### Gemini Robotics-ER 1.5: Embodied Reasoning for Physical Agents → [[algo-gemini-robotics-15]]
**Gemini Robotics Team — Google DeepMind, 2025**

A frontier VLM specialized for embodied reasoning — visuo-spatial-temporal understanding of the physical world. Key capabilities: **complex pointing** (localizing parts, generating manipulation trajectories), **progress understanding** (multi-view success detection, task completion estimation, video frame ordering), and **tool use** for agentic orchestration. Acts as the orchestrator in the Gemini Robotics 1.5 agentic system, combining with the GR 1.5 VLA for long-horizon task execution.

Achieves SOTA on embodied reasoning benchmarks while retaining frontier-model generality. Outperforms Gemini 2.5 Pro and GPT-5 on spatial/embodied tasks. Performance scales with inference-time thinking budget, better than Gemini 2.5 Flash on embodied tasks.

*Tags:* embodied reasoning, spatial understanding, success detection, pointing, orchestrator, agentic, 2025 | See: [[vision-language-action-models]], [[world-models]]

---

### Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning
**NVIDIA, 2026**

A foundation model focused on **physical common sense** as a prerequisite for embodied reasoning. Unlike pure language models that reason about abstract concepts, Cosmos-Reason1 is grounded in the physics of the real world — understanding that objects fall, surfaces resist, liquids flow.

This physical reasoning layer enables robots to make predictions about consequences of actions before executing them — a form of model-based reasoning without a learned world model. Positioned as a foundation for VLA systems that need physical intuition.

*Tags:* physical common sense, embodied reasoning, NVIDIA, foundation model, 2026 | See: [[world-models]]

---

## Enabling Technologies for LLM-Based Robotics

### QLoRA: Efficient Finetuning of Quantized LLMs → [[algo-qlora]]
**Dettmers, Pagnoni, Holtzman, Zettlemoyer — University of Washington, 2023**

Introduced QLoRA — a method for fine-tuning large language models (65B parameters) on a **single 48GB GPU** by combining:
- **4-bit NormalFloat (NF4)** quantization: a data type theoretically optimal for normally distributed weights
- **Double quantization**: reduces memory footprint by quantizing the quantization constants themselves
- **Paged optimizers**: manage memory spikes during training
- **LoRA (Low-Rank Adaptation)**: gradients backpropagate through the frozen quantized model into small low-rank adapter weights

The **Guanaco** model family (fine-tuned with QLoRA) reaches **99.3% of ChatGPT performance** on the Vicuna benchmark within 24 hours of fine-tuning on a single GPU.

Relevance to robotics: QLoRA enables fine-tuning large VLA models (like [[algo-openvla|OpenVLA 7B]]) on consumer hardware without sacrificing performance. Referenced explicitly in the OpenVLA paper.

*Technique:* 4-bit quantization + LoRA | *Tags:* fine-tuning, quantization, memory efficiency, LLM, 2023 | See: [[vision-language-action-models]], [[algo-openvla]]

---

### Molmo2: Open Weights and Data for VLMs with Video Understanding and Grounding → [[algo-molmo2]]
**2026**

Open-weights VLM with capabilities in:
- Video understanding: processes temporal visual sequences
- Visual grounding: localizes objects referenced in text

Molmo2 is the foundation model for the Molmo robotics ecosystem ([[algo-molmoact2|MolmoAct2]], [[algo-molmobot|MolmoB0T]], [[algo-molmospaces|MolmoSpaces]]). Releasing weights and training data openly is a key contribution — enabling the research community to build on top of it.

*Tags:* VLM, open weights, video understanding, visual grounding, Molmo ecosystem, 2026 | See: [[vision-language-action-models]], [[simulation-and-tools]], [[algo-molmoact2]], [[algo-molmobot]], [[algo-molmospaces]]

---

## The LLM → VLA Pipeline

A typical LLM-enhanced robotics system works as:

```
Natural Language Instruction
        ↓
  LLM / VLM (high-level planner)
        ↓
  Task decomposition → subtask sequence
        ↓
  VLA model (per-subtask visuomotor policy)
        ↓
  Robot actions
```

LLMs provide semantic reasoning and task decomposition; VLAs translate high-level subtasks into concrete robot movements. See [[vision-language-action-models]] for VLA details.

The Gemini Robotics 1.5 agentic system ([[algo-gemini-robotics-15]]) is a concrete realization of this architecture, replacing the generic LLM with a robotics-specialized embodied reasoning model (GR-ER 1.5) as orchestrator — cutting planning failure rates from 25.5% to 9%.

---

## LLMs for Reward Design

A growing application: using LLMs to write reward functions automatically (instead of hand-engineering them for RL). The Eureka system (mentioned in the LLM survey) prompts an LLM with task descriptions and generates Python reward functions for MuJoCo environments. This removes one of the major bottlenecks in applying RL to new tasks.

See [[reinforcement-learning]] for the RL side of this.

---

## Related Topics
- [[vision-language-action-models]] — VLAs as the downstream output of LLM research for robotics
- [[world-models]] — world models leveraging language understanding
- [[reinforcement-learning]] — LLMs as reward designers for RL
- [[imitation-learning]] — IL datasets annotated with language instructions
