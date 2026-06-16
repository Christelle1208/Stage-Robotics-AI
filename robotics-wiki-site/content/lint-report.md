# Wiki Health Report

**Generated:** 2026-05-22 | **Scope:** all 35 files in `wiki/` + 57 files in `raw_sources/`
**Next scheduled run:** 2026-05-29

---

## Summary

| Category | Issues found | Severity |
|----------|-------------|----------|
| Contradictions | 3 | 🔴 Fix immediately |
| Missing metrics (source vs wiki) | 3 | 🟡 Update soon |
| Orphan / low-inbound algo pages | 3 | 🟡 Add links |
| Concepts needing dedicated pages | 7 | 🟡 Create pages |
| Potentially outdated framing | 2 | 🟢 Review |
| Broken internal references | 1 | 🔴 Fix immediately |

---

## 🔴 Contradictions

### C-1 · RT-2-X benchmark score: 62% vs 62.5%

**Files:** `algo-octo.md:124` vs `algo-openvla.md:105`

`algo-octo.md` reports RT-2-X achieving **62%** task success; `algo-openvla.md` reports **62.5%** for the same model. These are likely from different evaluation protocols (Octo uses its own fine-tuning benchmark; OpenVLA uses BridgeV2 29-task suite), but both pages present the number without qualification, making them look contradictory to a reader comparing tables.

**Fix:** Add a footnote in each table clarifying the evaluation protocol. Example:
- `algo-octo.md` table caption: *(RT-2-X score on Octo's multi-platform fine-tuning benchmark)*
- `algo-openvla.md` table caption: *(RT-2-X score on BridgeV2 29-task evaluation)*

---

### C-2 · VLA-RL "+4.5% on 40 tasks" — misleading phrasing

**File:** `vision-language-action-models.md:99`

> "OpenVLA-7B surpasses strongest fine-tuned baseline by **4.5% on 40 tasks**"

This implies a uniform +4.5% across 40 tasks. The truth (correctly stated in `algo-vla-rl.md:130`) is that 4.5% is the **average across 4 LIBERO suites**, each containing 10 tasks — individual suite gains range from +4.2% to +4.8%.

**Fix:** Change to:
> "OpenVLA-7B surpasses strongest fine-tuned baseline by an average of **+4.5% across LIBERO's 40 tasks** (4 suites × 10 tasks)"

---

### C-3 · GF-VLA placement accuracy not captured

**Files:** `vision-language-action-models.md:121`, `grasping-and-manipulation.md:64` vs raw source

Raw source (`Information-theoretic graph fusion...`) reports **three** key metrics:
- 94% grasp success ✅ — captured in wiki
- **89% placement accuracy** ❌ — missing from wiki
- 90% task success ✅ — captured in wiki

**Fix:** Add "**89% placement accuracy**" to both entries where GF-VLA results appear.

---

## 🔴 Broken Internal Reference

### B-1 · Broken anchor link in `reinforcement-learning.md`

**File:** `reinforcement-learning.md:17`

> "SAC balances exploration and exploitation... (see `[[#Composable-DRL-SQL]]` below for SQL)"

The anchor `#Composable-DRL-SQL` will not resolve in Obsidian — the actual heading is `### Composable Deep Reinforcement Learning for Robotic Manipulation (SQL) → [[algo-sql-sac]]`. Obsidian generates anchors from heading text after stripping special characters, making the actual anchor something like `#composable-deep-reinforcement-learning-for-robotic-manipulation-sql--algo-sql-sac`.

**Fix:** Replace the anchor with an explicit wikilink:
> "SAC balances exploration and exploitation... (see [[algo-sql-sac]] for SQL foundations)"

---

## 🟡 Missing Metrics (source richer than wiki)

### M-1 · GigaWorld-Policy: missing π0-5 comparison

**File:** `world-models.md:15–25`

Raw source abstract states: *"compared with pi-0.5, GigaWorld-Policy improves performance by **95% on RoboTwin 2.0**"* — a striking result not captured anywhere in the wiki. The wiki only mentions the Motus comparison.

**Fix:** Add to `world-models.md` GigaWorld-Policy entry:
> Also outperforms **π0-5 by 95%** on the RoboTwin 2.0 benchmark.

---

### M-2 · Mamba2Diff: CGRU module missing

**File:** `imitation-learning.md:75–81`

The wiki mentions only the BDGM module. Raw source describes two components: BDGM (global long-term features) and **CGRU (Convolution-Enhanced Gated Recurrent Unit)** for short-term dependencies. The two modules together explain why Mamba2Diff outperforms diffusion-only and non-diffusion baselines.

**Fix:** Add to the Mamba2Diff entry:
> Combines two modules: **BDGM** for long-horizon global features (bidirectional) and **CGRU** for short-term local dependencies. Evaluated on 3 simulated environments + 1 custom real-world platform.

---

### M-3 · GF-VLA: graph accuracy metric missing

**File:** `vision-language-action-models.md:118–123`, `grasping-and-manipulation.md:61–66`

Raw source also reports **95% graph accuracy** and **93% subtask segmentation** for the scene graph representation — neither appears in the wiki. These intermediate metrics explain *why* the 94%/90% task results are achievable.

**Fix:** Add to GF-VLA entries:
> Scene graph representation achieves **95% graph accuracy** and **93% subtask segmentation**, enabling the LLM planner to generate reliable task policies.

---

## 🟡 Orphan / Low-Inbound Algorithm Pages

Pages with only **2 inbound links** (index.md + one topic page) are effectively isolated — a reader exploring the wiki cannot reach them from multiple entry points.

### O-1 · `algo-safevla` — 2 inbound links

Only referenced from `vision-language-action-models.md` and `index.md`. SafeVLA uses **constrained RL (CMDP)** — it should appear in `reinforcement-learning.md` under a "Safe RL" or "Constrained RL" subsection. Its long-horizon mobile manipulation evaluation also connects to `simulation-and-tools.md`.

**Fix:**
- Add a brief entry in `reinforcement-learning.md` under a new `## Safe / Constrained RL` section → `[[algo-safevla]]`
- Add `See: [[algo-safevla]]` to the SafeVLA mention in `hybrid-il-rl.md` Related Topics

---

### O-2 · `algo-namr-rrt` — 3 inbound links

Referenced from `trajectory-planning.md`, `pick-and-place.md` (via trajectory planning), and `index.md`. The real-world validation was conducted in dynamic crowded outdoor environments (shopping malls) — this connects to `simulation-and-tools.md` (real-world testing) and `grasping-and-manipulation.md` is unrelated, but `pick-and-place.md` could link it more explicitly.

**Fix:**
- Add `See: [[algo-namr-rrt]]` to the trajectory comparison table row in `pick-and-place.md` "Long-Horizon" section

---

### O-3 · `algo-qlora` — 3 inbound links

Referenced from `llms-for-robotics.md`, `vision-language-action-models.md`, and `index.md`. QLoRA is the enabling technology for fine-tuning OpenVLA on consumer GPUs — it should also appear in `imitation-learning.md`'s discussion of fine-tuning VLA models from demonstrations.

**Fix:**
- Add `See: [[algo-qlora]]` to `imitation-learning.md` Related Topics section

---

## 🟡 Concepts Needing Dedicated Pages

The following models have **substantial source coverage** (full papers in `raw_sources/`) and are mentioned ≥4 times across topic pages, but have no `algo-*.md` explainer. They currently appear only as brief entries in topic pages, making them second-class citizens compared to peers like `algo-octo.md` or `algo-act.md`.

| Concept | Mentions | Source file | Priority |
|---------|----------|-------------|----------|
| **SmolVLA** | 6 | `SmolVLA A Vision-Language-Action Model...` | High |
| **TinyVLA** | 8 | `TinyVLA Towards Fast, Data-Efficient...` | High |
| **X-VLA** | 4 | `X-VLA Soft-Prompted Transformer...` | High |
| **GF-VLA** | 6 | `Information-theoretic graph fusion...` | High |
| **GigaWorld-Policy** | 4 | `GigaWorld-Policy An Efficient...` | Medium |
| **Mamba2Diff** | 4 | `Mamba2Diff An enhanced diffusion...` | Medium |
| **Cosmos-Reason1** | 4 | `Cosmos-Reason1 From Physical Common Sense...` | Medium |

**Suggested creation order:** SmolVLA → TinyVLA → X-VLA → GF-VLA (all 2025–2026 VLA family, reads as a coherent set), then GigaWorld-Policy → Mamba2Diff → Cosmos-Reason1.

**Note on DDPG:** 16 mentions across the wiki, but no source paper for DDPG exists in `raw_sources/`. DDPG appears only as a baseline in HER and Multi-Goal RL papers. A brief inline definition in `reinforcement-learning.md` (rather than a full algo page) would be sufficient.

---

## 🟢 Potentially Outdated Framing

### F-1 · VLA paradigm intro overstates the size-performance tradeoff

**File:** `vision-language-action-models.md:14`

> "large models (7B+ parameters) are slow at inference and expensive to train"

This was accurate when written (against RT-2-X's 55B). But the 2025 papers in the wiki now demonstrate:
- SmolVLA (<1B) matches VLAs 10× larger
- TinyVLA (<1B) is faster and more data-efficient than OpenVLA 7B
- X-VLA 0.9B achieves SOTA across 9 platforms

The framing implies 7B is the minimum viable size, which the newer papers contradict.

**Fix:** Update to:
> "The main tradeoff: large models (7B+) were historically slow and expensive, motivating a wave of efficient sub-1B VLAs (SmolVLA, TinyVLA, X-VLA) that now match or exceed 7B baselines."

---

### F-2 · `algo-vla-rl.md` Con: "simulation-only evaluation" may be stale

**File:** `algo-vla-rl.md:167`

> "Simulation-only — evaluation is entirely in LIBERO (simulation); real-robot results not reported"

This was accurate at paper submission (2025). If a subsequent version or extended paper exists in `raw_sources/` with real-robot results, this con should be updated. No such update was found in current raw sources — but flag for review when ingesting new sources.

---

## 🟢 Structural Notes

### S-1 · VLA comparison table missing Molmo family

**File:** `vision-language-action-models.md:156–167`

The comparison table has 8 rows (Octo through GF-VLA) but the Molmo Family section immediately above it (MolmoAct2, MolmoB0T) is not in the table. This creates an asymmetry where the table appears to summarize the full page but silently omits 3 models.

**Fix:** Add rows:
```
| MolmoAct2 | — | Yes | Yes | Explicit reasoning chains; real-world deployment |
| MolmoB0T  | — | Yes | Zero-shot | Zero-shot via simulation scale (MolmoSpaces) |
```

---

### S-2 · `algo-sql-sac.md` ↔ `algo-sac.md` cross-reference

Both pages cover related content (SQL is the precursor to SAC). Verify that `algo-sql-sac.md` contains an explicit `See: [[algo-sac]]` link, and vice versa. If not, add them. (Check: `algo-sql-sac.md` should have been updated to reference `algo-sac.md` when the SAC page was created — confirm this is the case.)

---

## Fixes Prioritized by Effort

| Priority | Fix | Effort |
|----------|-----|--------|
| 🔴 Now | C-3 · Add GF-VLA placement accuracy (89%) to 2 pages | 2 edits |
| 🔴 Now | B-1 · Fix broken anchor in `reinforcement-learning.md` | 1 edit |
| 🔴 Now | C-2 · Clarify VLA-RL "4.5% on 40 tasks" phrasing | 1 edit |
| 🔴 Now | C-1 · Add evaluation-context footnotes to RT-2-X scores | 2 edits |
| 🟡 Soon | M-1 · Add GigaWorld π0-5 result to `world-models.md` | 1 edit |
| 🟡 Soon | M-2 · Add CGRU module to Mamba2Diff entry | 1 edit |
| 🟡 Soon | M-3 · Add graph accuracy to GF-VLA entries | 2 edits |
| 🟡 Soon | O-1 · Add SafeVLA to `reinforcement-learning.md` Safe RL section | 1 edit |
| 🟡 Soon | F-1 · Update VLA paradigm intro tradeoff description | 1 edit |
| 🟡 Soon | S-1 · Add Molmo family to VLA comparison table | 1 edit |
| 🟡 Later | O-2, O-3 · Extra links for NAMR-RRT and QLoRA | 2 edits |
| 🟢 Later | Create algo pages: SmolVLA, TinyVLA, X-VLA, GF-VLA | 4 new files |
| 🟢 Later | Create algo pages: GigaWorld-Policy, Mamba2Diff, Cosmos-Reason1 | 3 new files |

---

*Lint methodology: (1) grep-based inbound link count for all 23 algo pages and 10 topic pages; (2) cross-file grep for key numeric claims; (3) raw source abstract spot-check for 8 papers; (4) manual review of all topic page `*Algorithm:*` and `*Tags:*` fields read in this session.*
