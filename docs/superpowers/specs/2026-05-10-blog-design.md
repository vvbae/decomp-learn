# Blog Design Spec: Decomposition Learning — Negative Result

**Date:** 2026-05-10
**Author:** viviwei
**Status:** Approved by author

---

## Overview

A hybrid personal/technical blog post documenting a first-time robotics ML experiment by an SDE with no prior ML research background. The experiment tested whether decomposing a manipulation task (peg-in-hole) into a geometric part (motion planning) and a contact part (learned policy) could beat an end-to-end Diffusion Policy baseline. It didn't — and the blog honestly documents why.

**Core message:** "I had a clean hypothesis, ran the experiment properly, and it failed. Here's exactly what broke and what I'd do differently."

**Tone:** Hybrid — personal/reflective framing, technical depth in the middle sections.

**Languages:** English (primary) + Chinese (adapted translation, not word-for-word).

**Publications:**
- English: GitHub Pages (Markdown, inline images, YouTube embeds)
- Chinese: Zhihu + WeChat Official Account (images uploaded directly, Bilibili embeds; WeChat gets a brief platform intro)

---

## Section Breakdown

### Section 1 — The Origin (~300 words)
**Purpose:** Hook the reader with the human story before any technical content.

**Content:**
- Author background: SDE, no prior ML/robotics research experience
- What drew them to embodied AI: the gap between impressive demos and actual generalization
- Why the decomposition idea felt compelling: "replace what geometry can solve, learn only what physics makes hard"
- Introduce the task: peg-in-hole (PegInsertionSide-v1 in ManiSkill3)

**Media:** One success video embedded at the end of this section — show the reader what a successful episode looks like before any explanation.

---

### Section 2 — The Hypothesis (~300 words)
**Purpose:** State the idea clearly so the reader understands what's being tested.

**Content:**
- Task decomposition: geometric part (motion planner) vs contact part (Diffusion Policy)
- Key design choice: policy input = relative TCP-to-hole position, not joint angles → embodiment-agnostic
- Two claims being tested:
  1. Data efficiency: contact policy should reach the same success rate with fewer demos
  2. Cross-embodiment: contact policy trained on Panda should transfer zero-shot to xArm6
- Brief ASCII/text diagram of the pipeline:

```
Demo data → split at contact onset (force > 0.1N)
  ↓
Geometric part → replaced by motion planner at eval time
Contact part   → Diffusion Policy trained on relative coords
```

---

### Section 3 — Building It (~400 words)
**Purpose:** Show the experiment setup honestly, including mistakes. This is where the outsider narrative lands hardest.

**Content:**
- Tools: ManiSkill3, 996 expert demos, LeRobot framework, Diffusion Policy
- Mistakes made and fixed:
  - Wrong control mode initially (absolute joint angles → switched to delta)
  - No train/test split initially — all 996 demos used for training, no held-out set
  - How the split was fixed: 800 train / 200 held-out test (demos 796–995)
- What the contact split looks like: ~29 steps per episode (short), ~80% of trajectory is geometric
- Brief mention of the tooling learning curve (uv, tmux, LeRobot configs)

**Tone note:** Be direct about the mistakes. "I was wrong to skip a test set" is more credible than glossing over it.

---

### Section 4 — The Numbers (~250 words)
**Purpose:** Present results clearly and without editorializing. Let the numbers speak.

**Content:**
- E2E baseline result: **65.5%** on 200 held-out test episodes (800 demos, 200k steps)
- Contact policy sweep table:

| Demos | Success Rate |
|-------|-------------|
| 50    | 36.0%       |
| 100   | 42.5%       |
| 200   | 40.0%       |
| 500   | 37.0%       |
| 796   | 49.5%       |

- Learning curve chart: flat from 1k to 100k steps (39–56%, no trend)
- One-line observation: "More training doesn't help. More data doesn't help. Something more fundamental is wrong."

**Media:** Learning curve chart (to be generated from `results/contact_policy_v1/learning_curve_796demos.txt`).

---

### Section 5 — The Diagnosis (~500 words)
**Purpose:** The technical core. Explain the failure mode precisely.

**Content:**
- Failure video first — show what failure looks like before explaining it
- The failure visualization (`viz_failures.py`): ran on contact policy 20k checkpoint, 50 test episodes
- Walk through each of the three plots:
  1. `insertion_depth.png`: successes insert steadily; failures never reach the hole entrance
  2. `lateral_error.png`: failures scatter 3–4× the hole radius laterally
  3. `final_lateral.png`: completely bimodal — no near-misses, either nails it or fails entirely
- Key numbers: lateral mean = 54.7mm, hole radius ≈ 15mm
- Explain covariate shift:
  - Training demos come from a scripted policy on a fixed approach path → contact onset states have a narrow distribution
  - At eval time, small variations in initial state put the robot in a contact onset state the policy has never seen
  - Policy outputs garbage — not because the model is too small or undertrained, but because the input is out-of-distribution
- Why this can't be fixed by training longer or adding architecture capacity

**Media:** Failure video + all 3 diagnostic PNGs inline.

---

### Section 6 — The Fix That Didn't Work (~300 words)
**Purpose:** Show that you tried to fix the problem systematically, and be honest about why it still failed.

**Content:**
- The idea: offline data augmentation at contact onset
  - Take each demo's contact onset frame
  - Add random y/z perturbations (±3mm, ±7mm, ±15mm)
  - Let the scripted policy recover, keep successful trajectories
- Why this is simpler than DAgger (no eval loop changes, fully offline)
- What it produced: 4539 trajectories from 996 original demos
- Result: **41.0%** — worse than unaugmented (49.5%)
- Why augmentation failed: the scripted policy's own recovery behavior is stereotyped — adding offset copies of the same correction pattern doesn't add real behavioral diversity. The ceiling is the scripted policy's own diversity, which is low.

**No media needed here** — the number (41%) is the finding.

---

### Section 7 — What I Learned (~400 words)
**Purpose:** The payoff section. Three layers of reflection.

**Content:**

**Technical:**
- Covariate shift is invisible until you run the experiment. The training loss was fine. The architecture was fine. The failure lived entirely in the data distribution.
- "Contact segment" from scripted demos is not what you think it is: it's a narrow, stereotyped path, not a diverse set of contact-handling behaviors
- A proper train/test split is not optional even for imitation learning — especially when doing comparative experiments

**Research process:**
- Run the diagnostic (failure visualization) early — not after exhausting other explanations
- Negative results have a natural stopping criterion: when you can explain *why* it fails and the explanation points to a structural problem, not a tuning problem, you stop
- The augmentation experiment was worth running even though it failed — it ruled out the simplest fix

**Meta (outsider perspective):**
- The gap between "I understand this conceptually" and "I can run this experiment" is large
- ML research infrastructure (frameworks, configs, evaluation pipelines) takes real time to learn
- The things that surprised me most were not the ML parts but the experimental hygiene parts (test splits, reproducibility, proper baselines)

---

### Section 8 — What Would Actually Work (~200 words)
**Purpose:** Close with intellectual honesty about the path forward. Not promises — directions.

**Content:**
- **DAgger**: the principled fix for covariate shift. Requires online data collection but would produce genuinely diverse recovery behaviors
- **Longer contact tasks**: peg-in-hole contact segment is only ~29 steps. Tasks with richer contact dynamics (PlugCharger-v1, already analyzed) would give the policy more to learn
- **Cross-embodiment**: still an open question — the relative-coordinate design is sound, but needs to be tested once the single-robot version works
- Brief closing: the experiment failed on its primary claim, but the infrastructure is in place and the failure is understood. That's enough to build on.

---

## Assets

### Existing (in `~/viviwei/blog/`)
| Asset | Location | Used in |
|-------|----------|---------|
| Success video | `figures/videos/success/ep1_seed2_success.mp4` | Section 1 |
| Failure video | `figures/videos/failure/ep3_seed5_failure.mp4` | Section 5 |
| `insertion_depth.png` | `figures/failure_analysis/` | Section 5 |
| `lateral_error.png` | `figures/failure_analysis/` | Section 5 |
| `final_lateral.png` | `figures/failure_analysis/` | Section 5 |

### To Generate
| Asset | Source data | Used in |
|-------|-------------|---------|
| `learning_curve.png` | `results/contact_policy_v1/learning_curve_796demos.txt` — line chart, x=training steps (1k–100k), y=success rate (%) | Section 4 |
| `sweep_comparison.png` | `results/contact_policy_v1/sweep_results.txt` + E2E 65.5% — grouped bar chart: x=demo count (50/100/200/500/796), two bars per group (E2E vs contact policy) | Section 4 |

---

## Output Files

```
~/viviwei/blog/
├── posts/
│   ├── en.md          ← English (GitHub Pages source)
│   └── zh.md          ← Chinese (Zhihu + WeChat)
└── figures/
    ├── (existing PNGs — already present)
    ├── learning_curve.png     ← to generate
    └── sweep_comparison.png   ← to generate
```

---

## Bilingual Notes

- Write English first, in full
- Chinese translation: adapted (not literal) — Zhihu/WeChat readers expect a slightly more conversational register
- WeChat version: add a one-paragraph platform intro at the top (WeChat doesn't show link previews, so context needs to be self-contained)
- Videos: YouTube embed for GitHub Pages, Bilibili embed for Zhihu/WeChat (upload same mp4 to both platforms)

---

## Out of Scope

- Cross-embodiment experiment (xArm6) — not yet run, mentioned only as future work
- Data efficiency sweep with augmented model — augmentation failed, no need to run
- Any new experiments before writing
