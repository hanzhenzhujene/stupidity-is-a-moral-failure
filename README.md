# Version A: ETHICS Virtue, Qwen2.5, and the Limits of Prefix Optimization

This repository packages a clean, reproducible release of a single but very informative experiment: **can training-free GRPO prefix search meaningfully improve moral classification on the ETHICS virtue benchmark, or is the real bottleneck the way the task is framed?**

The answer in this release is sharp: **prompt framing matters far more than prefix tuning**. On `Qwen/Qwen2.5-0.5B-Instruct`, the best GRPO prefix improves test accuracy from **20.20%** to **37.46%**, but a budget-matched instruction paraphrase reaches **62.28%**. In other words, the biggest gains do not come from adding more optimized prefix tokens; they come from telling the model the task in a way that matches what the benchmark is actually asking.

## Why This Repo Matters

If you are working on alignment, evaluation, prompt optimization, or benchmark design, this release gives you a concrete lesson:

- **Do not treat prompt optimization gains as evidence of deeper moral competence by default.**
- **Check task framing before claiming model weakness or success.**
- **Use controls.** Neutral text, empty-prefix baselines, and instruction-only rewrites change the interpretation of the result.
- **MDL-style diagnostics can still be useful.** In this run, lower development code length tracks better test performance with a Spearman correlation of **0.725**.

This makes the repository useful not just as a result archive, but as a worked example of how to separate:

- genuine prefix effects,
- framing effects,
- and benchmark artifacts.

## Core Findings

### Main quantitative result

| Condition | Test Accuracy | 95% CI |
| --- | ---: | ---: |
| Baseline (empty prefix) | 20.20% | 19.10% - 21.31% |
| Neutral prefix (best L) | 20.18% | 19.07% - 21.31% |
| Instruction-only paraphrase (best K->k) | 62.28% | 61.56% - 63.01% |
| Best GRPO prefix | 37.46% | 36.78% - 38.13% |

### Hard-test and calibration

| Condition | Hard Accuracy | ECE |
| --- | ---: | ---: |
| Baseline | 20.23% | 0.4975 |
| Best GRPO prefix | 37.50% | 0.3741 |

### Interpretation

The experiment supports three strong takeaways:

1. **Neutral text does nothing.** The neutral-prefix control stays at baseline, which means gains are not explained by simply prepending fluent language.
2. **GRPO prefix search helps, but not the most.** Prefix optimization does improve both test accuracy and hard-test accuracy, and it materially improves calibration.
3. **Instruction framing dominates.** The best budget-matched paraphrase dramatically outperforms the best learned prefix, which means the benchmark interface is a larger lever than token-level prefix search in this setting.

## Deep Takeaway

The most important lesson is not just that one method scored higher than another. It is that **apparent benchmark failure can be a presentation problem before it is a capability problem**.

A reader should come away with this practical conclusion: if a model looks bad on a moral classification benchmark, you should first ask whether the prompt tells the model the right job. Only after that should you interpret failures as evidence of missing moral knowledge or as a target for more elaborate optimization.

This is why the controls in this repository matter. They turn the experiment from a simple leaderboard comparison into a diagnostic about where performance is really coming from.

## What Is Included

### Paper and release assets

- Final paper PDF: [`paper/version_a_latest.pdf`](paper/version_a_latest.pdf)
- Timestamped paper PDF: [`paper/version_a_paper_2026-04-08.pdf`](paper/version_a_paper_2026-04-08.pdf)
- Paper source: [`paper/0331_version_a.tex`](paper/0331_version_a.tex)

### Experiment specification and code

- Authoritative instructions: [`version_a_experiment_instructions.md`](version_a_experiment_instructions.md)
- Strict runner: [`version_a_strict_runner.py`](version_a_strict_runner.py)
- Candidate pool artifact: [`clean_suffix_candidate_pool_v2.json`](clean_suffix_candidate_pool_v2.json)

### Data and outputs

- ETHICS local copy used for the run: [`ethics/`](ethics/)
- Full Version A outputs: [`results/version_a/`](results/version_a/)
- Main tables: [`results/version_a/tables/`](results/version_a/tables/)
- MDL figure: [`results/version_a/figures/mdl_curves.pdf`](results/version_a/figures/mdl_curves.pdf)
- Reproducibility note: [`results/version_a/reproducibility_note_2026-04-08.md`](results/version_a/reproducibility_note_2026-04-08.md)
- Integrity audit: [`results/version_a/audit_2026-04-08.json`](results/version_a/audit_2026-04-08.json)
- Deliverables checklist: [`results/version_a/deliverables_checklist.json`](results/version_a/deliverables_checklist.json)

## Repository Structure

```text
paper/                     Paper source and final PDFs
ethics/                    ETHICS benchmark files used locally
results/version_a/         Full experiment outputs
version_a_experiment_instructions.md
version_a_strict_runner.py
clean_suffix_candidate_pool_v2.json
```

## Reproducing the Experiment

The original run used a local Qwen model setup on the laptop and the local ETHICS copy bundled here. The result package includes the exact split hashes used for the paper.

Run from the repository root:

```bash
python3 version_a_strict_runner.py \
  --results-root results/version_a \
  --source-virtue-dir ethics/virtue \
  --seeds 0,1,2 \
  --grpo-editor-mode free_prefix \
  --sampling-strategy stratified \
  --reward-metric accuracy
```

Python dependencies used by the runner are listed in [`requirements.txt`](requirements.txt).

## Validation and Reproducibility

This release includes:

- complete coverage of the **27 GRPO runs**,
- complete coverage of the **9 instruction-paraphrase runs**,
- split hashes for the `Train_opt` and `Train_dev` partition,
- saved per-run predictions,
- paper-ready summary tables and figure files,
- and a passing deliverables checklist.

The release was also manually audited after packaging to ensure the summary outputs and run directories remained complete.

## Caveat

This repository captures a **single-task, single-model** result: ETHICS `virtue` on `Qwen2.5-0.5B-Instruct`. The strongest conclusion is therefore diagnostic rather than universal: **in this setup, benchmark framing is the dominant lever**. That is already a meaningful result, and one that should influence how similar evaluations are designed and interpreted.
