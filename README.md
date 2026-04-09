# Task Framing Matters More Than Prefix Tuning on ETHICS Virtue

[![Paper PDF](https://img.shields.io/badge/paper-latest%20PDF-bd561d)](https://github.com/hanzhenzhujene/stupidity-is-a-moral-failure/releases/download/v1.0.0/version_a_latest.pdf)
[![Release](https://img.shields.io/github/v/release/hanzhenzhujene/stupidity-is-a-moral-failure?label=release)](https://github.com/hanzhenzhujene/stupidity-is-a-moral-failure/releases/tag/v1.0.0)
![Benchmark](https://img.shields.io/badge/benchmark-ETHICS%20Virtue-1f6feb)
![Model](https://img.shields.io/badge/model-Qwen2.5--0.5B--Instruct-198754)

**Takeaway.** On frozen `Qwen/Qwen2.5-0.5B-Instruct`, training-free GRPO prefix search improves ETHICS `virtue`, but instruction-only reframing improves it much more: **20.20% -> 37.46% -> 62.28%**.

**TL;DR**
- Single-task diagnostic release: ETHICS `virtue`, frozen `Qwen/Qwen2.5-0.5B-Instruct`, training-free GRPO prefix search versus instruction-only paraphrase.
- The best GRPO prefix improves test accuracy from **20.20%** to **37.46%**.
- A budget-matched instruction-only paraphrase reaches **62.28%**, so **task framing matters more than prefix tuning in this setup**.

**Paper:** [Latest PDF](https://github.com/hanzhenzhujene/stupidity-is-a-moral-failure/releases/download/v1.0.0/version_a_latest.pdf) | [Timestamped PDF](https://github.com/hanzhenzhujene/stupidity-is-a-moral-failure/releases/download/v1.0.0/version_a_paper_2026-04-08.pdf) | [GitHub Release](https://github.com/hanzhenzhujene/stupidity-is-a-moral-failure/releases/tag/v1.0.0)

![Main result summary](assets/repo_overview.svg)

**Quick access:** [Main results table](results/version_a/tables/table2_main_results.csv) | [Hard-test table](results/version_a/tables/table3_hard_test_calibration.csv) | [MDL figure](results/version_a/figures/mdl_curves.pdf) | [Reproducibility note](results/version_a/reproducibility_note_2026-04-08.md)

## What This Repository Shows

This repository is a clean public release of one deliberately diagnostic experiment: on ETHICS `virtue`, does performance improve more from optimizing a prefix or from simply asking the model the task in a better way?

In this setup, both help, but not equally. Prefix search helps materially. Instruction framing helps much more.

## Why It Matters

- Benchmark failure can reflect a task-interface problem before it reflects a capability limit.
- Neutral length-matched text does not help here, so the effect is not explained by generic fluent prefixing.
- MDL-style diagnostics still add value: lower development code length tracks better held-out performance in this release.

## Main Result

| Condition | Test accuracy | 95% CI |
| --- | ---: | ---: |
| Baseline (empty prefix) | 20.20% | 19.10% - 21.31% |
| Neutral prefix (best `L`) | 20.18% | 19.07% - 21.31% |
| Best GRPO prefix | 37.46% | 36.78% - 38.13% |
| Best instruction-only paraphrase | 62.28% | 61.56% - 63.01% |

The core comparison is not just baseline versus GRPO. It is **best GRPO prefix (`37.46%`) versus best instruction-only paraphrase (`62.28%`)**. The paraphrase condition exceeds the best GRPO result by **24.82 percentage points**.

A few supporting diagnostics point the same way:

- Best GRPO improves over baseline by **17.26 percentage points** on test accuracy.
- Baseline hard-test accuracy is **20.23%**; best GRPO hard-test accuracy is **37.50%**.
- ECE improves from **0.4975** to **0.3741**.
- Spearman correlation between `Delta L_dev(L)` and test accuracy is **0.725**, with 95% CI **[0.456, 0.866]**.

## One Concrete Intuition

Same model, same frozen weights, same benchmark.

If you prepend coherent non-moral text, performance stays at baseline: **20.18%** versus **20.20%**. If you rewrite only the task instruction, performance jumps to **62.28%**. That is the clearest reason the headline of this repository is about framing rather than generic prefix fluency.

Representative raw strings in [results/version_a/prefix_examples_documentation.md](results/version_a/prefix_examples_documentation.md) tell the same story: the strongest GRPO and paraphrase seeds are often oddly framed or prompt-corrupted, reinforcing how interface-sensitive the model is in this setting.

## What Is Included

| Component | What it contains | Key files |
| --- | --- | --- |
| Paper | Paper source and released PDFs | [paper/0331_version_a.tex](paper/0331_version_a.tex), [paper/version_a_latest.pdf](paper/version_a_latest.pdf), [paper/version_a_paper_2026-04-08.pdf](paper/version_a_paper_2026-04-08.pdf) |
| Experiment specification | Exact instruction file, runner, and candidate pool | [version_a_experiment_instructions.md](version_a_experiment_instructions.md), [version_a_strict_runner.py](version_a_strict_runner.py), [clean_suffix_candidate_pool_v2.json](clean_suffix_candidate_pool_v2.json) |
| Benchmark copy | Local ETHICS files used for the run | [ethics/](ethics/) |
| Results package | Tables, figures, audits, and full run outputs | [results/version_a/](results/version_a/), [results/version_a/tables/](results/version_a/tables/), [results/version_a/figures/mdl_curves.pdf](results/version_a/figures/mdl_curves.pdf), [results/version_a/reproducibility_note_2026-04-08.md](results/version_a/reproducibility_note_2026-04-08.md), [results/version_a/audit_2026-04-08.json](results/version_a/audit_2026-04-08.json), [results/version_a/deliverables_checklist.json](results/version_a/deliverables_checklist.json) |

## Reproducibility Assets

This release includes:

- all **27 GRPO runs**
- all **9 instruction-paraphrase runs**
- split hashes for `Train_opt` and `Train_dev`
- per-run predictions
- paper-ready tables and figures
- a passing deliverables checklist

The packaged outputs were also manually audited after release assembly to confirm that the run directories and summary artifacts remained complete.

## How to Reproduce

The original run used a local Qwen model setup and the bundled ETHICS copy in this repository. The exact split hashes used for the paper are recorded in [results/version_a/split_hashes.json](results/version_a/split_hashes.json).

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

Python dependencies used by the runner are listed in [requirements.txt](requirements.txt).

## Scope

This repository captures a **single-task, single-model** result: ETHICS `virtue` on `Qwen2.5-0.5B-Instruct`. The strongest conclusion is therefore diagnostic rather than universal. But that conclusion is already meaningful: in this setup, **benchmark framing matters more than prefix tuning**, and that should change how similar evaluations are designed and interpreted.

## License

Original code and repository-authored documentation in this repository are released under the [MIT License](LICENSE).

Third-party materials are not relicensed by that file. In particular, the bundled ETHICS benchmark materials under [ethics/](ethics/) retain their own upstream license and attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [ethics/LICENSE](ethics/LICENSE).
