# Version A ETHICS Qwen2.5 Release

This repository is a clean release snapshot of the `Version A` experiment for:

- ETHICS benchmark (`virtue` task)
- `Qwen/Qwen2.5-0.5B-Instruct`
- Training-Free GRPO prefix optimization

The release was prepared on `2026-04-08` from the local experiment workspace and includes the finalized result artifacts, paper-ready tables and figures, the experiment instruction file, the runner used for the strict sweep, and the paper PDF.

## Key Outputs

- Final paper PDF: `paper/version_a_latest.pdf`
- Timestamped paper PDF: `paper/version_a_paper_2026-04-08.pdf`
- Experiment instructions: `version_a_experiment_instructions.md`
- Strict runner: `version_a_strict_runner.py`
- Main result tables: `results/version_a/tables/`
- MDL figure: `results/version_a/figures/mdl_curves.pdf`
- Reproducibility note: `results/version_a/reproducibility_note_2026-04-08.md`
- Integrity audit: `results/version_a/audit_2026-04-08.json`
- Deliverables checklist: `results/version_a/deliverables_checklist.json`

## Repository Layout

- `paper/`
  - LaTeX source and final exported PDFs
- `ethics/`
  - Local copy of the ETHICS benchmark used by the experiment
- `results/version_a/`
  - Full Version A experiment outputs, including per-run predictions and summary tables
- `version_a_experiment_instructions.md`
  - The authoritative instruction file followed for the release run
- `version_a_strict_runner.py`
  - The main experiment runner
- `clean_suffix_candidate_pool_v2.json`
  - Candidate pool artifact kept with the release for completeness

## Reproduction Notes

The original run used the local Qwen model setup on the laptop and the local ETHICS copy bundled here. The result package includes the exact split hashes and final artifacts used for paper insertion.

To rerun the strict experiment from the repository root:

```bash
python3 version_a_strict_runner.py   --results-root results/version_a   --source-virtue-dir ethics/virtue   --seeds 0,1,2   --grpo-editor-mode free_prefix   --sampling-strategy stratified   --reward-metric accuracy
```

## Paper Build Note

The paper source is stored under `paper/0331_version_a.tex`. The finalized PDF is already included, so the release remains usable even if a local TeX installation is incomplete.
