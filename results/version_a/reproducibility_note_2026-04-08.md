# Version A Reproducibility Note (2026-04-08)

## Instruction Source
- `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/version_a_experiment_instructions.md`

## Dataset and Model Anchors
- ETHICS local dataset root:
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/ethics`
- Qwen local Ollama manifest (requested anchor):
  - `/Users/hanzhenzhu/.ollama/models/manifests/registry.ollama.ai/library/qwen2.5/0.5b-instruct`
- Qwen HF cache (local-only runtime source used by strict runner):
  - `/Users/hanzhenzhu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`

## Fixed Split Hashes (seed=42 permutation protocol)
- `train_opt_sha256`: `788571319104f862bc5ff0919dec13b8609a15aa86c951e5e90d6d05f97d06e8`
- `train_dev_sha256`: `d973ca71779c2989a1928b3ea9564bebed7b05d11a27b30c530b15bcd593d00f`
- Sizes logged in `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/split_hashes.json`

## Required Run Coverage
- GRPO runs present: 27 / 27
- Paraphrase runs present: 9 / 9
- Deliverables validation status: PASS
- Validation file:
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/deliverables_checklist.json`
- Integrity audit file:
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/audit_2026-04-08.json`

## Paper-Ready Output Files
- Table 2:
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/tables/table2_main_results.csv`
- Table 3:
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/tables/table3_hard_test_calibration.csv`
- Full sweep appendix table:
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/tables/full_hyperparameter_sweep.csv`
- Spearman table (rho + 95% CI):
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/tables/spearman_table.csv`
- Figure 2 (MDL curves):
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/figures/mdl_curves.pdf`
  - `/Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a/figures/mdl_curves.png`

## Notes
- The strict runner verification was refreshed on 2026-04-08.
- Spearman CI bounds were explicitly recomputed with 10,000 bootstrap resamples over all 27 `(L, K->k, seed)` tuples and written to the spearman table.

## Reproduction Command
- `/opt/anaconda3/bin/python3.12 /Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/version_a_strict_runner.py --results-root /Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/results/version_a --source-virtue-dir /Users/hanzhenzhu/Desktop/CEI_Research/experimental design/0329VersionA_run_codex/ethics/virtue --seeds 0,1,2 --grpo-editor-mode free_prefix --sampling-strategy stratified --reward-metric accuracy`
