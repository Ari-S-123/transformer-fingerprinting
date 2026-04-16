# transformer-fingerprinting

Contrastive audio fingerprinting experiments built around Jupyter notebooks. The project starts with dataset exploration, moves through baseline training and robustness evaluation, and ends with a Colab-first `fma_medium` scale-up notebook.

## Notebook sequence

The notebooks are intended to be read and run in order:

1. `jupyter-notebooks/01_fma_exploration.ipynb`
   Dataset exploration and early sanity checks on FMA metadata and audio layout.
2. `jupyter-notebooks/02_song_fingerprinting_experiments_colab.ipynb`
   Baseline training and retrieval experiments on the smaller benchmark setup.
3. `jupyter-notebooks/03_robustness_ablation_and_realistic_evaluation.ipynb`
   Robustness ablations, harsher retrieval degradations, multi-window indexing, and historical comparison logic.
4. `jupyter-notebooks/04_fma_medium_scaleup_hard_negatives_and_temporal_aggregation.ipynb`
   Final project-stage notebook. This notebook scales evaluation from `fma_small` to `fma_medium`, re-evaluates historical runs, enables `realistic_hard` and `multi5_even` by default, adds but does not use grouped multi-segment aggregation or mine hard negatives (we had a bug and could not rerun the whole notebook because of time and resource limits), retrains a targeted subset of models, and exports final comparison artifacts.

## Notebook 4 scope

Notebook 4 is the final synthesis notebook. It is deliberately self-contained and is derived from Notebook 3 instead of introducing a new shared Python package layer.

The notebook covers:

- Runtime inspection for GPU, RAM, disk, Drive mount status, and write access.
- Historical artifact discovery across Notebook 2, Notebook 3, and partial Notebook 4 outputs.
- `fma_medium` bootstrap, extraction, validation, and undecodable-audio reporting.
- Baseline historical evaluation on the harder retrieval matrix.
- Final ranking tables, failure cases, markdown conclusions, presentation summary, plots, and zipped export bundle.

## Execution environment

The notebooks are designed for Google Colab with a GPU runtime and Google Drive.

That is not optional in practice if you want the full intended run:

- The runtime/output paths are Colab-style paths under `/content/...`.
- The notebook expects Drive-backed caching and artifact reuse.

## Main outputs

The notebooks write outputs under:

- `/content/song_fingerprinting_outputs`

The main exported artifacts are expected in the notebook's `exports` directory and include:

(notebook 4 example)
- `runtime_report.csv`
- `artifact_check_report.csv`
- `dataset_layout_report.csv`
- `undecodable_audio_tracks.csv`
- `smoke_test_report.csv`
- `notebook4_base_eval_long.csv`
- `notebook4_base_eval_summary.csv`
- `notebook4_aggregation_eval_long.csv` (not used)
- `notebook4_aggregation_eval_summary.csv` (not used)
- `notebook4_hard_negative_eval_long.csv` (not used)
- `notebook4_hard_negative_eval_summary.csv` (not used)
- `notebook4_cross_model_comparison.csv`
- `notebook4_failure_cases.csv`
- `faiss_sweep_results.csv`
- `notebook4_runtime_info.json`
- `notebook4_config.json`
- `notebook4_execution_manifest.json`
- `notebook4_conclusions.md`
- `notebook4_presentation_summary.md`
- `notebook4_outputs.zip`

Plots are exported separately to the notebook's `plots` directory.
