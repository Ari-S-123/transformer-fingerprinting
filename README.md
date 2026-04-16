# transformer-fingerprinting

Contrastive audio fingerprinting experiments built around Jupyter notebooks. The project starts with dataset exploration, moves through baseline training and robustness evaluation, and ends with a Colab-first `fma_medium` scale-up notebook.

**Authors:** Aritra Saharay, Aaron Gordoa, Edwin Yu

## What this project shows

The project was structured as a scientific arc, not a straight-line build:

1. **Initial hypothesis:** a hybrid spectrogram transformer would beat a small CNN baseline at neural audio fingerprinting.
2. **Contradictory evidence:** under a *high-pass* degradation (bass removed, like a phone line), the hybrid transformer collapsed to near-zero Top-1 while frozen MERT held up at ~53%. Adding high-pass filters to training fixed that condition but broke others — augmentation isn't a volume dial, it's selection pressure that reshapes which invariances get learned.
3. **Harder benchmark, different verdict:** Notebook 3 replaced the saturated "clean, centered, 3-second" test with short, off-center, realistic-hard queries (protocol from Nikou & Giannakopoulos 2025). Under this benchmark, the CNN baseline overtook the transformer on the combined ranking score.
4. **Scale-up confirms it:** Notebook 4 moved from `fma_small` (8K tracks) to `fma_medium` (~25K tracks). The CNN winner held, but the worst-case floor dropped 39%. Scale exposed robustness weaknesses the small benchmark masked.

### Final system recipe (at `fma_medium` scale)

- **Model:** `cnn_baseline_embed128` (trained in Notebook 2)
- **Reference windowing:** `multi5_even` (five evenly spaced fingerprints per track instead of one)
- **Retrieval index:** `ivfflat_nprobe8` (approximate FAISS, ~12× faster than exact)
- **Ranking score:** 0.4009
- **Clean Top-1:** 0.8346 · **Mean degraded Top-1:** 0.3642 · **MRR:** 0.5774
- **Worst-condition Top-1:** 0.0145
- **Latency:** 0.0103 ms/query · **Index size:** 6.41 MB

Frozen MERT has a better worst-case floor (0.0591 vs. CNN's 0.0145) and a smaller index, so it's a defensible alternative when robustness-under-worst-case matters more than combined-metric performance.

### Honest limitations

The winning CNN still fails hard on **short, off-center, high-pass-filtered queries** (Top-1 near 1%) and on multi-segment fragmented evidence. Temporal aggregation and hard-negative retraining were scaffolded in Notebook 4 but **disabled in the executed run** (Colab budget / bug). See Progress Report 4 for the honest post-mortem.

## Repository structure

```
.
├── jupyter-notebooks/              # Notebooks 01–04 (scientific arc)
├── progress-reports/               # Reports 1–4 (PDF + final as markdown)
├── outputs/                        # Results, CSVs, plots per notebook
├── papers/                         # The three reference papers (PDF)
├── Transformer-Fingerprinting-Presentation.pptx   # 15-min final talk
├── Transformer-Fingerprinting-Presentation.pdf    # Same, PDF export
├── CS7150 Concept Presentation.pdf                # Early concept pitch
├── LICENSE
└── README.md
```

## Notebook sequence

The notebooks are intended to be read and run in order:

1. `jupyter-notebooks/01_fma_exploration.ipynb`
   Dataset exploration and early sanity checks on FMA metadata and audio layout.
2. `jupyter-notebooks/02_song_fingerprinting_experiments_colab.ipynb`
   Baseline training and retrieval experiments on the smaller benchmark setup (CNN, hybrid transformer, frozen MERT).
3. `jupyter-notebooks/03_robustness_ablation_and_realistic_evaluation.ipynb`
   Robustness ablations, harsher retrieval degradations, multi-window indexing, and historical comparison logic.
4. `jupyter-notebooks/04_fma_medium_scaleup_hard_negatives_and_temporal_aggregation.ipynb`
   Final project-stage notebook. This notebook scales evaluation from `fma_small` to `fma_medium`, re-evaluates historical runs, enables `realistic_hard` queries and `multi5_even` windowing by default, and exports final comparison artifacts. Grouped multi-segment aggregation and hard-negative retraining were scaffolded but left disabled (`run_aggregation_eval=False`, `run_hard_negative_retraining=False`) due to a bug and exhausted Colab budget.

## Notebook 4 scope

Notebook 4 is the final synthesis notebook. It is deliberately self-contained and is derived from Notebook 3 instead of introducing a new shared Python package layer.

The notebook covers:

- Runtime inspection for GPU, RAM, disk, Drive mount status, and write access.
- Historical artifact discovery across Notebook 2, Notebook 3, and partial Notebook 4 outputs.
- `fma_medium` bootstrap, extraction, validation, and undecodable-audio reporting.
- Baseline historical evaluation on the harder retrieval matrix (8 historical runs × 3 windowing strategies × 5 index variants = 120 baseline combinations).
- FAISS index sweep (exact vs. IVFFlat vs. IVFPQ) with quality/latency/memory trade-offs.
- Final ranking tables, failure cases, markdown conclusions, presentation summary, plots, and zipped export bundle.

## Execution environment

The notebooks are designed for Google Colab with a GPU runtime and Google Drive.

That is not optional in practice if you want the full intended run:

- The runtime/output paths are Colab-style paths under `/content/...`.
- The notebook expects Drive-backed caching and artifact reuse.

Notebook 4 was executed on an NVIDIA H100 80GB HBM3 runtime (Python 3.12, PyTorch 2.10+cu128, Transformers 5.0.0, FAISS 1.13.2).

## Main outputs

The notebooks write outputs under:

- `/content/song_fingerprinting_outputs`

The main exported artifacts are expected in the notebook's `exports` directory and include:

(Notebook 4 example)

- `runtime_report.csv`
- `artifact_check_report.csv`
- `dataset_layout_report.csv`
- `undecodable_audio_tracks.csv`
- `smoke_test_report.csv`
- `notebook4_base_eval_long.csv`
- `notebook4_base_eval_summary.csv`
- `notebook4_aggregation_eval_long.csv` (empty — aggregation disabled)
- `notebook4_aggregation_eval_summary.csv` (empty — aggregation disabled)
- `notebook4_hard_negative_eval_long.csv` (empty — hard-negative retraining disabled)
- `notebook4_hard_negative_eval_summary.csv` (empty — hard-negative retraining disabled)
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

## References

The project builds directly on three papers (all included in `papers/`):

1. **Chang, Lee, Park, Lim, Lee, Ko, Han (2021).** "Neural Audio Fingerprint for High-Specific Audio Retrieval Based on Contrastive Learning." *ICASSP 2021* — [arXiv:2010.11910](https://arxiv.org/abs/2010.11910). Source of the contrastive + NT-Xent + in-batch-negatives recipe and the mel-spectrogram configuration used here.
2. **Singh, Bhat, Riley, Resnick, Thickstun, De Brouwer (2025).** "Robust Neural Audio Fingerprinting using Music Foundation Models." [arXiv:2511.05399](https://arxiv.org/abs/2511.05399). Motivation for the frozen MERT baseline.
3. **Nikou & Giannakopoulos (2025).** "Contrastive and Transfer Learning for Effective Audio Fingerprinting through a Real-World Evaluation Protocol." *IJMSTA* 7(1): 68–82 — [arXiv:2507.06070](https://arxiv.org/abs/2507.06070). Motivation for the harder, realistic-query benchmark in Notebook 3.

Dataset: **Defferrard, Benzi, Vandergheynst, Bresson (2017).** "FMA: A Dataset For Music Analysis." *ISMIR 2017* — [arXiv:1612.01840](https://arxiv.org/abs/1612.01840).
