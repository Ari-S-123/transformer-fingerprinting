# Progress Report 3: Post-Notebook-3 Robustness Ablation and Realistic Evaluation

**Project:** Transformer-Based Music Fingerprinting

**Date:** April 3, 2026

**Authors:** Aritra Saharay, Aaron Gordoa, Edwin Yu

**Repository:** <https://github.com/Ari-S-123/transformer-fingerprinting>

## 1. Executive Summary

This report consolidates what was completed and what was learned after the third notebook (`03_robustness_ablation_and_realistic_evaluation.ipynb`) finished running. The core purpose of Notebook 3 was not to build a new model family from scratch. It was to stress-test the retrieval pipeline built earlier, make the benchmark more realistic, determine whether the hybrid spectrogram transformer could be made robust in a balanced way, and produce an explicit recommendation for the next notebook.

The most important outcome is that Notebook 3 changed the model recommendation. Notebook 2 had concluded that the strongest overall model was the **baseline hybrid transformer with 128-dimensional embeddings**, while also exposing its catastrophic weakness on high-pass filtering. Notebook 3 revisited that conclusion under a much harder benchmark and a broader evaluation grid. The final exported conclusion instead recommends **`cnn_baseline_embed128`** as the best configuration, with **mean degraded Top-1 = 0.4817** and **worst-condition Top-1 = 0.0239**.

That result matters because it means the earlier story, “the hybrid transformer is probably the long-term winner once we harden the benchmark”, did **not** survive more realistic evaluation.

## 2. Project Context Before Notebook 3

The project had already progressed through two earlier stages:

1. **Progress Report 1 / Notebook 1 stage:** dataset selection, FMA-based preprocessing, spectrogram generation, augmentation primitives, and contrastive-pair dataset construction.
2. **Progress Report 2 / Notebook 2 stage:** end-to-end training and retrieval evaluation for three model families, CNN, hybrid spectrogram transformer, and frozen MERT, using FAISS-based nearest-neighbor retrieval.

At the end of Notebook 2, the baseline hybrid transformer looked strongest overall on degraded retrieval, but it had an obvious blind spot: its high-pass retrieval performance essentially collapsed. The extended transformer improved high-pass robustness, but only by giving up too much on noise, pitch, and time-stretch robustness. That directly motivated Notebook 3.

## 3. What Was Done in Notebook 3

Notebook 3 substantially expanded the experimental system rather than just re-running Notebook 2 with extra plots.

### 3.1 Reliability and artifact reuse

The notebook first verified that the prior Notebook 2 artifacts were usable. It checked for the existence of the expected checkpoints, metrics, configs, and degradation-breakdown files for five previously trained runs. All five artifact groups were present, which allowed Notebook 3 to treat Notebook 2 results as stable control baselines instead of retraining everything from zero.

It also ran model smoke tests across all three model families:

- CNN (spectrogram input): loss 0.957792, embedding shape (4, 128)
- Hybrid transformer (spectrogram input): loss 1.805847, embedding shape (4, 128)
- Frozen MERT (waveform input): loss 1.523258, embedding shape (8, 128)

These checks confirmed that the data path, forward pass, contrastive loss, and embedding dimensionality were all functioning before the full experiment grid was executed.

### 3.2 Benchmark upgrades

Notebook 3 upgraded the evaluation protocol in four important ways.

**First, it introduced policy-driven augmentation instead of only keeping the old `baseline` and `extended` presets.**
Three new hybrid-transformer training variants were executed:

- `hybrid_transformer_one_of_k_embed128`
- `hybrid_transformer_severity_controlled_embed128`
- `hybrid_transformer_filters_only_embed128`

These were designed to test whether a more selective augmentation policy could recover balanced robustness without repeating the failure mode of the original “stack everything” extended setup.

**Second, it made the query benchmark harder.**
Notebook 2 had already shown that clean centered 3-second retrieval was almost saturated. Notebook 3 therefore added:

- short centered queries at 1 s, 2 s, and 3 s
- short off-center queries at 1 s, 2 s, and 3 s
- combined moderate degradations
- multi-segment same-track queries

This is closer to what an actual fingerprinting system would face, because real queries are often short, misaligned, and degraded.

**Third, it added multi-window reference indexing.**
Instead of always indexing one centered reference segment per track, Notebook 3 compared:

- `single_center`
- `multi3_even` (three evenly spaced reference windows per track)

This made it possible to test whether indexing more of each song was worth the added memory cost.

**Fourth, it ran a FAISS sweep rather than treating search as fixed.**
The exported results include **400 FAISS sweep rows**, covering exact inner-product search and approximate variants such as IVF Flat and IVFPQ. That matters because in a real system, retrieval quality is only part of the problem; memory footprint and latency matter too.

## 4. Exported Outputs

Notebook 3 completed the full export pipeline and produced:

- `final_metrics_long.csv` (3920 rows)
- `final_metrics_summary.csv` (80 rows)
- `faiss_sweep_results.csv` (400 rows)
- `failure_cases.csv` (200 rows)
- `artifact_check_report.csv`
- `model_smoke_test_report.csv`
- `notebook3_conclusions.md`

## 5. Key Results

### 5.1 Best overall configuration

The winning configuration exported by Notebook 3 was:

- **Run:** `cnn_baseline_embed128`
- **Source:** Notebook 2 control artifact
- **Reference windowing:** `multi3_even`
- **Index:** `exact_ip`
- **Ranking score:** 0.4749
- **Mean degraded Top-1:** 0.4817
- **Worst-condition Top-1:** 0.0239
- **Combined moderate Top-1:** 0.2452
- **Latency:** 0.1062 ms/query
- **Index size:** 1.1719 MB

### 5.2 Best result by major configuration family

| Configuration | Best source/setup | Mean degraded Top-1 | Worst-condition Top-1 | Main interpretation |
|---|---|---:|---:|---|
| `cnn_baseline_embed128` | Notebook 2, `multi3_even`, `exact_ip` | 0.4817 | 0.0239 | Best overall practical choice under the harder benchmark. |
| `hybrid_transformer_baseline_embed128` | Notebook 2, `multi3_even`, `exact_ip` | 0.3468 | 0.0066 | Strong average degraded retrieval, but catastrophic floor remains. |
| `hybrid_transformer_one_of_k_embed128` | Notebook 3, `single_center`, `exact_ip` | 0.2995 | 0.1045 | Best new selective-policy transformer; much better floor, weaker overall average. |
| `hybrid_transformer_severity_controlled_embed128` | Notebook 3, `single_center`, `exact_ip` | 0.2880 | 0.1080 | Similar story: better balance than old hybrid, but not enough to win. |
| `frozen_mert_extended_embed128` | Notebook 2, `single_center`, `ivfflat_nprobe8` | 0.3006 | 0.1350 | Middle-of-the-pack baseline; robust floor, but no clear dominance. |
| `hybrid_transformer_extended_embed128` | Notebook 2, `single_center`, `exact_ip` | 0.1512 | 0.0468 | Confirms that the old heavy augmentation policy over-corrected in the wrong way. |

## 6. What Was Learned

### 6.1 Harder evaluation changed the project conclusion

This is the biggest lesson. Under the easier Notebook 2 framing, the baseline hybrid transformer looked like the strongest overall architecture. Under Notebook 3’s harder evaluation, the project recommendation flipped to the CNN. That means the project was previously overestimating how much the hybrid transformer’s apparent strength would survive realistic evaluation.

### 6.2 Selective augmentation helped the transformer’s floor, but did not make it the winner

Compared with the baseline hybrid transformer in the single-center exact-search setting, the new selective policies materially improved worst-case robustness:

- baseline hybrid worst-condition Top-1: **0.0062**
- `one_of_k` worst-condition Top-1: **0.1045**
- `severity_controlled` worst-condition Top-1: **0.1080**

That is a real improvement. However, the selective policies still did not overtake the CNN overall, and they still paid noticeable costs in the average degraded metric.

In practical terms, selective augmentation helped the hybrid transformer become **less brittle**, but not **best**.

### 6.3 The old “extended” augmentation idea was still too blunt

Notebook 2 had already suggested that adding low-pass and high-pass filtering into the training pipeline changed the learned invariances rather than making the model uniformly stronger. Notebook 3 reinforced that lesson. The 128-dim and 256-dim extended hybrid models remained among the weaker overall configurations, despite fixing some specific failure modes.

This means the project should stop treating “more augmentation” as a generic solution. The issue is not augmentation quantity; it is augmentation **selection pressure**.

### 6.4 Multi-window indexing is useful, but not universally

For the winning CNN configuration, `multi3_even` clearly helped:

- clean Top-1 improved from **0.7740** to **0.8646**
- mean degraded Top-1 improved from **0.4471** to **0.4817**
- combined moderate Top-1 improved from **0.1940** to **0.2453**

However, the same improvement was not universal across every transformer variant. Multi-window indexing therefore looks worthwhile as a **system-level retrieval enhancement**, but not as a substitute for better learned representations.

### 6.5 Approximate FAISS remained viable

The FAISS sweep supports using approximate search when needed. On average across comparable run/window pairs, `ivfflat_nprobe8` changed ranking score by only **0.0002**, changed mean degraded Top-1 by **-0.0016**, and reduced latency by about **0.0216 ms/query**, at the cost of a small index-size increase.

The practical reading is straightforward: exact search is still fine at this project scale, but approximate FAISS is already good enough that it should not be a blocker if the index grows later.

### 6.6 Some failure modes remain unresolved even for the winner

The winning CNN still has a **very poor worst-case condition**: worst-condition Top-1 is only **0.0239**. The exported failure-case file is dominated by mistakes on **1-second multi-segment same-track retrieval**, which shows that the system is still vulnerable when the evidence is short and fragmented.

So the project is not “solved.” Notebook 3 simply clarified which direction is currently the least wrong.

## 7. Hypothesis Review

| Hypothesis | Verdict | Evidence |
|---|---|---|
| The hybrid transformer would remain the strongest practical choice once the benchmark became harder. | Not supported. | The exported Notebook 3 conclusion recommends cnn_baseline_embed128 rather than any hybrid transformer. The best hybrid result was hybrid_transformer_baseline_embed128 at ranking_score=0.3777, well behind the CNN at 0.4749. |
| Selective augmentation would recover more balanced robustness than the old extended policy. | Partially supported. | Compared with the baseline hybrid single-center exact setup, one_of_k lifted worst-condition Top-1 from 0.0062 to 0.1045, and severity_controlled lifted it to 0.1080. But neither surpassed the CNN overall, and both reduced mean degraded Top-1 relative to the baseline hybrid’s multi-window best case. |
| Multi-window indexing would help enough to justify the extra complexity. | Conditionally supported. | For the CNN, moving from single_center to multi3_even exact search raised clean Top-1 from 0.7740 to 0.8646 and mean degraded Top-1 from 0.4471 to 0.4817. The gain was not universal across all transformer variants, but it clearly helped the winning configuration. |
| Approximate FAISS could remain usable if quality loss stayed small. | Supported. | Across run/window pairs, ivfflat_nprobe8 changed ranking_score by only 0.0002 on average and mean degraded Top-1 by -0.0016, while reducing latency by about 0.0216 ms/query on average. |

## 8. Recommended Direction for Notebook 4

Notebook 4 should **not** default to more architecture churn. Notebook 3 already showed that benchmark quality and augmentation design can overturn architecture-level conclusions.

Our best next steps are:

1. Finally make the move from the smaller `fma_small` subset to the full `fma_full` dataset.

2. **Continue from the exported best configuration (`cnn_baseline_embed128`).**
   That is the recommendation explicitly encoded by Notebook 3.

3. **Focus on query-aware robustness rather than broad augmentation stacking.**
   The remaining weak points are short, off-center, and fragmented retrieval conditions.

4. **Target the floor, not just the mean.**
   Worst-condition performance is still extremely low for the winner, so Notebook 4 should optimize for robustness floor as a first-class objective.

5. **Introduce harder negatives and better aggregation.**
   Notebook 3 exposed the weakness of short multi-segment evidence. That points toward hard negative mining, multi-segment aggregation logic, or both.

6. **Only revisit transformer fine-tuning if it is tightly scoped.**
   The new augmentation-policy runs proved that the transformer can be made less brittle, but not yet superior. A future transformer revisit should therefore be selective and hypothesis-driven, not broad.

## 9. Bottom Line

Notebook 3 was successful because it changed the project from “we have a working retrieval model” to “we now know which parts of the earlier conclusion were fragile.”

The third notebook accomplished five things:

- It validated and reused the Notebook 2 artifacts.
- It turned the benchmark into a more realistic retrieval testbed.
- It tested multiple new augmentation-policy variants.
- It measured the retrieval-quality/latency/memory tradeoff through FAISS sweeps.
- It produced a decisive recommendation for what to do next.

The main scientific lesson is that **robust music fingerprinting is being determined less by raw model family and more by the invariances induced by the training and evaluation design**. Once the benchmark became harder, the project’s preferred model changed.

## 10. Source Materials

We used the following project artifacts to write this report:

- `03_robustness_ablation_and_realistic_evaluation.ipynb`
- `final_metrics_summary.csv`
- `final_metrics_long.csv`
- `faiss_sweep_results.csv`
- `failure_cases.csv`
- `artifact_check_report.csv`
- `model_smoke_test_report.csv`
- `notebook3_conclusions.md`
- `Progress Report 1.pdf`
- `Progress Report 2.pdf`
