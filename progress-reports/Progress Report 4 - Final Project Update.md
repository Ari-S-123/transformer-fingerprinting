# Progress Report 4: Final `fma_medium` Scale-Up, Historical Re-Evaluation, and Final System Recommendation

**Project:** Transformer-Based Music Fingerprinting

**Date:** April 16, 2026

**Authors:** Aritra Saharay, Aaron Gordoa, Edwin Yu

**Repository:** <https://github.com/Ari-S-123/transformer-fingerprinting>

## 1. Executive Summary

This report consolidates what was completed and what was learned from the fourth and final project notebook, `04_fma_medium_scaleup_hard_negatives_and_temporal_aggregation.ipynb`. Notebook 4 was intended to serve as the final synthesis stage of the project: scale the retrieval benchmark from `fma_small` to `fma_medium`, re-evaluate all historically trained checkpoints under the harder benchmark, test system-level upgrades such as richer reference windowing and approximate FAISS search, and, if time and runtime budget permitted, evaluate grouped temporal aggregation and targeted hard-negative retraining.

The most important high-level result is that the **winner from Notebook 3 remained the winner in Notebook 4**, but the reasons for recommending it became more nuanced. The final best baseline configuration at `fma_medium` scale was:

- **Run:** `cnn_baseline_embed128`
- **Source:** Notebook 2 historical artifact
- **Reference windowing:** `multi5_even`
- **Index:** `ivfflat_nprobe8`
- **Ranking score:** **0.4009**
- **Mean degraded Top-1:** **0.3642**
- **Worst-condition Top-1:** **0.0145**
- **MRR:** **0.5774**
- **Latency:** **0.0103 ms/query**
- **Index size:** **6.4093 MB**

That is the final project recommendation. However, Notebook 4 also made two things clear:

1. **Scaling from `fma_small` to `fma_medium` made the retrieval floor worse even though the winner stayed the same.**
2. **The final artifact set is primarily a baseline historical re-evaluation at larger scale, not a full aggregation-and-hard-negative study**, because aggregation and hard-negative retraining were disabled in the actual executed configuration and no aggregation or hard-negative evaluation rows were exported.

This means our final conclusion is not “Notebook 4 proved every planned upgrade.” The honest conclusion is more specific:

- the **best overall practical system** is still the CNN baseline,
- the **best system-level retrieval setup** changed from Notebook 3’s `multi3_even` + `exact_ip` to Notebook 4’s `multi5_even` + `ivfflat_nprobe8`,
- the **robustness floor remains weak** for the overall winner,
- and the **best floor-oriented baseline** is actually the frozen MERT run, not the CNN.

## 2. Project Context Before Notebook 4

Before Notebook 4, the project had already progressed through three distinct stages:

1. **Notebook 1:** early exploration of the FMA dataset, preprocessing design, spectrogram generation, and contrastive training-pair construction.
2. **Notebook 2:** baseline model training and retrieval evaluation across three model families:
   - CNN
   - hybrid spectrogram transformer
   - frozen MERT
3. **Notebook 3:** robustness ablation and realistic evaluation, which:
   - reused Notebook 2 artifacts,
   - added harsher query regimes,
   - introduced multi-window indexing,
   - tested more selective augmentation policies,
   - and changed the project recommendation from the hybrid transformer to the CNN baseline.

Notebook 3’s final recommendation was `cnn_baseline_embed128` with `multi3_even` + `exact_ip`, with:

- ranking score = **0.4749**
- mean degraded Top-1 = **0.4817**
- worst-condition Top-1 = **0.0239**

Notebook 4 was therefore not starting from a blank slate. It was designed to answer whether that recommendation still held once the benchmark was scaled up and the final system-level interventions were introduced.

## 3. What Notebook 4 Was Intended To Answer

The notebook’s own plan framed four core scientific questions:

1. Does the Notebook 3 ranking survive when the retrieval benchmark scales from `fma_small` to `fma_medium`?
2. Which historical models collapse most sharply on worst-case conditions once the database grows?
3. Does multi-segment aggregation recover the floor more efficiently than retraining?
4. Does hard-negative mining improve discriminability without wiping out average retrieval quality?

Those questions matter because they separate three different kinds of progress:

- **benchmark realism** (`fma_small` -> `fma_medium`, harder query regimes),
- **inference-time system design** (multi-window references, approximate search, aggregation),
- **training-time representation improvement** (hard-negative retraining).

The attached artifacts show that Notebook 4 fully answered the first two questions, partially answered the system-design question through baseline windowing/index sweeps, and did **not** fully answer the aggregation and hard-negative questions in executed results.

## 4. What We Executed

Notebook 4’s attached configuration and export manifest show the following executed defaults:

- `execution_mode = "standard"`
- `run_baseline_historical_eval = True`
- `run_aggregation_eval = False`
- `run_hard_negative_retraining = False`
- `run_failure_analysis = True`
- `run_plot_exports = True`
- `run_zip_export = True`
- `resume_from_partial_artifacts = True`
- `skip_existing_exports = True`
- active query regimes included:
  - `clean_current`
  - `short_centered`
  - `short_offcenter`
  - `combined_moderate`
  - `multi_segment_same_track`
  - `realistic_hard`
- active windowing included:
  - `single_center`
  - `multi3_even`
  - `multi5_even`

That execution choice is critical to interpreting the results. The notebook **contains code paths and helper logic** for temporal aggregation and hard-negative retraining, but what we actually did was just a **historical baseline re-evaluation**.

## 5. Execution Environment, Scale-Up, and Data Validation

Notebook 4 ran in a Colab-style GPU environment with the following recorded runtime stack:

- Python **3.12.13**
- PyTorch **2.10.0+cu128**
- Torchaudio **2.10.0+cu128**
- Transformers **5.0.0**
- FAISS **1.13.2**
- CUDA available: **True**
- Device: **NVIDIA H100 80GB HBM3**
- BF16 supported: **True**

The scale-up target was the **`fma_medium`** subset rather than the smaller benchmark used in earlier stages. The dataset summary printed by the notebook reports:

- **total tracks:** 24,995
- **training tracks:** 19,918
- **validation tracks:** 2,505
- **test tracks:** 2,572
- **missing audio files:** 0
- **excluded bad audio files:** 5
- **subset name:** `fma_medium`

The notebook explicitly excluded the following problematic track IDs:

- `1486`
- `5574`
- `99134`
- `108925`
- `133297`

This matters for two reasons.

First, Notebook 4 is not merely a conceptual scale-up. We actually bootstrapped and validated the `fma_medium` data layout.

Second, it shows that the final run paid attention to dataset hygiene rather than silently treating decode failures as random noise. That makes the evaluation more credible.

## 6. Reliability Checks and Artifact Reuse

Notebook 4 reused historical artifacts across Notebook 2 and Notebook 3 rather than retraining everything from scratch. The execution manifest lists **8 validated historical descriptors**:

### 6.1 Notebook 2 artifacts reused

- `cnn_baseline_embed128`
- `frozen_mert_extended_embed128`
- `hybrid_transformer_baseline_embed128`
- `hybrid_transformer_extended_embed128`
- `hybrid_transformer_extended_embed256`

### 6.2 Notebook 3 artifacts reused

- `hybrid_transformer_filters_only_embed128`
- `hybrid_transformer_one_of_k_embed128`
- `hybrid_transformer_severity_controlled_embed128`

The notebook’s artifact discovery output marked all 8 expected runs as **valid**, with checkpoint and config states recorded as **ok**.

Notebook 4 also ran smoke tests and reported all checks as `ok`, including:

- split inventory
- audio decode probe
- forward/backward pass for CNN
- forward/backward pass for hybrid transformer
- forward/backward pass for frozen MERT
- tiny retrieval and aggregation round-trip
- hard-negative cache round-trip

That means the final stage began from a stable experimental base. Our main reliability caveat is not “the system was broken everywhere.” Our real caveat is narrower: **some planned final synthesis branches were disabled or partially unfinished** because of time and other constraints.

## 7. Evaluation Matrix and Exported Outputs

The attached artifact set makes the final evaluation grid explicit.

### 7.1 Historical baseline evaluation grid

The baseline summary table contains **120 rows**. That corresponds to:

- **8 historical runs**
- **3 windowing strategies**
- **5 index choices**

So the executed baseline matrix covered **120 run/window/index combinations**.

The detailed baseline long table contains **6,000 rows**, which corresponds to:

- the same **120 run/window/index combinations**
- multiplied by **50 regime/condition/query-length combinations**

The active retrieval matrix therefore covered:

- single-condition continuity checks,
- short centered queries,
- short off-center queries,
- combined moderate degradations,
- grouped same-track multi-segment queries,
- realistic hard queries.

### 7.2 Search/index choices evaluated

The active index set was:

- `exact_ip`
- `ivfflat_nprobe1`
- `ivfflat_nprobe4`
- `ivfflat_nprobe8`
- `ivfpq_nprobe4`

### 7.3 Exported artifacts

The attached files show that Notebook 4 exported:

- `notebook4_base_eval_long.csv`
- `notebook4_base_eval_summary.csv`
- `notebook4_cross_model_comparison.csv`
- `notebook4_failure_cases.csv`
- `faiss_sweep_results.csv`
- `notebook4_runtime_info.json`
- `notebook4_config.json`
- `notebook4_execution_manifest.json`
- `notebook4_conclusions.md`
- `notebook4_presentation_summary.md`

The execution manifest reports:

- **baseline long rows:** 6000
- **aggregation long rows:** 0
- **hard-negative long rows:** 0
- **failure rows:** 250

Those zeros are decisive. They mean the final project evidence for Notebook 4 is baseline-heavy by design, not because the report is ignoring available aggregation/hard-negative results. We flipped some flags and turned them off because we encountered a bug during notebook execution and didn't have time to both fix it and re-run the entire notebook. We were also running out of Google Colab credits.

## 8. Best Overall Result

The strongest overall exported configuration was:

- **Run:** `cnn_baseline_embed128`
- **Windowing:** `multi5_even`
- **Index:** `ivfflat_nprobe8`
- **Ranking score:** **0.4009**
- **Clean Top-1:** **0.8346**
- **Combined moderate Top-1:** **0.1904**
- **Realistic hard Top-1:** **0.0452**
- **Mean degraded Top-1:** **0.3642**
- **Worst-condition Top-1:** **0.0145**
- **MRR:** **0.5774**
- **Latency:** **0.0103 ms/query**
- **Index size:** **6.4093 MB**

This is the final recommended deployment choice because it produced the best ranking score among all evaluated historical configurations at `fma_medium` scale.

However, it is important not to misread the result. The winner is best **overall**, not best on every axis:

- it is **not** the best floor-oriented model,
- it is **not** the smallest index,
- and it is **not** the model with the most stable worst-case behavior.

It wins because its total trade-off remains best once the benchmark is scaled up.

## 9. Best Result By Model Family / Run Family

The table below summarizes the best exported configuration for each historical run.

| Run | Family / training profile | Best windowing | Best index | Ranking score | Mean degraded Top-1 | Worst-condition Top-1 | Clean Top-1 | Realistic-hard Top-1 | Latency (ms/query) | Index size (MB) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cnn_baseline_embed128` | CNN / baseline | `multi5_even` | `ivfflat_nprobe8` | 0.4009 | 0.3642 | 0.0145 | 0.8346 | 0.0452 | 0.0103 | 6.4093 |
| `hybrid_transformer_baseline_embed128` | Hybrid transformer / baseline | `multi5_even` | `ivfflat_nprobe8` | 0.3139 | 0.2483 | 0.0052 | 0.5848 | 0.0240 | 0.0094 | 6.4093 |
| `frozen_mert_extended_embed128` | Frozen MERT / extended-existing | `single_center` | `ivfflat_nprobe8` | 0.3055 | 0.2062 | 0.0591 | 0.4571 | 0.0591 | 0.0075 | 1.3074 |
| `hybrid_transformer_one_of_k_embed128` | Hybrid transformer / one-of-k | `single_center` | `exact_ip` | 0.2948 | 0.2051 | 0.0218 | 0.4651 | 0.0218 | 0.0410 | 1.2559 |
| `hybrid_transformer_severity_controlled_embed128` | Hybrid transformer / severity-controlled | `single_center` | `ivfflat_nprobe8` | 0.2910 | 0.1968 | 0.0273 | 0.4369 | 0.0273 | 0.0073 | 1.3074 |
| `hybrid_transformer_filters_only_embed128` | Hybrid transformer / filters-only | `single_center` | `ivfpq_nprobe4` | 0.2743 | 0.1555 | 0.0076 | 0.3466 | 0.0076 | 0.0072 | 0.2158 |
| `hybrid_transformer_extended_embed128` | Hybrid transformer / extended-existing | `single_center` | `ivfpq_nprobe4` | 0.2210 | 0.0631 | 0.0061 | 0.3288 | 0.0061 | 0.0073 | 0.2158 |
| `hybrid_transformer_extended_embed256` | Hybrid transformer / extended-existing 256d | `single_center` | `ivfpq_nprobe4` | 0.2105 | 0.0545 | 0.0053 | 0.3285 | 0.0053 | 0.0076 | 0.3720 |

## 10. Main Lessons From The Cross-Run Ranking

### 10.1 The CNN remained the best overall practical choice

The most important final result is that `cnn_baseline_embed128` remained the strongest overall system even after scaling from `fma_small` to `fma_medium`.

Compared with the second-place `hybrid_transformer_baseline_embed128`, the best CNN configuration improved ranking score by roughly **27.7%** (0.4009 vs. 0.3139). That is not a tiny or ambiguous margin.

So the core Notebook 3 conclusion survived.

### 10.2 The frozen MERT model was the best floor-oriented baseline

The best worst-condition score in the entire cross-model comparison file was not from the CNN. It was from:

- **`frozen_mert_extended_embed128`**
- **worst-condition Top-1 / floor:** **0.0591**

This is approximately **4.08x** the CNN’s floor (0.0591 vs. 0.0145) and approximately **11.44x** the hybrid-baseline floor (0.0591 vs. 0.0052).

That matters because it changes the interpretation of “best model”:

- **best overall:** CNN
- **best floor / least catastrophic collapse:** frozen MERT

Those are not the same objective.

### 10.3 The baseline hybrid transformer remained competitive in average quality, but still brittle

The best hybrid-baseline configuration reached:

- ranking score = **0.3139**
- mean degraded Top-1 = **0.2483**
- worst-condition Top-1 = **0.0052**

That combination is still not enough. Its average degraded retrieval is decent relative to the rest of the pack, but its floor is extremely poor. The attached detailed results show that its hardest failures are still concentrated in very short, high-pass, off-center conditions.

So the old hybrid-transformer story still breaks at the floor.

### 10.4 Selective augmentation policies improved floor relative to the hybrid baseline, but not enough to win

Notebook 3 had already hinted that selective augmentation could improve the hybrid transformer’s worst-case robustness. Notebook 4 confirms that pattern persists at larger scale.

Relative to the hybrid baseline:

- `one_of_k` improved the floor from **0.0052** to **0.0218**
- `severity_controlled` improved the floor from **0.0052** to **0.0273**

That is about:

- **4.22x** better floor for `one_of_k`
- **5.29x** better floor for `severity_controlled`

But those gains came with lower average degraded retrieval:

- `one_of_k` mean degraded Top-1: **0.2051**
- `severity_controlled` mean degraded Top-1: **0.1968**
- hybrid baseline mean degraded Top-1: **0.2483**

So the selective policies made the transformer **less brittle**, but not **best overall**.

### 10.5 The old extended hybrid configurations remained weak

The two older extended hybrid runs were among the weakest overall configurations in the final table:

- `hybrid_transformer_extended_embed128`: ranking score **0.2210**
- `hybrid_transformer_extended_embed256`: ranking score **0.2105**

These runs still show the same core problem identified earlier: broad augmentation stacking did not produce a uniformly strong representation. It mostly redistributed failure modes.

## 11. Notebook 3 vs. Notebook 4: What Changed At `fma_medium` Scale

Notebook 4 is most useful when compared directly against Notebook 3.

### 11.1 The winner stayed the same, but the best configuration changed

Notebook 3’s best exported configuration was:

- `cnn_baseline_embed128`
- `multi3_even`
- `exact_ip`
- ranking score = **0.4749**

Notebook 4’s best exported configuration is:

- `cnn_baseline_embed128`
- `multi5_even`
- `ivfflat_nprobe8`
- ranking score = **0.4009**

So the **run identity** of the winner stayed the same, but the **best system-level retrieval design** changed.

The final synthesis therefore did not overturn the model recommendation, but it **did** refine the deployment recommendation.

### 11.2 The scale-up hurt the floor more than the average case

Comparing Notebook 3’s best result with Notebook 4’s best result:

| Metric | Notebook 3 best | Notebook 4 best | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Ranking score | 0.4749 | 0.4009 | -0.0740 | -15.6% |
| Mean degraded Top-1 | 0.4817 | 0.3642 | -0.1175 | -24.4% |
| Worst-condition Top-1 | 0.0239 | 0.0145 | -0.0094 | -39.3% |
| Clean Top-1 | 0.8646 | 0.8346 | -0.0300 | -3.5% |
| Combined-moderate Top-1 | 0.2452 | 0.1904 | -0.0548 | -22.4% |

The most important row is the worst-condition row. The mean degraded score dropped by about **24.4%**, but the worst-condition score dropped by about **39.3%**. That directly supports the project’s final presentation summary claim that the `fma_medium` scale-up changed the floor more than the average case.

### 11.3 The scale-up made the benchmark more honest

This is the central scientific value of Notebook 4. It did not suddenly produce a new winner. Instead, it made the surviving winner look **less invincible**.

That is useful. A benchmark that preserves ranking but exposes more realistic weakness is still a better benchmark.

## 12. Windowing Effects: `single_center` vs. `multi3_even` vs. `multi5_even`

Notebook 4’s most important successful system-level intervention was richer reference coverage through multi-window indexing.

### 12.1 CNN winner: `multi5_even` was worth it

For `cnn_baseline_embed128` with `ivfflat_nprobe8`:

| Windowing | Clean Top-1 | Mean degraded Top-1 | Realistic-hard Top-1 | Worst-condition Top-1 | Latency (ms/query) | Index size (MB) | Ranking score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `single_center` | 0.6512 | 0.3361 | 0.0368 | 0.0148 | 0.0074 | 1.3074 | 0.3859 |
| `multi3_even` | 0.8050 | 0.3481 | 0.0422 | 0.0142 | 0.0085 | 3.8583 | 0.3931 |
| `multi5_even` | 0.8346 | 0.3642 | 0.0452 | 0.0145 | 0.0103 | 6.4093 | 0.4009 |

Relative to `single_center`, `multi5_even` for the CNN:

- improved clean Top-1 by **0.1834**
- improved mean degraded Top-1 by **0.0282**
- improved realistic-hard Top-1 by **0.0084**
- increased latency by only about **0.0029 ms/query**
- increased index size by about **5.1019 MB**

The floor changed very little, but the broader retrieval profile improved enough to make `multi5_even` the winning system configuration.

### 12.2 The benefit was not purely CNN-specific

For the baseline hybrid transformer with `ivfflat_nprobe8`, moving from `single_center` to `multi5_even` also improved:

- clean Top-1 by **0.1702**
- mean degraded Top-1 by **0.0145**
- realistic-hard Top-1 by **0.0095**

So the multi-window idea remained useful. But the CNN benefited more in practical ranking terms, and the hybrid model still did not solve its floor problem.

### 12.3 Interpretation

Multi-window indexing is one of the clearest “cheap” upgrades validated by Notebook 4 because it:

- reuses historical checkpoints,
- does not require retraining,
- materially improves average retrieval,
- and became part of the final winning configuration.

That is exactly the kind of system-level improvement a final project should prioritize.

## 13. FAISS / Indexing Effects

The final winning configuration did **not** use exact search. It used `ivfflat_nprobe8`.

That is an important system-design result.

### 13.1 Why `ivfflat_nprobe8` won for the final recommendation

For the winning CNN with `multi5_even`:

| Index | Mean degraded Top-1 | Worst-condition Top-1 | MRR | Latency (ms/query) | Index size (MB) | Ranking score |
|---|---:|---:|---:|---:|---:|---:|
| `exact_ip` | 0.3668 | 0.0146 | 0.5803 | 0.1234 | 6.2793 | 0.3927 |
| `ivfflat_nprobe8` | 0.3642 | 0.0145 | 0.5774 | 0.0103 | 6.4093 | 0.4009 |
| `ivfflat_nprobe4` | 0.3543 | 0.0144 | 0.5660 | 0.0089 | 6.4093 | 0.3944 |
| `ivfpq_nprobe4` | 0.2902 | 0.0105 | 0.5059 | 0.0089 | 0.4513 | 0.3775 |
| `ivfflat_nprobe1` | 0.2787 | 0.0119 | 0.4785 | 0.0084 | 6.4093 | 0.3426 |

Compared with exact search, `ivfflat_nprobe8`:

- lost only **-0.0025** in mean degraded Top-1,
- lost only **-0.0001** in worst-condition Top-1,
- but reduced latency by about **-0.1131 ms/query**.

That is a very favorable trade-off, and it is exactly why the approximate index became the final recommendation.

### 13.2 Broad pattern across all runs

Averaged across all summary rows, the index trade-offs were:

| Index | Mean ranking score | Mean degraded Top-1 | Mean floor | Mean latency (ms/query) | Mean index size (MB) |
|---|---:|---:|---:|---:|---:|
| `ivfflat_nprobe8` | 0.2796 | 0.1847 | 0.0205 | 0.0084 | 4.3332 |
| `ivfflat_nprobe4` | 0.2749 | 0.1779 | 0.0199 | 0.0079 | 4.3332 |
| `exact_ip` | 0.2741 | 0.1868 | 0.0206 | 0.0883 | 4.2386 |
| `ivfpq_nprobe4` | 0.2736 | 0.1452 | 0.0157 | 0.0078 | 0.3530 |
| `ivfflat_nprobe1` | 0.2435 | 0.1344 | 0.0162 | 0.0077 | 4.3332 |

The broad takeaway is:

- **`ivfflat_nprobe8`** gave the best average ranking trade-off.
- **exact search** still had excellent quality, but paid a noticeable latency penalty.
- **IVFPQ** produced dramatically smaller indexes, but at a meaningful quality cost.
- **low-probe IVFFlat** (`nprobe1`) was too aggressive.

## 14. The Hardest Failure Modes

The final project would be incomplete without saying where the winning system still fails.

### 14.1 Worst rows for the final recommended CNN system

For the best overall configuration (`cnn_baseline_embed128`, `multi5_even`, `ivfflat_nprobe8`), the weakest detailed cases were:

| Regime | Condition | Query length (s) | Top-1 |
|---|---|---:|---:|
| `short_offcenter` | `highpass` | 1.0 | 0.0097 |
| `short_offcenter` | `highpass` | 3.0 | 0.0121 |
| `short_offcenter` | `highpass` | 2.0 | 0.0132 |
| `short_centered` | `highpass` | 1.0 | 0.0140 |
| `short_centered` | `highpass` | 2.0 | 0.0167 |
| `clean_current` | `highpass` | 3.0 | 0.0179 |
| `realistic_hard` | `realistic_hard` | 1.0 | 0.0408 |

So the final recommended system is still extremely vulnerable to **high-pass filtered, short, off-center queries**. That is the sharpest remaining unresolved weakness.

### 14.2 Hardest conditions for the baseline hybrid transformer

For the best baseline hybrid configuration, the weakest cases were even worse and again centered on short high-pass queries:

- `short_offcenter`, `highpass`, 1s: **0.0039**
- `short_centered`, `highpass`, 1s: **0.0047**
- `clean_current`, `highpass`, 3s: **0.0058**

This confirms that the hybrid baseline still fails on the same family of brittle frequency-response shifts.

### 14.3 Hardest conditions for frozen MERT

The frozen MERT run had a very different worst-case profile. Its hardest rows were not dominated by high-pass alone. Instead, its weakest cases included:

- `short_offcenter`, `pitch`, 1s: **0.0191**
- `realistic_hard`, 1s: **0.0362**
- `combined_moderate`, 1s: **0.0397**

That is exactly why its floor is higher overall: it is less catastrophically destroyed by the single worst perturbation family, even though its average retrieval remains weaker.

## 15. What The Exported Failure Sample Says

The attached `notebook4_failure_cases.csv` contains the **250 most severe remaining misses**, sorted by rank severity and score, not an unbiased random sample of all errors.

That distinction matters.

Within that exported failure sample:

- the file is heavily dominated by `multi_segment_same_track`
- especially **clean 1-second and 2-second grouped fragments**
- and especially for the weaker transformer variants, most notably `hybrid_transformer_filters_only_embed128`

The most common entries in the exported failure sample are:

| Run | Regime | Condition | Query length (s) | Count in exported top-250 severe misses |
|---|---|---|---:|---:|
| `hybrid_transformer_filters_only_embed128` | `multi_segment_same_track` | `clean` | 1.0 | 123 |
| `hybrid_transformer_filters_only_embed128` | `multi_segment_same_track` | `clean` | 2.0 | 77 |
| `hybrid_transformer_baseline_embed128` | `multi_segment_same_track` | `clean` | 2.0 | 16 |
| `hybrid_transformer_one_of_k_embed128` | `multi_segment_same_track` | `clean` | 1.0 | 11 |
| `hybrid_transformer_baseline_embed128` | `multi_segment_same_track` | `clean` | 1.0 | 10 |

The correct interpretation is not “only multi-segment retrieval is hard.” The correct interpretation is:

- when the notebook selects the **most severe misses** for manual review,
- grouped short-fragment retrieval disproportionately occupies those slots,
- especially for the weaker transformer runs.

That supports the original motivation for multi-segment aggregation logic, even though the actual aggregation evaluation was not executed in the final artifact set.

## 16. Aggregation and Hard-Negative Retraining

### 16.1 What we did

Notebook 4 **contains**:

- aggregation configuration,
- aggregation helper logic,
- hard-negative configuration,
- hard-negative cache logic,
- query-regime definitions for grouped fragments and hard-negative-oriented short queries,
- and presentation-level conclusions that treat aggregation as the cheapest inference-time intervention and hard-negative retraining as the training-time intervention for acoustically similar false matches.

### 16.2 What we didn't do but wanted to

The executed configuration and export manifest show:

- `run_aggregation_eval = False`
- `run_hard_negative_retraining = False`
- aggregation long rows = **0**
- hard-negative long rows = **0**
- hard-negative artifacts = **[]**
- hard-negative targets retrained = **0**

Therefore, we weren't able to include these experiments in the final report. We wanted to do these experiments, but we didn't have enough time time or Google Colab credits left to fix all the bugs and also run them.

## 17. Execution Issues

The saved Notebook 4 file includes a visible `KeyError` trace during the cross-model-comparison synthesis step when `build_cross_model_comparison(...)` attempted to access absent aggregation results.

That detail is important because it explains part of the  messiness: the notebook still had a synthesis assumption that aggregation results would exist, even when aggregation was disabled.

At the same time, the attached exports include:

- `notebook4_cross_model_comparison.csv`
- `notebook4_conclusions.md`
- the final zip-bundle output reference

The final notebook transcript itself is not a perfectly clean, single-pass execution log. This is a limitation we felt was worth documenting rather than hiding.

## 18. Final Hypothesis Review

| Hypothesis | Verdict | Reason |
|---|---|---|
| The Notebook 3 winner would survive the move to `fma_medium`. | **Supported.** | `cnn_baseline_embed128` remained the best overall run. |
| The best system configuration would remain unchanged. | **Not supported.** | The winning run stayed the same, but the best retrieval setup shifted to `multi5_even` + `ivfflat_nprobe8`. |
| Larger-scale evaluation would reveal sharper floor weakness than average-case weakness. | **Supported.** | The best model’s mean degraded Top-1 fell by about 24.4% relative to Notebook 3, but the worst-condition Top-1 fell by about 39.3%. |
| Multi-window indexing would still be useful at larger scale. | **Supported.** | `multi5_even` became part of the final winning configuration and materially improved average retrieval relative to `single_center`. |
| Aggregation would be shown to be the cheapest effective floor intervention. | **Not demonstrated in the attached final artifact set.** | Aggregation evaluation was disabled and no aggregation rows were exported. |
| Hard-negative retraining would be shown to improve discriminability. | **Not demonstrated in the attached final artifact set.** | Hard-negative retraining was disabled and no retrained targets were exported. |

## 19. Final Recommendation For The Project

### 19.1 Final recommended configuration

The final recommended system for the project report, presentation, and any practical default deployment discussion is:

- **Model:** `cnn_baseline_embed128`
- **Reference windowing:** `multi5_even`
- **Index:** `ivfflat_nprobe8`

This is the final recommendation because it provides the best overall ranking score and the strongest overall balance of retrieval quality and system cost among the actually executed configurations.

### 19.2 How to describe it honestly

Our final correct description should not be “the CNN solved the problem.”

Our correct description should be:

- the CNN is the **best overall trade-off**,
- multi-window references are the **best validated inference-time upgrade** that we validated at `fma_medium` scale,
- approximate IVFFlat search is **good enough to beat exact search on the project’s combined trade-off metric**,
- but the robustness floor is still poor, especially for short, high-pass, off-center queries.
- aggregation and hard-negative retraining are **well-motivated next steps** that we wanted to validate but didn't have time to run in the final artifact set.

### 19.3 How to frame the frozen MERT result

We want to present Frozen MERT as:

- the **best floor-oriented baseline**,
- a useful reference point for robustness-under-collapse,
- but not the best overall deployment candidate.

We feel that this is a more accurate and more interesting conclusion than simply calling it “third place.”

### 19.4 How to frame aggregation and hard negatives in the final project story

We want to present aggregation and hard-negative retraining as:

- Our **well-motivated next steps**
- **partially scaffolded in Notebook 4**
- but **not fully executed in the final attached artifact set due to unforseen execution issues and budget limits**

## 20. Bottom Line

Notebook 4 succeeded in its most important job: it delivered the final large-scale re-evaluation that the project needed before making a better recommendation.

So we place this project in hiatus with a slightly better amount of progress compared to just leaving it off after notebook 3.

For now our working conclusion is:

> **Use the CNN baseline as the final recommended model, pair it with richer multi-window references and `ivfflat_nprobe8`, and explicitly acknowledge that the remaining unsolved problem is worst-case robustness on short, high-pass, off-center queries.**

Ideally, with more time and resources, we would have been able to run the aggregation and hard-negative retraining experiments and validate them properly.

## 21. Source Materials

This report was written from the attached project materials:

- `04_fma_medium_scaleup_hard_negatives_and_temporal_aggregation.ipynb`
- `03_robustness_ablation_and_realistic_evaluation.ipynb`
- `Progress Report 3.md`
- `README.md`
- `experiment_registry.json`
- `notebook4_config.json`
- `notebook4_runtime_info.json`
- `notebook4_execution_manifest.json`
- `notebook4_conclusions.md`
- `notebook4_presentation_summary.md`
- `notebook4_base_eval_summary.csv`
- `notebook4_base_eval_long.csv`
- `notebook4_cross_model_comparison.csv`
- `notebook4_failure_cases.csv`
- `faiss_sweep_results.csv`

And all the papers under the `papers` directory.
