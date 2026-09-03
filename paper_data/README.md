# TC-SACP manuscript data

Numerical data for **Targeted Data Poisoning Attacks against Session-Based
Recommendation Systems via Target-Centered Suffix-Action Construction**,
by Sheng-Yung Pai and Bi-Ru Dai. Figure/table numbers refer to the Information
Sciences manuscript and its separate Supplementary Material (September 2026).

This is a compact result-data release, not a dump of every experimental run.
CSV files are UTF-8, have a single header row, and use a decimal point.
Unrounded source values are retained where available. Training logs, model
weights, session sequences, and the removed thesis appendix table are not included.

## Figure and table mapping

| Manuscript item | Files / selection |
|---|---|
| Table 1: dataset statistics | `table1_dataset_statistics.csv` (reported manuscript values) |
| Table 2: main Recall@20 comparison | `table2_recall20.csv`; underlying observations in `target_metrics.csv` |
| Figure 4: poisoning-budget sensitivity | `figure4_budget.csv`; underlying observations in `budget_target_metrics.csv` |
| Table 3: ranks under three surrogates | `table3_mean_ranks.csv`; underlying observations in `target_metrics.csv` |
| Figure 5: primary Recall@K curves | `metric_summary.csv`, `surrogate_setting=SR-GNN`, `metric=recall`, excluding `CLEAN` |
| Figure S1: ranked item frequencies | `figureS1_frequency_distribution.csv` |
| Table S1: frequency ranges | `tableS1_frequency_ranges.csv` (also includes the unsampled intermediate range) |
| Figure S2: MDHG-surrogate Recall@K | `metric_summary.csv`, `surrogate_setting=MDHG`, `metric=recall` |
| Figure S3: FreqRec-surrogate Recall@K | `metric_summary.csv`, `surrogate_setting=FreqRec`, `metric=recall` |
| Table S2: six-method ranks and attack time | `tableS2_mean_ranks.csv`, `tableS2_attack_time.csv`; rank inputs in `ablation_target_metrics.csv` and primary `target_metrics.csv` |
| Figure S4: individual-target action use | `figureS4_action_usage.csv` |
| Table S3: action-use mean and SD | `tableS3_action_summary.csv` |

Figures 1-3 are conceptual illustrations and have no experimental numerical data.

## Columns and units

- `dataset`: `diginetica` or `yoochoose1_64`; `target_type`: popular/unpopular cohort.
- `target_item`: item identifier in the experiment's processed data; it is not
  necessarily the identifier in the original downloaded dataset. `plot_label`
  maps the ascending target IDs to T1-T10 in Figure S4.
- `victim_model`: srgnn, miasrec, tron, mdhg, freqrec, or wearec.
- `surrogate_setting`: which TC-SACP surrogate comparison a row belongs to.
  Baselines are repeated in each comparison; this does not mean they use that
  surrogate. In Table 2, `surrogate=N/A` marks CLEAN and the baselines.
- `method`: manuscript method name. `TC-SACP-G w/o Sur.` is the no-surrogate
  uniform-policy ablation.
- `metric`: targeted `recall` or `mrr`; `k` is the recommendation cutoff.
  `value`, `mean`, `std`, `recall20`, `recall20_mean`, and `recall20_std`
  use the 0-1 scale, not percentages. `count` is the number of target items.
- `attack_size`: poisoned-session budget as a fraction (0.005 = 0.5%,
  0.01 = 1%, 0.03 = 3%).
- `mean_rank`: smaller is better. Table 3 ranks five methods separately under
  each surrogate. Table S2 ranks six methods, including the ablation.
- In `tableS2_attack_time.csv`, `minutes` is the published rounded mean when
  `comparison` is `=`; with `<`, it is a strict upper bound, **not** an exact
  measurement. These are attack-side construction times, excluding victim
  training/evaluation, not a four-GPU throughput-normalized quantity.
- `frequency`: occurrences of an item in clean canonical training sessions
  (`train_sub`), before sequence-label expansion. `rank` starts at 1;
  `rank_percentile = 100 * (rank - 1) / (number_of_items - 1)`.
  Popular items have frequency greater than the mean; unpopular items have
  frequency below 10. `share_percent` is the fraction of distinct items in
  a range, expressed as a percentage.
- Action `*_count` columns count sampled actions; `*_percent` columns use
  the 0-100 scale. Keep/generate each combine all suffix-length choices;
  the three action counts sum to `total_actions`.
- Table 1 preserves the manuscript's reported statistics and precision.
  Training/test counts refer to expanded sequence-label pairs, and mean
  expanded session length includes the next-item label. These columns are
  not the same statistics as the training-frequency distribution in Figure S1.

## Aggregation and source selection

Metric summaries use the arithmetic mean and **sample SD (ddof=1)** across
ten target items. Ranks are assigned to cohort-mean metric values within
each dataset/cohort/victim/cutoff, using descending performance and average
ranks for ties, then averaged across the 24 dataset/cohort/victim settings.
Action summaries also use sample SD across ten targets.

The three comparison exports come from the existing final comparison tables
(`formal_6method6victim_all`, `mdhg_surrogate_generated_copy_cem_average_recall`,
and `freqrec_surrogate_generated_copy_cem_average_recall`). Budget observations
come from the selected-value manifests for 0.5%/3% and the primary 1% comparison.
The no-surrogate data use the paper-matched August 24 analysis archive for
Diginetica and Yoochoose unpopular, plus the completed local Yoochoose popular
results. Later replacement runs are not substituted for the paper's inputs.
Action counts come from the selected rank-1 candidate of each primary run.
Table S2 timing values are released at their published precision.

Before export, all 1,800 Recall@K plot points, all 360 budget means, all 60
Table S2 rank entries, the Table S1 cohort counts, and all 24 rounded action
mean/SD values were checked against the corresponding source calculations.

## Benchmark sources

The original third-party datasets are **not redistributed** here. Obtain them
from their providers and follow the applicable access and usage terms.

- **Diginetica / CIKM Cup 2016 Track 2:**
  https://competitions.codalab.org/competitions/11161
- **Yoochoose / RecSys Challenge 2015:** official source description at
  https://recsys.acm.org/recsys15/challenge/ ; dataset paper:
  https://doi.org/10.1145/2792838.2798723 . The original challenge site is
  http://2015.recsyschallenge.com/challenge.html ; a third-party download
  mirror already referenced by this repository is
  https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015 .

The manuscript uses Diginetica and Yoochoose 1/64, not full Yoochoose.
Preprocessing and split definitions are described in the manuscript and
implemented in `attack/data/unified_split.py` and `attack/data/dataset_specs.py`
in the repository. No claim is made that third-party raw datasets are owned
by the manuscript authors.

Code snapshot accompanying this release:
https://github.com/Welonbai/SR-GNN-robustness/tree/67a98c97d555d692ae5f0e0f18cd1118f541a120
