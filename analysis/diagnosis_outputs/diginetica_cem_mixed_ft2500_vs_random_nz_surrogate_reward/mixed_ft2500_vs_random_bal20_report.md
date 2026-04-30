# Diginetica RankBucket-CEM mixed ft2500 Target 5334 Report

## Inputs
- mixed-ft2500 run: `outputs/runs/diginetica/attack_rank_bucket_cem_mixed_ft2500_srgnn_target5334/run_group_73b4185f37`
- bal20-ft500 run: `outputs/runs/diginetica/attack_rank_bucket_cem_bal20_ft500_srgnn_target5334/run_group_3787674b40`
- Random-NZ run: `outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_8679b974a1`
- target item: `5334`; victim model: `srgnn`

## Parameters
- CEM iterations/population/elite: `3` / `8` / `0.25`
- Reward: `target_result.mean` (`reward_metric=None`), reward_mode=`poisoned_target_utility`
- Normal mixed surrogate fine-tune: `fine_tune_steps=2500`, poison balance enabled=`False`
- Best trace actual optimizer steps: `2500`; configured steps: `2500`
- Natural mixed poison ratio was not directly tracked in normal path; expected poison prefix exposure is about `2500 * 100 * 3.7394% ~= 9348` poison prefix samples.

## Training Time
| run | total | CEM/search until victim starts | final SR-GNN victim | CEM candidate mean | fine-tune mean | score-target mean |
|---|---:|---:|---:|---:|---:|---:|
| `mixed-ft2500` | 1h 21m 41.0s | 46m 39.1s | 35m 1.9s | 1m 55.9s | 1m 42.2s | 0m 13.2s |
| `bal20-ft500` | 1h 4m 49.7s | 16m 5.0s | 48m 44.7s | 0m 39.6s | 0m 26.7s | 0m 13.0s |

- mixed-ft2500 CEM candidate total summed: `46m 22.6s` over `24` candidates.
- bal20-ft500 CEM candidate total summed: `15m 51.3s` over `24` candidates.

## Surrogate Reward Alignment
Random-NZ was rescored with the same normal mixed ft2500 surrogate evaluator.

| method | surrogate reward | target_result.mean | targeted_mrr@10 | targeted_recall@10 | targeted_recall@20 |
|---|---:|---:|---:|---:|---:|
| `cem_best_stored` | 0.00994209 | 0.00994209 | 0.080283 | 0.174466 | 0.268342 |
| `cem_best_rescored` | 0.00994209 | 0.00994209 | 0.080283 | 0.174466 | 0.268342 |
| `random_nz_ratio1` | 0.00800529 | 0.00800529 | 0.065554 | 0.140556 | 0.220167 |

- mixed CEM stored minus Random-NZ surrogate reward: `+0.00193679`
- mixed CEM rescored minus Random-NZ surrogate reward: `+0.00193680`
- mixed CEM rescored minus Random-NZ surrogate targeted_mrr@10: `+0.014730`
- mixed CEM rescored minus Random-NZ surrogate targeted_recall@10: `+0.033910`
- mixed CEM rescored minus Random-NZ surrogate targeted_recall@20: `+0.048175`

For reference, bal20-ft500 under its own fixed-ratio evaluator had a much larger CEM-vs-Random surrogate margin:
- bal20 CEM minus Random-NZ reward: `+0.01009899`
- bal20 CEM reward: `0.03302547`, Random-NZ reward: `0.02292648`

## Final Victim Target Metrics
- mixed vs Random-NZ targeted: `30/32` mixed wins, `2` Random-NZ wins, `0` ties.
- mixed vs bal20 targeted: `12/32` mixed wins, `20` bal20 wins, `0` ties.

| metric | mixed | bal20 | random | mixed-random | mixed-bal20 | rel vs random | rel vs bal20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `targeted_recall@5` | 0.077229 | 0.075421 | 0.075126 | +0.002103 | +0.001807 | 2.80% | 2.40% |
| `targeted_recall@10` | 0.135857 | 0.135118 | 0.128972 | +0.006885 | +0.000739 | 5.34% | 0.55% |
| `targeted_recall@20` | 0.217638 | 0.218114 | 0.214121 | +0.003516 | -0.000477 | 1.64% | -0.22% |
| `targeted_recall@30` | 0.279520 | 0.281919 | 0.276545 | +0.002974 | -0.002399 | 1.08% | -0.85% |
| `targeted_recall@50` | 0.371981 | 0.377436 | 0.375185 | -0.003204 | -0.005455 | -0.85% | -1.45% |
| `targeted_mrr@5` | 0.040340 | 0.039810 | 0.039681 | +0.000659 | +0.000529 | 1.66% | 1.33% |
| `targeted_mrr@10` | 0.047947 | 0.047576 | 0.046666 | +0.001280 | +0.000371 | 2.74% | 0.78% |
| `targeted_mrr@20` | 0.053475 | 0.053289 | 0.052452 | +0.001023 | +0.000186 | 1.95% | 0.35% |
| `targeted_mrr@30` | 0.055938 | 0.055851 | 0.054943 | +0.000994 | +0.000087 | 1.81% | 0.16% |
| `targeted_mrr@50` | 0.058301 | 0.058309 | 0.057462 | +0.000839 | -0.000007 | 1.46% | -0.01% |
| `targeted_ndcg@10` | 0.068112 | 0.067631 | 0.065574 | +0.002539 | +0.000481 | 3.87% | 0.71% |
| `targeted_ndcg@20` | 0.088606 | 0.088556 | 0.086947 | +0.001659 | +0.000050 | 1.91% | 0.06% |

## Final Victim Ground-Truth Metrics
- mixed vs Random-NZ ground-truth: `30/32` mixed wins, `2` Random-NZ wins, `0` ties.
- mixed vs bal20 ground-truth: `32/32` mixed wins, `0` bal20 wins, `0` ties.

| metric | mixed | bal20 | random | mixed-random | mixed-bal20 | rel vs random | rel vs bal20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ground_truth_recall@10` | 0.337425 | 0.335026 | 0.336406 | +0.001019 | +0.002399 | 0.30% | 0.72% |
| `ground_truth_recall@20` | 0.456555 | 0.453219 | 0.455372 | +0.001183 | +0.003336 | 0.26% | 0.74% |
| `ground_truth_recall@30` | 0.526521 | 0.523366 | 0.526356 | +0.000164 | +0.003155 | 0.03% | 0.60% |
| `ground_truth_mrr@10` | 0.146995 | 0.145335 | 0.146679 | +0.000316 | +0.001660 | 0.22% | 1.14% |
| `ground_truth_mrr@20` | 0.155197 | 0.153521 | 0.154922 | +0.000275 | +0.001676 | 0.18% | 1.09% |
| `ground_truth_mrr@30` | 0.158021 | 0.156354 | 0.157784 | +0.000237 | +0.001666 | 0.15% | 1.07% |

## Position Summary
| field | mixed-ft2500 | bal20-ft500 | Random-NZ |
|---|---:|---:|---:|
| `total_fake_sessions` | 6499.0000 | 6499.0000 | 6499.0000 |
| `mean_absolute_position` | 2.4967 | 2.3710 | 2.4662 |
| `median_absolute_position` | 1.0000 | 1.0000 | 2.0000 |
| `pos1_pct` | 51.9311 | 51.6695 | 48.6383 |
| `pos2_pct` | 8.6167 | 14.1406 | 20.7878 |
| `pos3_pct` | 18.3721 | 16.2025 | 11.6787 |
| `pos4_pos5_pct` | 12.5712 | 10.3554 | 10.1708 |
| `pos6plus_pct` | 8.5090 | 7.6319 | 8.7244 |
| `rank1_pct` | 51.9311 | 51.6695 | 48.6383 |
| `rank2_pct` | 8.6167 | 14.1406 | 20.7878 |
| `tail_pct` | 39.4522 | 34.1899 | 30.5739 |

## Interpretation
- Against Random-NZ, mixed-ft2500 final targeted_mrr@10 changes by `+0.001280` and targeted_recall@20 by `+0.003516`.
- Against bal20-ft500, mixed-ft2500 final targeted_mrr@10 changes by `+0.000371` and targeted_recall@20 by `-0.000477`.
- Surrogate ranking is aligned relative to Random-NZ for mixed-ft2500: CEM has higher mixed-ft2500 surrogate reward and higher final target metrics than Random-NZ.
- However, mixed-ft2500 is weaker than bal20-ft500 on final target metrics. This suggests bal20-ft500 did not improve only because the surrogate saw more poison samples; changing the batch composition to 20% poison also changed the reward-training dynamics in a useful way.
- Ground-truth quality is not the tradeoff in this mixed-ft2500 run: mixed-ft2500 beats Random-NZ on `30/32` ground-truth metrics and beats bal20-ft500 on `32/32` ground-truth metrics. The main tradeoff is runtime: mixed-ft2500 spends far more time in CEM surrogate fine-tuning than bal20-ft500.

## Output Files
- Report: `analysis/diagnosis_outputs/diginetica_cem_mixed_ft2500_vs_random_nz_surrogate_reward/mixed_ft2500_vs_random_bal20_report.md`
- Three-way final metrics: `analysis/diagnosis_outputs/diginetica_cem_mixed_ft2500_vs_random_nz_surrogate_reward/final_metric_three_way.csv`
- Compact three-way final metrics: `analysis/diagnosis_outputs/diginetica_cem_mixed_ft2500_vs_random_nz_surrogate_reward/final_metric_three_way_compact.csv`
- Mixed-ft2500 surrogate comparison: `analysis/diagnosis_outputs/diginetica_cem_mixed_ft2500_vs_random_nz_surrogate_reward/surrogate_reward_comparison.csv`
- Mixed-ft2500 surrogate pairwise: `analysis/diagnosis_outputs/diginetica_cem_mixed_ft2500_vs_random_nz_surrogate_reward/surrogate_reward_pairwise.csv`
