# Diginetica RankBucket-CEM bal20 ft500 Target 5334 Report

## Inputs
- CEM run root: `outputs/runs/diginetica/attack_rank_bucket_cem_bal20_ft500_srgnn_target5334/run_group_3787674b40`
- Random-NZ run root: `outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_8679b974a1`
- Surrogate rescore artifact: `analysis/diagnosis_outputs/diginetica_cem_bal20_ft500_vs_random_nz_surrogate_reward/bal20_ft500_random_nz_fixed_ratio_rescore.json`
- Target item: `5334`
- Victim model compared: `srgnn`

## Parameters
- CEM iterations/population/elite: `3` / `8` / `0.25`
- CEM std/smoothing/min_std: `1.0` / `0.3` / `0.2`
- Reward metric: `None` means `target_result.mean`
- Reward mode: `poisoned_target_utility`
- Fine-tune steps: `500`
- Poison balance: enabled=`True`, mode=`fixed_ratio`, ratio=`0.2`, loss_weighting=`none`
- Fixed-ratio batch: clean=`80`, poison=`20`, batch_size=`100`
- Candidate exposure, best CEM: clean examples seen=`40000`, poison examples seen=`10000`, unique poison prefix examples=`10000`, unique poison source sessions=`4571`
- Replacement top-k ratio: `1.0`; nonzero action when possible: `True`

## Training Time
| stage | time | notes |
|---|---:|---|
| total run wall time | 1h 4m 49.7s | progress elapsed_seconds |
| CEM/search until victim starts | 16m 5.0s | run start to SR-GNN victim start |
| final SR-GNN victim training/eval | 48m 44.7s | victim start to completed |
| CEM candidate total, summed | 15m 51.3s | 24 candidates |
| CEM candidate total, mean | 0m 39.6s | min 0m 29.2s, max 0m 51.5s |
| CEM fine-tune per candidate, mean | 0m 26.7s | sum 10m 40.4s |
| CEM score-target per candidate, mean | 0m 13.0s | sum 5m 10.9s |

### Timing Reference
| run | total | CEM/search until victim starts | final SR-GNN victim |
|---|---:|---:|---:|
| `mrr10_ft20` | 48m 53.4s | 9m 30.1s | 39m 23.2s |
| `mean_ft100` | 49m 17.3s | 10m 32.3s | 38m 45.0s |
| `bal20_ft500` | 1h 4m 49.7s | 16m 5.0s | 48m 44.7s |

## Surrogate Ranking Comparison
Same bal20/ft500 fixed-ratio surrogate evaluator was used to rescore Random-NZ@1.0.

| method | surrogate reward | target_result.mean | targeted_mrr@10 | targeted_recall@10 | targeted_recall@20 |
|---|---:|---:|---:|---:|---:|
| `cem_best_stored` | 0.03302547 | 0.03302547 | 0.222071 | 0.399681 | 0.502502 |
| `random_nz_fixed_ratio_rescore` | 0.02292648 | 0.02292648 | 0.198428 | 0.381000 | 0.494449 |

- CEM minus Random-NZ surrogate reward: `+0.01009899`
- CEM minus Random-NZ surrogate targeted_mrr@10: `+0.023642`
- CEM minus Random-NZ surrogate targeted_recall@10: `+0.018680`
- CEM minus Random-NZ surrogate targeted_recall@20: `+0.008053`

## Victim Target Metric Comparison
- Targeted metrics: `32/32` CEM wins, `0` Random-NZ wins, `0` ties.

| metric | CEM bal20-ft500 | Random-NZ ratio1 | delta | rel. delta | winner |
|---|---:|---:|---:|---:|---|
| `targeted_recall@5` | 0.075421 | 0.075126 | +0.000296 | 0.39% | cem_bal20_ft500 |
| `targeted_recall@10` | 0.135118 | 0.128972 | +0.006145 | 4.76% | cem_bal20_ft500 |
| `targeted_recall@20` | 0.218114 | 0.214121 | +0.003993 | 1.86% | cem_bal20_ft500 |
| `targeted_recall@30` | 0.281919 | 0.276545 | +0.005373 | 1.94% | cem_bal20_ft500 |
| `targeted_recall@50` | 0.377436 | 0.375185 | +0.002251 | 0.60% | cem_bal20_ft500 |
| `targeted_mrr@5` | 0.039810 | 0.039681 | +0.000130 | 0.33% | cem_bal20_ft500 |
| `targeted_mrr@10` | 0.047576 | 0.046666 | +0.000909 | 1.95% | cem_bal20_ft500 |
| `targeted_mrr@20` | 0.053289 | 0.052452 | +0.000837 | 1.60% | cem_bal20_ft500 |
| `targeted_mrr@30` | 0.055851 | 0.054943 | +0.000907 | 1.65% | cem_bal20_ft500 |
| `targeted_mrr@50` | 0.058309 | 0.057462 | +0.000847 | 1.47% | cem_bal20_ft500 |
| `targeted_ndcg@10` | 0.067631 | 0.065574 | +0.002058 | 3.14% | cem_bal20_ft500 |
| `targeted_ndcg@20` | 0.088556 | 0.086947 | +0.001609 | 1.85% | cem_bal20_ft500 |

## Victim Ground-Truth Metric Comparison
- Ground-truth metrics: `0/32` CEM wins, `32` Random-NZ wins, `0` ties.

| metric | CEM bal20-ft500 | Random-NZ ratio1 | delta | rel. delta | winner |
|---|---:|---:|---:|---:|---|
| `ground_truth_recall@10` | 0.335026 | 0.336406 | -0.001380 | -0.41% | random_nz_ratio1 |
| `ground_truth_recall@20` | 0.453219 | 0.455372 | -0.002153 | -0.47% | random_nz_ratio1 |
| `ground_truth_recall@30` | 0.523366 | 0.526356 | -0.002991 | -0.57% | random_nz_ratio1 |
| `ground_truth_mrr@10` | 0.145335 | 0.146679 | -0.001343 | -0.92% | random_nz_ratio1 |
| `ground_truth_mrr@20` | 0.153521 | 0.154922 | -0.001401 | -0.90% | random_nz_ratio1 |
| `ground_truth_mrr@30` | 0.156354 | 0.157784 | -0.001430 | -0.91% | random_nz_ratio1 |

## Position Summary
| field | CEM bal20-ft500 | Random-NZ ratio1 |
|---|---:|---:|
| `total_fake_sessions` | 6499.0000 | 6499.0000 |
| `mean_absolute_position` | 2.3710 | 2.4662 |
| `median_absolute_position` | 1.0000 | 2.0000 |
| `pos1_pct` | 51.6695 | 48.6383 |
| `pos2_pct` | 14.1406 | 20.7878 |
| `pos3_pct` | 16.2025 | 11.6787 |
| `pos4_pos5_pct` | 10.3554 | 10.1708 |
| `pos6plus_pct` | 7.6319 | 8.7244 |
| `rank1_pct` | 51.6695 | 48.6383 |
| `rank2_pct` | 14.1406 | 20.7878 |
| `tail_pct` | 34.1899 | 30.5739 |

## Interpretation
- Surrogate absolute values are still optimistic relative to final SR-GNN victim values. For example surrogate `targeted_mrr@10` for CEM best is `0.222071`, while final victim `targeted_mrr@10` is `0.047576`.
- Ranking is aligned in this run: CEM has higher surrogate reward than Random-NZ under the same bal20/ft500 evaluator, and CEM also wins every final targeted metric against Random-NZ for SR-GNN target 5334.
- The tradeoff is utility damage: Random-NZ wins every ground-truth metric in this comparison, so the target attack is stronger but normal recommendation quality is slightly worse.

## Output Files
- Full report: `analysis/diagnosis_outputs/diginetica_cem_bal20_ft500_vs_random_nz_surrogate_reward/report.md`
- Full victim metric comparison CSV: `analysis/diagnosis_outputs/diginetica_cem_bal20_ft500_vs_random_nz_surrogate_reward/victim_metric_comparison.csv`
- Compact victim metric comparison CSV: `analysis/diagnosis_outputs/diginetica_cem_bal20_ft500_vs_random_nz_surrogate_reward/victim_metric_comparison_compact.csv`
- Surrogate ranking comparison CSV: `analysis/diagnosis_outputs/diginetica_cem_bal20_ft500_vs_random_nz_surrogate_reward/surrogate_ranking_comparison.csv`
- Surrogate rescore JSON: `analysis/diagnosis_outputs/diginetica_cem_bal20_ft500_vs_random_nz_surrogate_reward/bal20_ft500_random_nz_fixed_ratio_rescore.json`
