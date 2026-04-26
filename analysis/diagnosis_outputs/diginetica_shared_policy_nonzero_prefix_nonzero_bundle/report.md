# SharedPolicy-local_context-NZ@1.0 vs Prefix-NZ@1.0 Bundle

## Scope
- Bundle type: artifact-only collection; no experiment reruns.
- Output directory: `analysis/diagnosis_outputs/diginetica_shared_policy_nonzero_prefix_nonzero_bundle`
- Core targets: 11103, 39588, 5334
- Victims: miasrec, srgnn, tron
- Metrics CSV rows: 2016
- Metric Ks present: 5, 10, 15, 20, 25, 30, 40, 50
- Metrics present: mrr, ndcg, precision, recall

## Experiment Identity
| method | run_root | run_group_key | dataset | target_cohort_key | target_items | victims | replacement_topk_ratio | nonzero_action_when_possible | policy_feature_set | reward_mode | final_policy_selection | deterministic_eval_every | deterministic_eval_include_final | attack_size | fake_session_count | clean_train_prefix_count | realized_poison_ratio | random_seeds_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prefix-NZ@1.0 | outputs/runs/diginetica/attack_prefix_nonzero_when_possible_ratio1/run_group_122a28bd27 | run_group_122a28bd27 | diginetica | target_cohort_8be070ab82 | 11103, 39588, 5334 | srgnn, miasrec, tron | 1 | True |  |  |  |  |  | 0.010000 | 6499 | 649932 | 0.010000 | {"fake_session_seed": 20260405, "position_opt_seed": 20260405, "surrogate_train_seed": 20260405, "target_selection_seed": 20260405, "victim_train_seed": 20260405} |
| SharedPolicy-local_context-NZ@1.0 | outputs/runs/diginetica/attack_position_opt_shared_policy_nonzero/run_group_0bce31ef52 | run_group_0bce31ef52 | diginetica | target_cohort_8be070ab82 | 11103, 39588, 5334 | srgnn, miasrec, tron | 1 | True | local_context | poisoned_target_utility | last | 0 | True | 0.010000 | 6499 | 649932 | 0.010000 | {"fake_session_seed": 20260405, "position_opt_seed": 20260405, "surrogate_train_seed": 20260405, "target_selection_seed": 20260405, "victim_train_seed": 20260405} |

## Included Methods In Metrics CSV
| method_key | method_label | run_group_key | replacement_topk_ratio | nonzero_action_when_possible | policy_feature_set | reward_mode |
| --- | --- | --- | --- | --- | --- | --- |
| clean | Clean | run_group_e0caef2757 | 0.200000 | False |  |  |
| dpsbr | DPSBR-random | run_group_7db577fb2e | 0.200000 | False |  |  |
| mvp | PosOptMVP@1.0 | run_group_3becc51c46 | 1 | False |  |  |
| prefix_r02 | Prefix-NZ@0.2 | run_group_14818d6dd6 | 0.200000 | True |  |  |
| prefix_r1 | Prefix-NZ@1.0 | run_group_122a28bd27 | 1 | True |  |  |
| sp | SharedPolicy-local_context@1.0 | run_group_c1835ab73f | 1 | False |  |  |
| sp_nz | SharedPolicy-local_context-NZ@1.0 | run_group_0bce31ef52 | 1 | True | local_context | poisoned_target_utility |

## Omitted Optional Methods
| label | reason |
| --- | --- |
| Random-NZ@1.0 | No completed run with replacement_topk_ratio=1.0 was found; only Random-NZ@0.2 is present. |

## Target Metadata
| target_item | popularity_bucket | train_frequency | test_frequency | popularity_rank |
| --- | --- | --- | --- | --- |
| 11103 | popular | 29 | 3 | 1 |
| 39588 | popular | 33 | 0 | 2 |
| 5334 | popular | 34 | 5 | 3 |

## Position Distribution Preview
| method | target_item | pos0_pct | pos1_pct | pos<=2_pct | mean_pos | median_pos |
| --- | --- | --- | --- | --- | --- | --- |
| Prefix-NZ@1.0 | 11103 | 0 | 56.516387 | 72.026466 | 2.348361 | 1 |
| SharedPolicy-local_context-NZ@1.0 | 11103 | 0 | 49.592245 | 68.918295 | 2.599323 | 2 |
| Prefix-NZ@1.0 | 39588 | 0 | 47.545776 | 68.656716 | 2.548700 | 2 |
| SharedPolicy-local_context-NZ@1.0 | 39588 | 0 | 57.624250 | 78.873673 | 1.916449 | 1 |
| Prefix-NZ@1.0 | 5334 | 0 | 50.915525 | 70.272350 | 2.411448 | 1 |
| SharedPolicy-local_context-NZ@1.0 | 5334 | 0 | 64.702262 | 84.689952 | 1.617172 | 1 |

## Shared Policy Training Summary
| target_item | final_reward | best_reward | best_reward_outer_step | initial_entropy | final_entropy | final_argmax_dominant_position | final_argmax_dominant_share_pct | final_pos0_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11103 | 0.000060 | 0.000150 | 29 | 1.000335 | 0.923731 | 1 | 49.592245 | 0 |
| 39588 | 0.000312 | 0.000413 | 10 | 1.000380 | 0.873694 | 1 | 57.624250 | 0 |
| 5334 | 0.000081 | 0.000188 | 2 | 1.004923 | 0.885106 | 1 | 64.702262 | 0 |

## Existing Signed-Change Color Renders
- The existing split-by-target signed-change render bundle is already available for the 5-method subset `{Clean, MVP, Prefix-NZ@1.0, SharedPolicy@1.0, SharedPolicy-NZ@1.0}`.
- `results/comparisons/diginetica_five_method_compare/five_method_metrics_split_by_target_item__target_item_11103/render.png`
- `results/comparisons/diginetica_five_method_compare/five_method_metrics_split_by_target_item__target_item_39588/render.png`
- `results/comparisons/diginetica_five_method_compare/five_method_metrics_split_by_target_item__target_item_5334/render.png`

## Generated Files
- `analysis/diagnosis_outputs/diginetica_shared_policy_nonzero_prefix_nonzero_bundle/report.md`
- `analysis/diagnosis_outputs/diginetica_shared_policy_nonzero_prefix_nonzero_bundle/metrics_long.csv`
- `analysis/diagnosis_outputs/diginetica_shared_policy_nonzero_prefix_nonzero_bundle/position_statistics.csv`
- `analysis/diagnosis_outputs/diginetica_shared_policy_nonzero_prefix_nonzero_bundle/shared_policy_training_diagnostics.json`
