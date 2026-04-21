# Prefix vs PosOptMVP Diagnosis: tron_pair

## What Was Analyzed

Same-victim case pair on TRON: failure case target 5334 where Prefix beats PosOptMVP, and success case target 11103 where PosOptMVP beats Prefix.

## Run Paths And Artifacts

- Shared fake-session pool: `outputs/shared/diginetica/attack/attack_shared_1c4345bfa3/fake_sessions.pkl`
- Prefix run group: `outputs/runs/diginetica/attack_prefix_nonzero_when_possible/run_group_14818d6dd6`
- PosOpt run group: `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46`
- Target 5334 Prefix artifacts:
  - `outputs/runs/diginetica/attack_prefix_nonzero_when_possible/run_group_14818d6dd6/targets/5334/position_stats.json`
  - `outputs/runs/diginetica/attack_prefix_nonzero_when_possible/run_group_14818d6dd6/targets/5334/prefix_nonzero_when_possible_metadata.pkl`
- Target 5334 PosOpt artifacts:
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/5334/position_stats.json`
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/5334/position_opt/selected_positions.json`
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/5334/position_opt/training_history.json`
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/5334/position_opt/run_metadata.json`
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/5334/position_opt/optimized_poisoned_sessions.pkl`
- Target 11103 Prefix artifacts:
  - `outputs/runs/diginetica/attack_prefix_nonzero_when_possible/run_group_14818d6dd6/targets/11103/position_stats.json`
  - `outputs/runs/diginetica/attack_prefix_nonzero_when_possible/run_group_14818d6dd6/targets/11103/prefix_nonzero_when_possible_metadata.pkl`
- Target 11103 PosOpt artifacts:
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/11103/position_stats.json`
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/11103/position_opt/selected_positions.json`
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/11103/position_opt/training_history.json`
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/11103/position_opt/run_metadata.json`
  - `outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46/targets/11103/position_opt/optimized_poisoned_sessions.pkl`

## High-Level Metric Comparison

| target_item | victim_model | metric_key | prefix_value | posopt_value | posopt_minus_prefix |
| --- | --- | --- | --- | --- | --- |
| 5334 | tron | targeted_mrr@10 | 0.117392 | 0.018313 | -0.099080 |
| 5334 | tron | targeted_recall@10 | 0.189885 | 0.048358 | -0.141526 |
| 11103 | tron | targeted_mrr@10 | 0.005495 | 0.056011 | 0.050517 |
| 11103 | tron | targeted_recall@10 | 0.013786 | 0.131766 | 0.117980 |

Generated chart:
- `plots/targeted_metrics_overview.png`

## High-Level Position Comparison

| target_item | method | mean_selected_position | median_selected_position | fraction_position0 | fraction_top10pct | fraction_top20pct |
| --- | --- | --- | --- | --- | --- | --- |
| 5334 | position_opt_mvp | 1.951223 | 1 | 0.293584 | 0.300662 | 0.357286 |
| 5334 | prefix_nonzero_when_possible | 0.343284 | 0 | 0.711340 | 0.769811 | 1 |
| 11103 | position_opt_mvp | 1.921834 | 1 | 0.299738 | 0.308971 | 0.367903 |
| 11103 | prefix_nonzero_when_possible | 0.329589 | 0 | 0.711340 | 0.778889 | 1 |

Position plots:
- `plots/final_position_histogram_target_5334.png`
- `plots/final_position_histogram_target_11103.png`
- `plots/normalized_position_histogram_target_5334.png`
- `plots/normalized_position_histogram_target_11103.png`

## High-Level PosOpt Training Dynamics

| target_item | step_count | initial_reward | final_reward | peak_reward | peak_reward_step | initial_mean_entropy | final_mean_entropy | entropy_drop | initial_average_selected_position | final_average_selected_position | final_fraction_position0 | final_fraction_top10pct | final_fraction_top20pct | final_max_position_share | final_unique_selected_positions | final_max_candidate_index_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | 30 | 0.000054 | 0.000098 | 0.000126 | 2 | 1.382516 | 1.335112 | 0.047403 | 1.947992 | 1.963533 | 0.290814 | 0.297738 | 0.351439 | 0.290814 | 25 | 0.290814 |
| 11103 | 30 | 0.000058 | 0.000034 | 0.000101 | 5 | 1.382516 | 1.341928 | 0.040588 | 1.924142 | 1.910602 | 0.296199 | 0.304816 | 0.362210 | 0.299277 | 26 | 0.299277 |

Training plots:
- `plots/posopt_reward_curve_target_5334.png`
- `plots/posopt_reward_curve_target_11103.png`
- `plots/posopt_baseline_curve_target_5334.png`
- `plots/posopt_baseline_curve_target_11103.png`
- `plots/posopt_advantage_curve_target_5334.png`
- `plots/posopt_advantage_curve_target_11103.png`
- `plots/posopt_mean_entropy_curve_target_5334.png`
- `plots/posopt_mean_entropy_curve_target_11103.png`
- `plots/posopt_average_selected_position_curve_target_5334.png`
- `plots/posopt_average_selected_position_curve_target_11103.png`

## High-Level Context Comparison

| target_item | shared_across_victims | victim_scope | session_count | fraction_same_selected_position | fraction_same_replaced_original_item | fraction_same_left_neighbor | fraction_same_right_neighbor | fraction_prefix_earlier | fraction_posopt_earlier | mean_abs_position_delta | validated_posopt_reconstruction_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | False | tron | 6499 | 0.289891 | 0.297123 | 0.291737 | 0.294199 | 0.669641 | 0.040468 | 1.696569 | 1 |
| 11103 | False | tron | 6499 | 0.302662 | 0.308663 | 0.304201 | 0.306355 | 0.660563 | 0.036775 | 1.673180 | 1 |

Preview of top replaced original items:

| target_item | method | rank | item_id | count | ratio_among_non_null |
| --- | --- | --- | --- | --- | --- |
| 5334 | position_opt_mvp | 1 | 3627 | 8 | 0.001231 |
| 5334 | position_opt_mvp | 2 | 549 | 7 | 0.001077 |
| 5334 | position_opt_mvp | 3 | 2792 | 7 | 0.001077 |
| 5334 | position_opt_mvp | 4 | 5725 | 7 | 0.001077 |
| 5334 | position_opt_mvp | 5 | 5 | 6 | 0.000923 |
| 5334 | position_opt_mvp | 6 | 490 | 6 | 0.000923 |
| 5334 | position_opt_mvp | 7 | 552 | 6 | 0.000923 |
| 5334 | position_opt_mvp | 8 | 636 | 6 | 0.000923 |
| 5334 | position_opt_mvp | 9 | 962 | 6 | 0.000923 |
| 5334 | position_opt_mvp | 10 | 2912 | 6 | 0.000923 |
| 5334 | position_opt_mvp | 11 | 3944 | 6 | 0.000923 |
| 5334 | position_opt_mvp | 12 | 4497 | 6 | 0.000923 |

## Candidate Hypotheses For Why Prefix Beats PosOpt Here

- For target 5334 on tron, Prefix wins while using a much stronger position-0 bias (0.711 vs 0.294). That supports an early-position-bias explanation more than a broad target-quality explanation.
- For target 5334 on tron, Prefix and PosOpt match on the exact selected position only 0.290 of the time, so different target-context compatibility is a plausible contributor.
- For target 5334, the PosOpt policy stays relatively diffuse (entropy drop 0.047, final max-position share 0.291), which is consistent with a policy training instability or weak concentration explanation.
- For target 11103 on tron, PosOpt improves targeted Recall@10 over Prefix by 0.118. This serves as a useful control case showing that broader position search can help when the target benefits from a less front-loaded replacement pattern.

## Output Files

- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/metrics_comparison.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/position_comparison.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/per_session_join.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/per_session_context_join.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/context_comparison.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/top_replaced_original_items.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/top_left_neighbors.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/top_right_neighbors.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/posopt_training_dynamics.csv`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/targeted_metrics_overview.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/final_position_histogram_target_5334.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/normalized_position_histogram_target_5334.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_reward_curve_target_5334.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_baseline_curve_target_5334.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_advantage_curve_target_5334.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_mean_entropy_curve_target_5334.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_average_selected_position_curve_target_5334.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/final_position_histogram_target_11103.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/normalized_position_histogram_target_11103.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_reward_curve_target_11103.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_baseline_curve_target_11103.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_advantage_curve_target_11103.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_mean_entropy_curve_target_11103.png`
- `analysis/diagnosis_outputs/prefix_vs_posopt/tron_case_pair_5334_vs_11103/plots/posopt_average_selected_position_curve_target_11103.png`
