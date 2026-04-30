# Target 5418 ft6500 Surrogate-Final Alignment

- status: complete
- target_item: 5418
- fine_tune_steps: 6500
- reward_mode: `paired_relative_mrr_recall_10_20`
- elapsed_seconds: 419.5
- config: `attack\configs\diginetica_attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5.yaml`
- CEM run: `outputs\runs\diginetica\attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5\run_group_e4d7034404`
- Random-NZ run: `outputs\runs\diginetica\attack_random_nonzero_when_possible_ratio1_srgnn_sample5\run_group_720516397a`
- clean surrogate: `outputs\surrogates\diginetica\clean_srgnn_surrogate_from_attack_a7fd31f6af.pt`
- CEM reward source: ft6500 rescore of existing CEM best poisoned sessions
- Random reward source: ft6500 rescore of reconstructed Random-NZ sessions
- paired-relative reward: each of targeted_mrr@10, targeted_mrr@20, targeted_recall@10, targeted_recall@20 is divided by max(CEM, Random) before averaging

## Candidate Scores

| method | objective_reward | mrr_part | recall_part | actual_steps | actual_epochs | target_mean | gt_mean | T_MRR10 | T_MRR20 | T_MRR30 | T_R10 | T_R20 | T_R30 | GT_MRR10 | GT_MRR20 | GT_MRR30 | GT_R10 | GT_R20 | GT_R30 | total_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cem_best_rescored_ft6500 | 0.9119749149 | 0.055457 | 0.223676 | 6500 | 1 | 0.0066270785 | 0.0308397883 | 0.050347 | 0.060567 | 0.065664 | 0.147488 | 0.299865 | 0.427292 | 0.135191 | 0.143482 | 0.146486 | 0.318243 | 0.438724 | 0.513388 | 212.3 |
| random_nz_ratio1_rescored_ft6500 | 1.0000000000 | 0.060599 | 0.242896 | 6500 | 1 | 0.0074495113 | 0.0306536361 | 0.055584 | 0.065615 | 0.070007 | 0.169102 | 0.316690 | 0.426486 | 0.134940 | 0.143188 | 0.146233 | 0.317913 | 0.437602 | 0.513259 | 207.1 |

## Four-Case Result

| target | CEM reward | Random reward | surrogate delta | surrogate judgement | final wins 6 | final wins @10/@20 4 | final judgement | case |
|---:|---:|---:|---:|---|---:|---:|---|---|
| 5418 | 0.9119749149 | 1.0000000000 | -0.0880250851 | CEM < Random | 0 | 0 | CEM < Random | search_or_reference_gap |

## Output Files

- `surrogate_reward_comparison.csv`
- `surrogate_final_alignment_ft6500_paired_relative_mrr_recall_10_20.csv`
- `final_metric_overlap.csv`
- `manifest.json`
