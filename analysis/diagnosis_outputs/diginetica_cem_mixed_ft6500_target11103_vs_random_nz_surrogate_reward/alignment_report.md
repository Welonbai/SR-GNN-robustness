# Target 11103 ft6500 Surrogate-Final Alignment

- status: complete
- target_item: 11103
- fine_tune_steps: 6500
- elapsed_seconds: 282.6
- config: `attack\configs\diginetica_attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5.yaml`
- CEM run: `outputs\runs\diginetica\attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5\run_group_e4d7034404`
- Random-NZ run: `outputs\runs\diginetica\attack_random_nonzero_when_possible_ratio1_srgnn_sample5\run_group_720516397a`
- clean surrogate: `outputs\surrogates\diginetica\clean_srgnn_surrogate_from_attack_a7fd31f6af.pt`
- CEM reward source: ft6500 rescore of existing CEM best poisoned sessions
- Random reward source: ft6500 rescore of reconstructed Random-NZ sessions

## Candidate Scores

| method | actual_steps | actual_epochs | target_mean | gt_mean | T_MRR10 | T_MRR20 | T_MRR30 | T_R10 | T_R20 | T_R30 | GT_MRR10 | GT_MRR20 | GT_MRR30 | GT_R10 | GT_R20 | GT_R30 | total_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cem_best_rescored_ft6500 | 6500 | 1 | 0.0077119060 | 0.0291078771 | 0.056863 | 0.061603 | 0.062819 | 0.168584 | 0.235655 | 0.265438 | 0.133708 | 0.142174 | 0.145282 | 0.316302 | 0.439271 | 0.516466 | 277.6 |
| random_nz_ratio1_rescored_ft6500 | 6500 | 1 | 0.0076106082 | 0.0289491146 | 0.060072 | 0.071065 | 0.076157 | 0.185769 | 0.348558 | 0.475222 | 0.133941 | 0.142327 | 0.145381 | 0.316474 | 0.438566 | 0.514467 | 282.5 |

## Four-Case Result

| target | CEM reward | Random reward | surrogate delta | surrogate judgement | final wins 6 | final wins @10/@20 4 | final judgement | case |
|---:|---:|---:|---:|---|---:|---:|---|---|
| 11103 | 0.0077119060 | 0.0076106082 | 0.0001012978 | CEM > Random | 1 | 1 | CEM < Random | reward_final_misalignment |

## Output Files

- `surrogate_reward_comparison.csv`
- `surrogate_final_alignment_ft10000.csv`
- `final_metric_overlap.csv`
- `manifest.json`
