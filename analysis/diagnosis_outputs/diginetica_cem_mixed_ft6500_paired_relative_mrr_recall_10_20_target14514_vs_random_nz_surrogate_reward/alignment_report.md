# Target 14514 ft6500 Surrogate-Final Alignment

- status: complete
- target_item: 14514
- fine_tune_steps: 6500
- reward_mode: `paired_relative_mrr_recall_10_20`
- elapsed_seconds: 411.9
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
| cem_best_rescored_ft6500 | 0.8417376155 | 0.053611 | 0.212855 | 6500 | 1 | 0.0061347599 | 0.0277972174 | 0.049227 | 0.057994 | 0.062139 | 0.147847 | 0.277862 | 0.381719 | 0.133905 | 0.142129 | 0.145157 | 0.316891 | 0.436236 | 0.511519 | 208.2 |
| random_nz_ratio1_rescored_ft6500 | 1.0000000000 | 0.064963 | 0.246894 | 6500 | 1 | 0.0072046642 | 0.0282707578 | 0.060072 | 0.069854 | 0.074351 | 0.174624 | 0.319164 | 0.431217 | 0.133480 | 0.141778 | 0.144848 | 0.315914 | 0.436121 | 0.512698 | 203.5 |

## Four-Case Result

| target | CEM reward | Random reward | surrogate delta | surrogate judgement | final wins 6 | final wins @10/@20 4 | final judgement | case |
|---:|---:|---:|---:|---|---:|---:|---|---|
| 14514 | 0.8417376155 | 1.0000000000 | -0.1582623845 | CEM < Random | 0 | 0 | CEM < Random | search_or_reference_gap |

## Output Files

- `surrogate_reward_comparison.csv`
- `surrogate_final_alignment_ft6500_paired_relative_mrr_recall_10_20.csv`
- `final_metric_overlap.csv`
- `manifest.json`
