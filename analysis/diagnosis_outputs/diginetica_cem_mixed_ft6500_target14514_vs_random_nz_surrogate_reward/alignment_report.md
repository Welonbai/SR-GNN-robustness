# Target 14514 ft6500 Surrogate-Final Alignment

- status: complete
- target_item: 14514
- fine_tune_steps: 6500
- elapsed_seconds: 281.9
- config: `attack\configs\diginetica_attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5.yaml`
- CEM run: `outputs\runs\diginetica\attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5\run_group_e4d7034404`
- Random-NZ run: `outputs\runs\diginetica\attack_random_nonzero_when_possible_ratio1_srgnn_sample5\run_group_720516397a`
- clean surrogate: `outputs\surrogates\diginetica\clean_srgnn_surrogate_from_attack_a7fd31f6af.pt`
- CEM reward source: ft6500 rescore of existing CEM best poisoned sessions
- Random reward source: ft6500 rescore of reconstructed Random-NZ sessions

## Candidate Scores

| method | actual_steps | actual_epochs | target_mean | gt_mean | T_MRR10 | T_MRR20 | T_MRR30 | T_R10 | T_R20 | T_R30 | GT_MRR10 | GT_MRR20 | GT_MRR30 | GT_R10 | GT_R20 | GT_R30 | total_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cem_best_rescored_ft6500 | 6500 | 1 | 0.0061347599 | 0.0277972174 | 0.049227 | 0.057994 | 0.062139 | 0.147847 | 0.277862 | 0.381719 | 0.133905 | 0.142129 | 0.145157 | 0.316891 | 0.436236 | 0.511519 | 280.2 |
| random_nz_ratio1_rescored_ft6500 | 6500 | 1 | 0.0072046642 | 0.0282707578 | 0.060072 | 0.069854 | 0.074351 | 0.174624 | 0.319164 | 0.431217 | 0.133480 | 0.141778 | 0.144848 | 0.315914 | 0.436121 | 0.512698 | 281.8 |

## Four-Case Result

| target | CEM reward | Random reward | surrogate delta | surrogate judgement | final wins 6 | final wins @10/@20 4 | final judgement | case |
|---:|---:|---:|---:|---|---:|---:|---|---|
| 14514 | 0.0061347599 | 0.0072046642 | -0.0010699043 | CEM < Random | 0 | 0 | CEM < Random | search_or_reference_gap |

## Output Files

- `surrogate_reward_comparison.csv`
- `surrogate_final_alignment_ft10000.csv`
- `final_metric_overlap.csv`
- `manifest.json`
