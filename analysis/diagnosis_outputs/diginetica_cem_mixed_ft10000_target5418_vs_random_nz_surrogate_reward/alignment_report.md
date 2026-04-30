# Target 5418 ft10000 Surrogate-Final Alignment

- status: complete
- target_item: 5418
- fine_tune_steps: 10000
- elapsed_seconds: 2699.1
- config: `attack\configs\diginetica_attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5.yaml`
- CEM run: `outputs\runs\diginetica\attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5\run_group_e4d7034404`
- Random-NZ run: `outputs\runs\diginetica\attack_random_nonzero_when_possible_ratio1_srgnn_sample5\run_group_720516397a`
- clean surrogate: `outputs\surrogates\diginetica\clean_srgnn_surrogate_from_attack_a7fd31f6af.pt`
- CEM reward source: ft10000 rescore of existing CEM best poisoned sessions
- Random reward source: ft10000 rescore of reconstructed Random-NZ sessions

## Candidate Scores

| method | target_mean | gt_mean | T_MRR10 | T_MRR20 | T_MRR30 | T_R10 | T_R20 | T_R30 | GT_MRR10 | GT_MRR20 | GT_MRR30 | GT_R10 | GT_R20 | GT_R30 | total_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cem_best_rescored_ft10000 | 0.0080201890 | 0.0280460323 | 0.063742 | 0.075271 | 0.080677 | 0.191018 | 0.362248 | 0.497210 | 0.133781 | 0.142088 | 0.145159 | 0.316949 | 0.437358 | 0.513690 | 1394.1 |
| random_nz_ratio1_rescored_ft10000 | 0.0112165083 | 0.0280284971 | 0.089333 | 0.101456 | 0.106130 | 0.257744 | 0.435862 | 0.552274 | 0.133326 | 0.141712 | 0.144750 | 0.316129 | 0.437703 | 0.513273 | 1304.7 |

## Four-Case Result

| target | CEM reward | Random reward | surrogate delta | surrogate judgement | final wins 6 | final wins @10/@20 4 | final judgement | case |
|---:|---:|---:|---:|---|---:|---:|---|---|
| 5418 | 0.0080201890 | 0.0112165083 | -0.0031963193 | CEM < Random | 0 | 0 | CEM < Random | search_or_reference_gap |

## Output Files

- `surrogate_reward_comparison.csv`
- `surrogate_final_alignment_ft10000.csv`
- `final_metric_overlap.csv`
- `manifest.json`
