# CEM vs Random-NZ Surrogate Reward Diagnosis

- config: `attack/configs/diginetica_attack_rank_bucket_cem_mixed_ft2500_srgnn_target5334.yaml`
- CEM run: `outputs/runs/diginetica/attack_rank_bucket_cem_mixed_ft2500_srgnn_target5334/run_group_73b4185f37`
- Random-NZ run: `outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_8679b974a1`
- clean surrogate: `outputs/surrogates/diginetica/clean_srgnn_surrogate_from_attack_a7fd31f6af.pt`
- targets: 5334

## Surrogate Reward Pairwise

| target | random_reward | cem_rescored_reward | cem-random | stored-rescored |
|---:|---:|---:|---:|---:|
| 5334 | 0.0080052932 | 0.0099420917 | 0.0019367985 | -0.0000000052 |

## Readout

- CEM rescored surrogate reward beats Random-NZ on 1/1 target(s).
- On final targeted recall/mrr@10/20/30 across victims, CEM beats Random-NZ on 6/6 cell(s).
- If CEM reward wins here but final metrics do not, the main issue is reward/proxy misalignment.
- If Random-NZ reward wins here, the main issue is CEM search budget or policy parameterization.

## Output Files

- `surrogate_reward_comparison.csv`
- `surrogate_reward_pairwise.csv`
- `final_metric_overlap.csv`
- `manifest.json`
