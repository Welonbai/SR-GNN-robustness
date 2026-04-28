# CEM vs Random-NZ Surrogate Reward Diagnosis

- config: `attack/configs/diginetica_attack_rank_bucket_cem.yaml`
- CEM run: `outputs/runs/diginetica/attack_rank_bucket_cem/run_group_cadb73910d`
- Random-NZ run: `outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_8679b974a1`
- clean surrogate: `outputs/surrogates/diginetica/clean_srgnn_surrogate_from_attack_a7fd31f6af.pt`
- targets: 5334, 11103

## Surrogate Reward Pairwise

| target | random_reward | cem_rescored_reward | cem-random | stored-rescored |
|---:|---:|---:|---:|---:|
| 5334 | 0.0000912975 | 0.0002475217 | 0.0001562242 | 0.0000000001 |
| 11103 | 0.0000711235 | 0.0001476918 | 0.0000765682 | 0.0000000001 |

## Readout

- CEM rescored surrogate reward beats Random-NZ on 2/2 target(s).
- On final targeted recall/mrr@10/20/30 across victims, CEM beats Random-NZ on 18/36 cell(s).
- If CEM reward wins here but final metrics do not, the main issue is reward/proxy misalignment.
- If Random-NZ reward wins here, the main issue is CEM search budget or policy parameterization.

## Output Files

- `surrogate_reward_comparison.csv`
- `surrogate_reward_pairwise.csv`
- `final_metric_overlap.csv`
- `manifest.json`
