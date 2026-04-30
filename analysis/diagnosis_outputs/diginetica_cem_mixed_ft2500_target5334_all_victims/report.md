# RankBucket-CEM mixed ft2500 target=5334 compact report

Dataset: Diginetica. Target item: `5334`. Existing artifacts only; no new experiment was run.

## 1. Run identifiers
| method | victim | run_group | path | target | reused_predictions | status |
|---|---|---|---|---:|---|---|
| Clean | srgnn | run_group_e0caef2757 | outputs/runs/diginetica/clean_run_no_attack/run_group_e0caef2757 | 5334 | True | completed |
| Clean | tron | run_group_e0caef2757 | outputs/runs/diginetica/clean_run_no_attack/run_group_e0caef2757 | 5334 | True | completed |
| Clean | miasrec | run_group_e0caef2757 | outputs/runs/diginetica/clean_run_no_attack/run_group_e0caef2757 | 5334 | True | completed |
| Random-NZ ratio1 | srgnn | run_group_720516397a | outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_720516397a | 5334 | True | completed |
| Random-NZ ratio1 | tron | run_group_720516397a | outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_720516397a | 5334 | True | completed |
| Random-NZ ratio1 | miasrec | run_group_720516397a | outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_720516397a | 5334 | True | completed |
| CEM ft2500 | srgnn | run_group_73b4185f37 | outputs/runs/diginetica/attack_rank_bucket_cem_mixed_ft2500_srgnn_target5334/run_group_73b4185f37 | 5334 | False | completed |
| CEM ft2500 | tron | run_group_73b4185f37 | outputs/runs/diginetica/attack_rank_bucket_cem_mixed_ft2500_srgnn_target5334/run_group_73b4185f37 | 5334 | False | completed |
| CEM ft2500 | miasrec | run_group_73b4185f37 | outputs/runs/diginetica/attack_rank_bucket_cem_mixed_ft2500_srgnn_target5334/run_group_73b4185f37 | 5334 | False | completed |

## 2. Minimal metric table
| victim | method | T_MRR10 | T_MRR20 | T_MRR30 | T_R10 | T_R20 | T_R30 | GT_MRR10 | GT_MRR20 | GT_MRR30 | GT_R10 | GT_R20 | GT_R30 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| srgnn | Clean | 0.000147 | 0.000163 | 0.000172 | 0.000460 | 0.000690 | 0.000920 | 0.147743 | 0.155969 | 0.158769 | 0.338674 | 0.457393 | 0.526899 |
| srgnn | Random-NZ ratio1 | 0.046666 | 0.052452 | 0.054943 | 0.128972 | 0.214121 | 0.276545 | 0.146679 | 0.154922 | 0.157784 | 0.336406 | 0.455372 | 0.526356 |
| srgnn | CEM ft2500 | 0.047947 | 0.053475 | 0.055938 | 0.135857 | 0.217638 | 0.279520 | 0.146995 | 0.155197 | 0.158021 | 0.337425 | 0.456555 | 0.526521 |
| tron | Clean | 0.000155 | 0.000177 | 0.000188 | 0.000411 | 0.000723 | 0.000986 | 0.128554 | 0.135381 | 0.137674 | 0.285484 | 0.384321 | 0.441273 |
| tron | Random-NZ ratio1 | 0.085562 | 0.088277 | 0.089448 | 0.154524 | 0.194272 | 0.223586 | 0.133176 | 0.140162 | 0.142546 | 0.298005 | 0.399077 | 0.458198 |
| tron | CEM ft2500 | 0.045842 | 0.048239 | 0.049264 | 0.095698 | 0.130418 | 0.155986 | 0.130198 | 0.136910 | 0.139222 | 0.286470 | 0.383581 | 0.441043 |
| miasrec | Clean | 0.000237 | 0.000254 | 0.000257 | 0.000723 | 0.000969 | 0.001035 | 0.179241 | 0.187491 | 0.190397 | 0.388166 | 0.507871 | 0.580055 |
| miasrec | Random-NZ ratio1 | 0.039443 | 0.059959 | 0.069401 | 0.191610 | 0.498636 | 0.732180 | 0.177512 | 0.185628 | 0.188460 | 0.380049 | 0.497601 | 0.568011 |
| miasrec | CEM ft2500 | 0.046240 | 0.065570 | 0.073039 | 0.221121 | 0.506918 | 0.691544 | 0.177814 | 0.185950 | 0.188762 | 0.381708 | 0.499589 | 0.569194 |

## 3. CEM ft2500 vs Random-NZ win counts
| victim | targeted selected | GT selected | combined selected | targeted full | GT full | combined full |
|---|---:|---:|---:|---:|---:|---:|
| srgnn | 6/6 (Random 0, ties 0) | 6/6 (Random 0, ties 0) | 12/12 (Random 0, ties 0) | 30/32 (Random 2, ties 0) | 30/32 (Random 2, ties 0) | 60/64 (Random 4, ties 0) |
| tron | 0/6 (Random 6, ties 0) | 0/6 (Random 6, ties 0) | 0/12 (Random 12, ties 0) | 0/32 (Random 32, ties 0) | 0/32 (Random 32, ties 0) | 0/64 (Random 64, ties 0) |
| miasrec | 5/6 (Random 1, ties 0) | 6/6 (Random 0, ties 0) | 11/12 (Random 1, ties 0) | 21/32 (Random 11, ties 0) | 32/32 (Random 0, ties 0) | 53/64 (Random 11, ties 0) |

## 4. Relative changes
| victim | CEM vs Random T_MRR30 | CEM vs Random T_R30 | CEM vs Random GT_MRR30 | CEM vs Random GT_R30 | CEM GT_MRR30 vs Clean | CEM GT_R30 vs Clean | Random GT_MRR30 vs Clean | Random GT_R30 vs Clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| srgnn | +1.81% | +1.08% | +0.15% | +0.03% | -0.47% | -0.07% | -0.62% | -0.10% |
| tron | -44.92% | -30.23% | -2.33% | -3.74% | +1.12% | -0.05% | +3.54% | +3.84% |
| miasrec | +5.24% | -5.55% | +0.16% | +0.21% | -0.86% | -1.87% | -1.02% | -2.08% |

## 5. Position distribution
| method | pos1_pct | pos2_pct | pos3_pct | pos4_5_pct | pos6_plus_pct | tail_pct | mean_abs_pos | median_abs_pos | unique_positions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CEM ft2500 | 51.93 | 8.62 | 18.37 | 12.57 | 8.51 | 39.45 | 2.50 | 1.00 | 22.00 |
| Random-NZ ratio1 | 48.64 | 20.79 | 11.68 | 10.17 | 8.72 | 30.57 | 2.47 | 2.00 | 26.00 |

## 6. Surrogate reward / CEM trace summary
| field | value |
|---|---:|
| best_iteration | 0 |
| best_candidate_id | 2 |
| best_surrogate_reward | 0.0099420864 |
| reward_metric_used | target_result.mean |
| fine_tune_steps | 2500 |
| actual_optimizer_steps_best_candidate | 2500 |
| CEM candidate count | 24 |
| validation_subset_effective_size | 69538 |

Surrogate rescore under the same ft2500 evaluator, where available:
| method | mean reward | T_MRR10 | T_MRR20 | T_MRR30 | T_R10 | T_R20 | T_R30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cem_best_rescored | 0.009942091660880864 | 0.08028345355993548 | n/a | n/a | 0.17446575972849376 | 0.2683424889988208 | n/a |
| random_nz_ratio1 | 0.008005293178737336 | 0.06555385156545673 | n/a | n/a | 0.14055624262992897 | 0.2201673904915298 | n/a |

## 7. Short interpretation
- `srgnn`: CEM beats Random-NZ on selected targeted 6/6, selected GT 6/6, full combined 60/64.
- `tron`: CEM beats Random-NZ on selected targeted 0/6, selected GT 0/6, full combined 0/64.
- `miasrec`: CEM beats Random-NZ on selected targeted 5/6, selected GT 6/6, full combined 53/64.
- Transfer is victim-dependent: CEM ft2500 transfers well to MiaSRec, but not to TRON. TRON is worse than Random-NZ on every available metric for target 5334.
- GT trade-off is victim-dependent: SR-GNN GT is slightly positive vs Random-NZ, MiaSRec GT is positive vs Random-NZ but still below Clean, and TRON GT is worse than Random-NZ.
- For target=5334, CEM ft2500 is stronger than Random-NZ overall across SR-GNN and MiaSRec, but not universally across victims. Caveats: single target, single CEM seed, and possible candidate sampling luck.
