# CEM ft2500 SRGNN 5-Item Analysis

Scope: Diginetica, SRGNN victim only, 5 sampled popular target items, RankBucket-CEM normal mixed fine_tune_steps=2500, non-bal20. Experiments were not rerun; this report reads existing artifacts only.

## Artifacts Used
| method | run_group | run_root | resolved_config | targets | victim | attack_method | replacement_topk_ratio | nonzero | fine_tune_steps | cem_iters | cem_pop | candidate_count |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| clean | run_group_e0caef2757 | outputs\runs\diginetica\clean_run_no_attack_srgnn_sample5\run_group_e0caef2757 | outputs\runs\diginetica\clean_run_no_attack_srgnn_sample5\run_group_e0caef2757\resolved_config.json | 11103,39588,5334,5418,14514 | srgnn | clean |  |  |  |  |  |  |
| random-nz | run_group_720516397a | outputs\runs\diginetica\attack_random_nonzero_when_possible_ratio1_srgnn_sample5\run_group_720516397a | outputs\runs\diginetica\attack_random_nonzero_when_possible_ratio1_srgnn_sample5\run_group_720516397a\resolved_config.json | 11103,39588,5334,5418,14514 | srgnn | random_nonzero_when_possible | 1.0 | True |  |  |  |  |
| cem-ft2500 | run_group_e4d7034404 | outputs\runs\diginetica\attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5\run_group_e4d7034404 | outputs\runs\diginetica\attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5\run_group_e4d7034404\resolved_config.json | 11103,39588,5334,5418,14514 | srgnn | rank_bucket_cem | 1.0 | True | 2500 | 3 | 8 | 24 |

Seeds:
- clean: `{"fake_session_seed": 20260405, "position_opt_seed": 20260405, "surrogate_train_seed": 20260405, "target_selection_seed": 20260405, "victim_train_seed": 20260405}`
- random-nz: `{"fake_session_seed": 20260405, "position_opt_seed": 20260405, "surrogate_train_seed": 20260405, "target_selection_seed": 20260405, "victim_train_seed": 20260405}`
- cem-ft2500: `{"fake_session_seed": 20260405, "position_opt_seed": 20260405, "surrogate_train_seed": 20260405, "target_selection_seed": 20260405, "victim_train_seed": 20260405}`

Metric mapping: `targeted_mrr@K -> T_MRRK`, `targeted_recall@K -> T_RK`, `ground_truth_mrr@K -> GT_MRRK`, `ground_truth_recall@K -> GT_RK`.

## Main Metric Table
| target_item | method | T_MRR10 | T_MRR20 | T_MRR30 | T_R10 | T_R20 | T_R30 | GT_MRR10 | GT_MRR20 | GT_MRR30 | GT_R10 | GT_R20 | GT_R30 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11103 | clean | 0.000114 | 0.000132 | 0.000143 | 0.000345 | 0.000608 | 0.000887 | 0.147743 | 0.155969 | 0.158769 | 0.338674 | 0.457393 | 0.526899 |
| 11103 | random-nz | 0.047333 | 0.052994 | 0.055780 | 0.124174 | 0.208945 | 0.278468 | 0.143556 | 0.151741 | 0.154638 | 0.333810 | 0.452167 | 0.523941 |
| 11103 | cem-ft2500 | 0.046499 | 0.050525 | 0.052142 | 0.128792 | 0.187108 | 0.227398 | 0.144147 | 0.152440 | 0.155343 | 0.334286 | 0.454057 | 0.526258 |
| 39588 | clean | 0.000139 | 0.000168 | 0.000188 | 0.000444 | 0.000871 | 0.001347 | 0.147743 | 0.155969 | 0.158769 | 0.338674 | 0.457393 | 0.526899 |
| 39588 | random-nz | 0.047157 | 0.052592 | 0.055072 | 0.129120 | 0.209488 | 0.271156 | 0.146750 | 0.154888 | 0.157781 | 0.336406 | 0.454435 | 0.526274 |
| 39588 | cem-ft2500 | 0.050175 | 0.055558 | 0.057912 | 0.130928 | 0.210260 | 0.268888 | 0.148876 | 0.157057 | 0.159984 | 0.340070 | 0.458165 | 0.531039 |
| 5334 | clean | 0.000147 | 0.000163 | 0.000172 | 0.000460 | 0.000690 | 0.000920 | 0.147743 | 0.155969 | 0.158769 | 0.338674 | 0.457393 | 0.526899 |
| 5334 | random-nz | 0.046666 | 0.052452 | 0.054943 | 0.128972 | 0.214121 | 0.276545 | 0.146679 | 0.154922 | 0.157784 | 0.336406 | 0.455372 | 0.526356 |
| 5334 | cem-ft2500 | 0.047947 | 0.053475 | 0.055938 | 0.135857 | 0.217638 | 0.279520 | 0.146995 | 0.155197 | 0.158021 | 0.337425 | 0.456555 | 0.526521 |
| 5418 | clean | 0.000007 | 0.000009 | 0.000011 | 0.000033 | 0.000066 | 0.000115 | 0.147743 | 0.155969 | 0.158769 | 0.338674 | 0.457393 | 0.526899 |
| 5418 | random-nz | 0.049844 | 0.055890 | 0.058493 | 0.132308 | 0.220826 | 0.286010 | 0.145720 | 0.153950 | 0.156889 | 0.335174 | 0.454271 | 0.527441 |
| 5418 | cem-ft2500 | 0.047215 | 0.052364 | 0.054864 | 0.118719 | 0.195701 | 0.257830 | 0.146920 | 0.154918 | 0.157809 | 0.338394 | 0.454320 | 0.526258 |
| 14514 | clean | 0.000918 | 0.001125 | 0.001286 | 0.003500 | 0.006573 | 0.010697 | 0.147743 | 0.155969 | 0.158769 | 0.338674 | 0.457393 | 0.526899 |
| 14514 | random-nz | 0.050250 | 0.056221 | 0.058929 | 0.133770 | 0.221992 | 0.289494 | 0.146169 | 0.154316 | 0.157241 | 0.336948 | 0.454435 | 0.526981 |
| 14514 | cem-ft2500 | 0.046461 | 0.051519 | 0.053527 | 0.120773 | 0.194338 | 0.244323 | 0.146687 | 0.154909 | 0.157859 | 0.337803 | 0.456965 | 0.530103 |

## CEM ft2500 vs Random-NZ
| target_item | dT_MRR10 | dT_MRR20 | dT_MRR30 | dT_R10 | dT_R20 | dT_R30 | dGT_MRR10 | dGT_MRR20 | dGT_MRR30 | dGT_R10 | dGT_R20 | dGT_R30 | target_wins_6 | gt_wins_6 | combined_wins_12 | overall_label |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 11103 | -0.000834 | -0.002469 | -0.003638 | 0.004617 | -0.021838 | -0.051070 | 0.000590 | 0.000698 | 0.000705 | 0.000477 | 0.001890 | 0.002317 | 1 | 6 | 7 | fail |
| 39588 | 0.003018 | 0.002966 | 0.002839 | 0.001807 | 0.000772 | -0.002268 | 0.002126 | 0.002169 | 0.002203 | 0.003664 | 0.003730 | 0.004765 | 5 | 6 | 11 | strong_win |
| 5334 | 0.001280 | 0.001023 | 0.000994 | 0.006885 | 0.003516 | 0.002974 | 0.000316 | 0.000275 | 0.000237 | 0.001019 | 0.001183 | 0.000164 | 6 | 6 | 12 | strong_win |
| 5418 | -0.002629 | -0.003526 | -0.003629 | -0.013589 | -0.025124 | -0.028180 | 0.001199 | 0.000968 | 0.000921 | 0.003221 | 0.000049 | -0.001183 | 0 | 5 | 5 | fail |
| 14514 | -0.003789 | -0.004701 | -0.005401 | -0.012997 | -0.027655 | -0.045171 | 0.000518 | 0.000593 | 0.000618 | 0.000854 | 0.002530 | 0.003122 | 0 | 6 | 6 | fail |

Label rules: `strong_win` = at least 5/6 targeted and 5/6 GT selected metric wins; `target_win_gt_tradeoff` = at least 5/6 targeted but fewer than 5/6 GT wins; `mixed` = 3-4 targeted wins; `fail` = fewer than 3 targeted wins.

## Full Metric Win Rates
Metrics included: precision / recall / mrr / ndcg at K = 5, 10, 15, 20, 25, 30, 40, 50.
| target_item | targeted | targeted_random | gt | gt_random | combined | combined_random | ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11103 | 5/32 | 27/32 | 32/32 | 0/32 | 37/64 | 27/64 | 0 |
| 39588 | 24/32 | 8/32 | 32/32 | 0/32 | 56/64 | 8/64 | 0 |
| 5334 | 30/32 | 2/32 | 30/32 | 2/32 | 60/64 | 4/64 | 0 |
| 5418 | 0/32 | 32/32 | 28/32 | 4/32 | 28/64 | 36/64 | 0 |
| 14514 | 0/32 | 32/32 | 32/32 | 0/32 | 32/64 | 32/64 | 0 |
| AGGREGATE | 59/160 | 101/160 | 154/160 | 6/160 | 213/320 | 107/320 | 0 |

## Selected Delta Summary
| metric | mean_delta | median_delta | min_delta | max_delta |
| --- | ---: | ---: | ---: | ---: |
| T_MRR10 | -0.000591 | -0.000834 | -0.003789 | 0.003018 |
| T_MRR20 | -0.001342 | -0.002469 | -0.004701 | 0.002966 |
| T_MRR30 | -0.001767 | -0.003629 | -0.005401 | 0.002839 |
| T_R10 | -0.002655 | 0.001807 | -0.013589 | 0.006885 |
| T_R20 | -0.014066 | -0.021838 | -0.027655 | 0.003516 |
| T_R30 | -0.024743 | -0.028180 | -0.051070 | 0.002974 |
| GT_MRR10 | 0.000950 | 0.000590 | 0.000316 | 0.002126 |
| GT_MRR20 | 0.000941 | 0.000698 | 0.000275 | 0.002169 |
| GT_MRR30 | 0.000937 | 0.000705 | 0.000237 | 0.002203 |
| GT_R10 | 0.001847 | 0.001019 | 0.000477 | 0.003664 |
| GT_R20 | 0.001876 | 0.001890 | 0.000049 | 0.003730 |
| GT_R30 | 0.001837 | 0.002317 | -0.001183 | 0.004765 |

## GT Signed Change vs Clean
| target_item | method | GT_MRR10_vs_clean | GT_MRR20_vs_clean | GT_MRR30_vs_clean | GT_R10_vs_clean | GT_R20_vs_clean | GT_R30_vs_clean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11103 | random-nz | -2.83% | -2.71% | -2.60% | -1.44% | -1.14% | -0.56% |
| 11103 | cem-ft2500 | -2.43% | -2.26% | -2.16% | -1.30% | -0.73% | -0.12% |
| 39588 | random-nz | -0.67% | -0.69% | -0.62% | -0.67% | -0.65% | -0.12% |
| 39588 | cem-ft2500 | 0.77% | 0.70% | 0.77% | 0.41% | 0.17% | 0.79% |
| 5334 | random-nz | -0.72% | -0.67% | -0.62% | -0.67% | -0.44% | -0.10% |
| 5334 | cem-ft2500 | -0.51% | -0.50% | -0.47% | -0.37% | -0.18% | -0.07% |
| 5418 | random-nz | -1.37% | -1.29% | -1.18% | -1.03% | -0.68% | 0.10% |
| 5418 | cem-ft2500 | -0.56% | -0.67% | -0.60% | -0.08% | -0.67% | -0.12% |
| 14514 | random-nz | -1.07% | -1.06% | -0.96% | -0.51% | -0.65% | 0.02% |
| 14514 | cem-ft2500 | -0.72% | -0.68% | -0.57% | -0.26% | -0.09% | 0.61% |

## CEM Diagnostics
| target_item | best_iter | best_cand | best_reward | second_best | margin | margin_label | final_selected | candidates | actual_steps | cem_time_min | ft_sum_min | ft_mean_s | phase | improved_after_iter0 | iter_best_rewards |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 11103 | 2 | 2 | 0.009938 | 0.006804 | 0.003134 | large | iter2_cand2 | 24 | 2500 | 45.03 | 39.71 | 99.27 | last | True | iter0:0.00680355; iter1:0.00625704; iter2:0.00993767 |
| 39588 | 1 | 7 | 0.008415 | 0.008143 | 0.000272 | small | iter1_cand7 | 24 | 2500 | 43.58 | 38.22 | 95.56 | middle | True | iter0:0.00814327; iter1:0.00841497; iter2:0.00762922 |
| 5334 | 0 | 2 | 0.009942 | 0.009812 | 0.000131 | small | iter0_cand2 | 24 | 2500 | 40.98 | 35.88 | 89.69 | early | False | iter0:0.00994209; iter1:0.00835878; iter2:0.00981152 |
| 5418 | 1 | 2 | 0.004673 | 0.004350 | 0.000323 | moderate | iter1_cand2 | 24 | 2500 | 40.39 | 35.50 | 88.75 | middle | True | iter0:0.00326778; iter1:0.00467314; iter2:0.00433658 |
| 14514 | 0 | 6 | 0.006718 | 0.006693 | 0.000026 | small | iter0_cand6 | 24 | 2500 | 40.35 | 35.43 | 88.58 | early | False | iter0:0.00671836; iter1:0.00669257; iter2:0.00632635 |

`cem_time_min` is the sum of per-candidate `candidate_total_seconds` from `cem_trace.jsonl`; it is a trace-level candidate evaluation total, not necessarily full wall-clock including setup and final victim training.

## Position Distribution
| target_item | method | pos1_pct | pos2_pct | pos3_pct | pos4_5_pct | pos6plus_pct | tail_pct | mean_position | unique_positions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11103 | random-nz | 48.64% | 20.79% | 11.68% | 10.17% | 8.72% | 30.57% | 2.466 | 26 |
| 11103 | cem-ft2500 | 85.38% | 11.79% | 1.28% | 0.95% | 0.60% | 2.83% | 1.215 | 13 |
| 39588 | random-nz | 48.64% | 20.79% | 11.68% | 10.17% | 8.72% | 30.57% | 2.466 | 26 |
| 39588 | cem-ft2500 | 40.87% | 30.25% | 13.37% | 9.40% | 6.11% | 28.88% | 2.339 | 22 |
| 5334 | random-nz | 48.64% | 20.79% | 11.68% | 10.17% | 8.72% | 30.57% | 2.466 | 26 |
| 5334 | cem-ft2500 | 51.93% | 8.62% | 18.37% | 12.57% | 8.51% | 39.45% | 2.497 | 22 |
| 5418 | random-nz | 48.64% | 20.79% | 11.68% | 10.17% | 8.72% | 30.57% | 2.466 | 26 |
| 5418 | cem-ft2500 | 35.11% | 25.87% | 18.13% | 12.49% | 8.40% | 39.02% | 2.654 | 24 |
| 14514 | random-nz | 48.64% | 20.79% | 11.68% | 10.17% | 8.72% | 30.57% | 2.466 | 26 |
| 14514 | cem-ft2500 | 52.75% | 40.64% | 3.14% | 2.03% | 1.45% | 6.62% | 1.650 | 18 |

Random-NZ position buckets are computed from `position_stats.json` absolute position counts. CEM buckets are read from `final_position_summary.json`; for both, `tail_pct` is positions >= 3.

## Concise Conclusion

### Does CEM ft2500 beat Random-NZ on target metrics?
CEM wins targeted selected metrics on counts [1, 5, 6, 0, 0] out of 6 across targets [11103, 39588, 5334, 5418, 14514]. It is not uniformly strong: 2/5 targets are `strong_win`, and target failures are [11103, 5418, 14514]. Full targeted metrics aggregate: CEM 59/160 vs Random 101/160, ties 0.

### Does CEM ft2500 preserve GT metrics?
GT selected wins per target are [6, 6, 6, 5, 6]/6. CEM preserves/improves GT better than Random-NZ on most selected metrics for 5/5 targets by the >=4/6 GT-win heuristic. Full GT aggregate: CEM 154/160 vs Random 6/160, ties 0.

### Is the result stable across 5 target items?
No. The paired tables show item-level variation; macro averages alone would hide failed or mixed targets.

### Is there any catastrophic failure item?
By the requested `<3 targeted wins out of 6` rule: 11103, 5418, 14514. Still inspect `mixed` labels before claiming robustness.

### Do best candidates mostly appear in early, middle, or late CEM iterations?
Best candidate phase counts: {'last': 1, 'middle': 2, 'early': 2}. This indicates whether the 3-iteration search is still improving late or mostly finding winners early.

### Should we keep ft2500, increase iterations/population, or avoid changing parameters for now?
Based on these artifacts, do not change parameters only from macro-average gains. Keep ft2500 as the current candidate setting for comparison, but use paired per-target results to decide whether to increase iterations/population. If best candidates are often early and later iterations do not improve reward, increasing population may be more informative than increasing iterations; if best candidates are late, more iterations may be worth testing. Avoid claiming robust superiority until failures/mixed targets are resolved or replicated with more seeds/items.

## Output Files
- `analysis\reports\cem_ft2500_srg_5item_analysis.md`
- `analysis\reports\cem_ft2500_srg_5item_main_metrics.csv`
- `analysis\reports\cem_ft2500_srg_5item_comparison.csv`
- `analysis\reports\cem_ft2500_srg_5item_full_win_rates.csv`
- `analysis\reports\cem_ft2500_srg_5item_gt_change_vs_clean.csv`
- `analysis\reports\cem_ft2500_srg_5item_cem_diagnostics.csv`
- `analysis\reports\cem_ft2500_srg_5item_position_distribution.csv`
- `analysis\reports\cem_ft2500_srg_5item_delta_stats.csv`
