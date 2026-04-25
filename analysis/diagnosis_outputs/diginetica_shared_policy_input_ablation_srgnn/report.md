# diginetica_shared_policy_input_ablation_srgnn

## Scope
- Dataset: `diginetica`
- Victim: `srgnn`
- Selected targets: 5334, 11103
- Blank cells indicate unavailable method-target artifacts.
- Authoritative final metrics come from `summary_current.json`.
- Reference method: `Shared Policy local_context`

## A. Final Metrics Comparison
| target_item | method | target_recall@10 | target_recall@20 | target_recall@30 | target_mrr@10 | target_mrr@20 | target_mrr@30 | gt_recall@10 | gt_recall@20 | gt_recall@30 | gt_mrr@10 | gt_mrr@20 | gt_mrr@30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | Clean | 0.000460 | 0.000690 | 0.000920 | 0.000147 | 0.000163 | 0.000172 | 0.338674 | 0.457393 | 0.526899 | 0.147743 | 0.155969 | 0.158769 |
| 5334 | Prefix-NZ | 0.044316 | 0.076161 | 0.104752 | 0.016330 | 0.018483 | 0.019627 | 0.335256 | 0.452841 | 0.524040 | 0.145870 | 0.154003 | 0.156867 |
| 5334 | PosOptMVP | 0.091278 | 0.148805 | 0.198626 | 0.034332 | 0.038208 | 0.040211 | 0.336077 | 0.452315 | 0.524910 | 0.145861 | 0.153863 | 0.156787 |
| 5334 | Shared Policy local_context | 0.117207 | 0.192711 | 0.252259 | 0.045729 | 0.050835 | 0.053196 | 0.336012 | 0.451789 | 0.523859 | 0.146477 | 0.154468 | 0.157373 |
| 5334 | Shared Policy normalized_position_only | 0.000016 | 0.000016 | 0.000033 | 0.000002 | 0.000002 | 0.000002 | 0.335650 | 0.451296 | 0.521624 | 0.144858 | 0.152852 | 0.155683 |
| 5334 | Shared Policy target_normalized_position | 0.116484 | 0.188241 | 0.244816 | 0.045914 | 0.050747 | 0.053007 | 0.336899 | 0.454156 | 0.525486 | 0.145789 | 0.153888 | 0.156754 |
| 5334 | Shared Policy target_original_normalized_position | 0.087712 | 0.149676 | 0.201600 | 0.033213 | 0.037398 | 0.039464 | 0.337162 | 0.454188 | 0.526011 | 0.146288 | 0.154370 | 0.157262 |
| 5334 | Shared Policy full_context_normalized_position | 0.083752 | 0.141017 | 0.193434 | 0.031622 | 0.035515 | 0.037612 | 0.336899 | 0.453827 | 0.525683 | 0.145412 | 0.153482 | 0.156380 |
| 11103 | Clean | 0.000345 | 0.000608 | 0.000887 | 0.000114 | 0.000132 | 0.000143 | 0.338674 | 0.457393 | 0.526899 | 0.147743 | 0.155969 | 0.158769 |
| 11103 | Prefix-NZ | 0.043084 | 0.074633 | 0.098738 | 0.016262 | 0.018442 | 0.019403 | 0.330556 | 0.450491 | 0.522741 | 0.142886 | 0.151206 | 0.154119 |
| 11103 | PosOptMVP | 0.098015 | 0.165977 | 0.224013 | 0.037822 | 0.042412 | 0.044720 | 0.332347 | 0.450705 | 0.521345 | 0.143244 | 0.151423 | 0.154269 |
| 11103 | Shared Policy local_context | 0.015265 | 0.026094 | 0.037793 | 0.007312 | 0.008036 | 0.008501 | 0.332052 | 0.450738 | 0.521575 | 0.142167 | 0.150373 | 0.153225 |
| 11103 | Shared Policy normalized_position_only | 0.000016 | 0.000033 | 0.000033 | 0.000005 | 0.000006 | 0.000006 | 0.330441 | 0.448372 | 0.519537 | 0.142293 | 0.150449 | 0.153313 |
| 11103 | Shared Policy target_normalized_position | 0.000016 | 0.000033 | 0.000033 | 0.000005 | 0.000006 | 0.000006 | 0.330441 | 0.448372 | 0.519537 | 0.142293 | 0.150449 | 0.153313 |
| 11103 | Shared Policy target_original_normalized_position | 0.084525 | 0.145190 | 0.194321 | 0.033205 | 0.037347 | 0.039303 | 0.334023 | 0.452512 | 0.523464 | 0.142893 | 0.151078 | 0.153931 |
| 11103 | Shared Policy full_context_normalized_position | 0.026932 | 0.046912 | 0.067238 | 0.011932 | 0.013275 | 0.014086 | 0.330984 | 0.448372 | 0.521723 | 0.143329 | 0.151456 | 0.154415 |

## B. Delta vs Reference
| target_item | method | reference_method | target_recall@10 | target_recall@20 | target_recall@30 | target_mrr@10 | target_mrr@20 | target_mrr@30 | gt_recall@10 | gt_recall@20 | gt_recall@30 | gt_mrr@10 | gt_mrr@20 | gt_mrr@30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | Clean | Shared Policy local_context | -0.116747 | -0.192021 | -0.251339 | -0.045582 | -0.050672 | -0.053024 | 0.002662 | 0.005603 | 0.003040 | 0.001267 | 0.001502 | 0.001395 |
| 5334 | Prefix-NZ | Shared Policy local_context | -0.072891 | -0.116550 | -0.147507 | -0.029399 | -0.032352 | -0.033569 | -0.000756 | 0.001052 | 0.000181 | -0.000606 | -0.000465 | -0.000506 |
| 5334 | PosOptMVP | Shared Policy local_context | -0.025929 | -0.043905 | -0.053633 | -0.011398 | -0.012627 | -0.012985 | 0.000066 | 0.000526 | 0.001052 | -0.000616 | -0.000604 | -0.000586 |
| 5334 | Shared Policy normalized_position_only | Shared Policy local_context | -0.117191 | -0.192694 | -0.252226 | -0.045728 | -0.050833 | -0.053194 | -0.000361 | -0.000493 | -0.002235 | -0.001619 | -0.001615 | -0.001691 |
| 5334 | Shared Policy target_normalized_position | Shared Policy local_context | -0.000723 | -0.004469 | -0.007444 | 0.000185 | -0.000088 | -0.000189 | 0.000887 | 0.002366 | 0.001627 | -0.000688 | -0.000579 | -0.000619 |
| 5334 | Shared Policy target_original_normalized_position | Shared Policy local_context | -0.029495 | -0.043035 | -0.050659 | -0.012516 | -0.013437 | -0.013732 | 0.001150 | 0.002399 | 0.002153 | -0.000189 | -0.000098 | -0.000111 |
| 5334 | Shared Policy full_context_normalized_position | Shared Policy local_context | -0.033455 | -0.051694 | -0.058825 | -0.014108 | -0.015320 | -0.015584 | 0.000887 | 0.002038 | 0.001824 | -0.001065 | -0.000985 | -0.000993 |
| 11103 | Clean | Shared Policy local_context | -0.014920 | -0.025486 | -0.036906 | -0.007199 | -0.007904 | -0.008358 | 0.006622 | 0.006655 | 0.005324 | 0.005577 | 0.005596 | 0.005543 |
| 11103 | Prefix-NZ | Shared Policy local_context | 0.027819 | 0.048539 | 0.060945 | 0.008950 | 0.010406 | 0.010902 | -0.001495 | -0.000246 | 0.001167 | 0.000719 | 0.000833 | 0.000893 |
| 11103 | PosOptMVP | Shared Policy local_context | 0.082750 | 0.139883 | 0.186220 | 0.030509 | 0.034376 | 0.036220 | 0.000296 | -0.000033 | -0.000230 | 0.001077 | 0.001050 | 0.001044 |
| 11103 | Shared Policy normalized_position_only | Shared Policy local_context | -0.015249 | -0.026061 | -0.037760 | -0.007307 | -0.008030 | -0.008494 | -0.001610 | -0.002366 | -0.002038 | 0.000126 | 0.000075 | 0.000088 |
| 11103 | Shared Policy target_normalized_position | Shared Policy local_context | -0.015249 | -0.026061 | -0.037760 | -0.007307 | -0.008030 | -0.008494 | -0.001610 | -0.002366 | -0.002038 | 0.000126 | 0.000075 | 0.000088 |
| 11103 | Shared Policy target_original_normalized_position | Shared Policy local_context | 0.069260 | 0.119097 | 0.156528 | 0.025892 | 0.029311 | 0.030802 | 0.001972 | 0.001775 | 0.001890 | 0.000726 | 0.000705 | 0.000706 |
| 11103 | Shared Policy full_context_normalized_position | Shared Policy local_context | 0.011667 | 0.020819 | 0.029446 | 0.004619 | 0.005239 | 0.005585 | -0.001068 | -0.002366 | 0.000148 | 0.001162 | 0.001083 | 0.001189 |

## C. Final Position Summary
| target_item | method | total | unique_positions | dominant_position | dominant_pct | pos0_pct | pos<=1_pct | pos<=2_pct | pos<=5_pct | top5_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | Clean |  |  |  |  |  |  |  |  |  |
| 5334 | Prefix-NZ | 6499 | 8 | 0 | 71.134021 | 71.134021 | 95.814741 | 99.076781 | 99.969226 | 0:71.13%, 1:24.68%, 2:3.26%, 3:0.68%, 4:0.18% |
| 5334 | PosOptMVP | 6499 | 25 | 0 | 29.358363 | 29.358363 | 58.362825 | 73.872904 | 92.106478 | 0:29.36%, 1:29.00%, 2:15.51%, 3:9.40%, 4:5.25% |
| 5334 | Shared Policy local_context | 6499 | 27 | 1 | 33.774427 | 2.692722 | 36.467149 | 57.039545 | 85.459301 | 1:33.77%, 2:20.57%, 3:13.56%, 4:8.97%, 5:5.89% |
| 5334 | Shared Policy normalized_position_only | 6499 | 1 | 0 | 100 | 100 | 100 | 100 | 100 | 0:100.00% |
| 5334 | Shared Policy target_normalized_position | 6499 | 32 | 1 | 28.096630 | 0 | 28.096630 | 47.699646 | 77.935067 | 1:28.10%, 2:19.60%, 3:13.42%, 4:10.02%, 5:6.80% |
| 5334 | Shared Policy target_original_normalized_position | 6499 | 26 | 0 | 31.035544 | 31.035544 | 59.916910 | 74.734575 | 92.091091 | 0:31.04%, 1:28.88%, 2:14.82%, 3:8.96%, 4:5.34% |
| 5334 | Shared Policy full_context_normalized_position | 6499 | 26 | 1 | 31.851054 | 30.389291 | 62.240345 | 77.504231 | 93.737498 | 1:31.85%, 0:30.39%, 2:15.26%, 3:8.28%, 4:5.08% |
| 11103 | Clean |  |  |  |  |  |  |  |  |  |
| 11103 | Prefix-NZ | 6499 | 7 | 0 | 71.134021 | 71.134021 | 96.922603 | 99.230651 | 99.984613 | 0:71.13%, 1:25.79%, 2:2.31%, 3:0.58%, 4:0.14% |
| 11103 | PosOptMVP | 6499 | 23 | 0 | 29.973842 | 29.973842 | 59.039852 | 74.349900 | 92.229574 | 0:29.97%, 1:29.07%, 2:15.31%, 3:8.89%, 4:5.65% |
| 11103 | Shared Policy local_context | 6499 | 12 | 0 | 92.014156 | 92.014156 | 95.737806 | 97.414987 | 99.461456 | 0:92.01%, 1:3.72%, 2:1.68%, 3:1.03%, 4:0.62% |
| 11103 | Shared Policy normalized_position_only | 6499 | 1 | 0 | 100 | 100 | 100 | 100 | 100 | 0:100.00% |
| 11103 | Shared Policy target_normalized_position | 6499 | 1 | 0 | 100 | 100 | 100 | 100 | 100 | 0:100.00% |
| 11103 | Shared Policy target_original_normalized_position | 6499 | 24 | 0 | 34.543776 | 34.543776 | 62.517310 | 77.181105 | 93.968303 | 0:34.54%, 1:27.97%, 2:14.66%, 3:8.26%, 4:5.68% |
| 11103 | Shared Policy full_context_normalized_position | 6499 | 23 | 0 | 81.104785 | 81.104785 | 87.505770 | 91.290968 | 96.861056 | 0:81.10%, 1:6.40%, 2:3.79%, 3:2.69%, 4:1.69% |

## D. Verification Summary
| target_item | method | policy_feature_set | active_item_features | active_scalar_features | policy_input_dim | policy_embedding_dim | policy_hidden_dim | prefix_score_enabled | run_metadata_present | training_history_present | learned_logits_present | expected_policy_feature_set | expected_active_item_features | expected_active_scalar_features | expected_input_dim_from_features | feature_set_matches_expected | active_features_match_expected | input_dim_matches_expected | prefix_flag_matches_expected | artifact_consistency_status | verification_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | Shared Policy local_context |  |  |  |  | 16 | 32 |  | True | True | True | local_context | ['target_item', 'original_item', 'left_item', 'right_item'] | ['position_index', 'normalized_position', 'session_length'] | 67 |  |  |  |  | legacy_missing_fields | legacy_missing_fields |
| 5334 | Shared Policy normalized_position_only | normalized_position_only | [] | ['normalized_position'] | 1 | 16 | 32 | False | True | True | True | normalized_position_only | [] | ['normalized_position'] | 1 | True | True | True | True | ok | ok |
| 5334 | Shared Policy target_normalized_position | target_normalized_position | ['target_item'] | ['normalized_position'] | 17 | 16 | 32 | False | True | True | True | target_normalized_position | ['target_item'] | ['normalized_position'] | 17 | True | True | True | True | ok | ok |
| 5334 | Shared Policy target_original_normalized_position | target_original_normalized_position | ['target_item', 'original_item'] | ['normalized_position'] | 33 | 16 | 32 | False | True | True | True | target_original_normalized_position | ['target_item', 'original_item'] | ['normalized_position'] | 33 | True | True | True | True | ok | ok |
| 5334 | Shared Policy full_context_normalized_position | full_context_normalized_position | ['target_item', 'original_item', 'left_item', 'right_item'] | ['normalized_position'] | 65 | 16 | 32 | False | True | True | True | full_context_normalized_position | ['target_item', 'original_item', 'left_item', 'right_item'] | ['normalized_position'] | 65 | True | True | True | True | ok | ok |
| 11103 | Shared Policy local_context |  |  |  |  | 16 | 32 |  | True | True | True | local_context | ['target_item', 'original_item', 'left_item', 'right_item'] | ['position_index', 'normalized_position', 'session_length'] | 67 |  |  |  |  | legacy_missing_fields | legacy_missing_fields |
| 11103 | Shared Policy normalized_position_only | normalized_position_only | [] | ['normalized_position'] | 1 | 16 | 32 | False | True | True | True | normalized_position_only | [] | ['normalized_position'] | 1 | True | True | True | True | ok | ok |
| 11103 | Shared Policy target_normalized_position | target_normalized_position | ['target_item'] | ['normalized_position'] | 17 | 16 | 32 | False | True | True | True | target_normalized_position | ['target_item'] | ['normalized_position'] | 17 | True | True | True | True | ok | ok |
| 11103 | Shared Policy target_original_normalized_position | target_original_normalized_position | ['target_item', 'original_item'] | ['normalized_position'] | 33 | 16 | 32 | False | True | True | True | target_original_normalized_position | ['target_item', 'original_item'] | ['normalized_position'] | 33 | True | True | True | True | ok | ok |
| 11103 | Shared Policy full_context_normalized_position | full_context_normalized_position | ['target_item', 'original_item', 'left_item', 'right_item'] | ['normalized_position'] | 65 | 16 | 32 | False | True | True | True | full_context_normalized_position | ['target_item', 'original_item', 'left_item', 'right_item'] | ['normalized_position'] | 65 | True | True | True | True | ok | ok |

## E. Training Final-Step Summary
| method | target_item | outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty | entropy_drop | final_dominant_position | final_dominant_share_pct | final_unique_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PosOptMVP | 5334 | 29 | 0.000098 | 0.000075 | 0.000023 | 1.335112 | 0.196493 | 0.000098 | 1.963533 | 1 | 0.290814 | 0.580551 | 0.741806 | 0 | 0 | 0.047403 | 0 | 29.081397 | 25 |
| Shared Policy local_context | 5334 | 29 | 0.000112 | 0.000091 | 0.000021 | 1.151750 | 0.157789 | 0.000112 | 2.409140 | 2 | 0.138637 | 0.486229 | 0.679643 | 0 | 0 | 0.209923 | 1 | 34.759194 | 25 |
| Shared Policy normalized_position_only | 5334 | 29 | 0.000089 | 0.000070 | 0.000018 | 1.376627 | 0.162384 | 0.000089 | 1.825512 | 1 | 0.326358 | 0.612402 | 0.763964 | 0 | 0 | 0.005522 | 0 | 32.635790 | 24 |
| Shared Policy target_normalized_position | 5334 | 29 | 0.000095 | 0.000076 | 0.000020 | 1.382479 | 0.175859 | 0.000095 | 1.976612 | 1 | 0.286352 | 0.582551 | 0.742884 | 0 | 0 | 0.000035 | 1 | 29.619942 | 25 |
| Shared Policy target_original_normalized_position | 5334 | 29 | 0.000091 | 0.000075 | 0.000016 | 1.315922 | 0.133087 | 0.000091 | 1.934298 | 1 | 0.292968 | 0.589475 | 0.749038 | 0 | 0 | 0.062415 | 1 | 29.650715 | 25 |
| Shared Policy full_context_normalized_position | 5334 | 29 | 0.000097 | 0.000081 | 0.000016 | 1.340021 | 0.137984 | 0.000097 | 1.941529 | 1 | 0.284044 | 0.589321 | 0.749038 | 0 | 0 | 0.029999 | 1 | 30.527774 | 25 |
| PosOptMVP | 11103 | 29 | 0.000034 | 0.000049 | -0.000014 | 1.341928 | -0.122388 | 0.000034 | 1.910602 | 1 | 0.296199 | 0.595476 | 0.741191 | 0 | 0 | 0.040588 | 1 | 29.927681 | 26 |
| Shared Policy local_context | 11103 | 29 | 0.000023 | 0.000032 | -0.000010 | 0.476913 | -0.031139 | 0.000023 | 0.470380 | 0 | 0.823050 | 0.896599 | 0.931990 | 0 | 0 | 0.892516 | 0 | 82.304970 | 17 |
| Shared Policy normalized_position_only | 11103 | 29 | 0.000027 | 0.000046 | -0.000020 | 1.364328 | -0.172527 | 0.000027 | 1.635329 | 1 | 0.360517 | 0.655485 | 0.784736 | 0 | 0 | 0.017940 | 0 | 36.051700 | 26 |
| Shared Policy target_normalized_position | 11103 | 29 | 0.000029 | 0.000048 | -0.000019 | 1.379221 | -0.173265 | 0.000029 | 1.785044 | 1 | 0.321742 | 0.624404 | 0.762733 | 0 | 0 | 0.003294 | 0 | 32.174181 | 26 |
| Shared Policy target_original_normalized_position | 11103 | 29 | 0.000035 | 0.000049 | -0.000014 | 1.287011 | -0.112411 | 0.000035 | 1.849361 | 1 | 0.311433 | 0.609940 | 0.751962 | 0 | 0 | 0.092535 | 0 | 31.143253 | 25 |
| Shared Policy full_context_normalized_position | 11103 | 29 | 0.000030 | 0.000035 | -0.000005 | 0.867631 | -0.030143 | 0.000030 | 1.091706 | 0 | 0.638714 | 0.773965 | 0.846746 | 0 | 0 | 0.495727 | 0 | 63.871365 | 23 |

## F. Training Dynamics

Full training dynamics are available in `training_dynamics.csv`; the tables below show the first 5 rows per method-target pair.

### PosOptMVP / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000054 |  | 0.000054 | 1.382516 | 0.489624 | 0.000054 | 1.947992 | 1 | 0.285736 | 0.583474 | 0.736267 | 0 | 0 |
| 1 | 0.000085 | 0.000054 | 0.000030 | 1.381571 | 0.270236 | 0.000085 | 1.930605 | 1 | 0.298969 | 0.594245 | 0.742576 | 0 | 0 |
| 2 | 0.000126 | 0.000058 | 0.000068 | 1.379753 | 0.612030 | 0.000126 | 1.946915 | 1 | 0.296815 | 0.590091 | 0.741960 | 0 | 0 |
| 3 | 0.000082 | 0.000064 | 0.000018 | 1.377695 | 0.160568 | 0.000082 | 1.943222 | 1 | 0.296507 | 0.593784 | 0.749192 | 0 | 0 |
| 4 | 0.000052 | 0.000066 | -0.000015 | 1.375265 | -0.130187 | 0.000052 | 1.917526 | 1 | 0.296353 | 0.601323 | 0.745653 | 0 | 0 |

### Shared Policy local_context / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000053 |  | 0.000053 | 1.361674 | 0.465107 | 0.000053 | 1.896138 | 1 | 0.313741 | 0.594707 | 0.741037 | 0 | 0 |
| 1 | 0.000082 | 0.000053 | 0.000030 | 1.350187 | 0.259357 | 0.000082 | 2.061086 | 1 | 0.265887 | 0.565933 | 0.721649 | 0 | 0 |
| 2 | 0.000134 | 0.000056 | 0.000078 | 1.326924 | 0.674373 | 0.000134 | 2.186336 | 1 | 0.230959 | 0.542391 | 0.707032 | 0 | 0 |
| 3 | 0.000094 | 0.000064 | 0.000030 | 1.283606 | 0.252164 | 0.000094 | 2.312971 | 1 | 0.197569 | 0.519618 | 0.699954 | 0 | 0 |
| 4 | 0.000081 | 0.000067 | 0.000014 | 1.255073 | 0.114307 | 0.000081 | 2.364056 | 1 | 0.181413 | 0.514695 | 0.687337 | 0 | 0 |

### Shared Policy normalized_position_only / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000048 |  | 0.000048 | 1.382149 | 0.430430 | 0.000048 | 1.983998 | 1 | 0.280043 | 0.576396 | 0.730728 | 0 | 0 |
| 1 | 0.000084 | 0.000048 | 0.000036 | 1.381065 | 0.319901 | 0.000084 | 2.020311 | 1 | 0.278966 | 0.575319 | 0.729651 | 0 | 0 |
| 2 | 0.000126 | 0.000051 | 0.000074 | 1.379760 | 0.665955 | 0.000126 | 2.039698 | 1 | 0.276658 | 0.567472 | 0.726573 | 0 | 0 |
| 3 | 0.000079 | 0.000059 | 0.000020 | 1.379373 | 0.180101 | 0.000079 | 2.044007 | 1 | 0.273888 | 0.571626 | 0.734113 | 0 | 0 |
| 4 | 0.000071 | 0.000061 | 0.000010 | 1.379541 | 0.086998 | 0.000071 | 2.027697 | 1 | 0.273888 | 0.577473 | 0.731343 | 0 | 0 |

### Shared Policy target_normalized_position / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000054 |  | 0.000054 | 1.382514 | 0.489613 | 0.000054 | 1.951685 | 1 | 0.285121 | 0.582859 | 0.736113 | 0 | 0 |
| 1 | 0.000085 | 0.000054 | 0.000030 | 1.382445 | 0.270640 | 0.000085 | 1.945530 | 1 | 0.295122 | 0.590245 | 0.740268 | 0 | 0 |
| 2 | 0.000126 | 0.000058 | 0.000068 | 1.382352 | 0.613022 | 0.000126 | 1.968611 | 1 | 0.291122 | 0.584244 | 0.738267 | 0 | 0 |
| 3 | 0.000079 | 0.000064 | 0.000015 | 1.382258 | 0.131572 | 0.000079 | 1.973996 | 1 | 0.288660 | 0.586398 | 0.744422 | 0 | 0 |
| 4 | 0.000059 | 0.000066 | -0.000007 | 1.382196 | -0.059972 | 0.000059 | 1.949685 | 1 | 0.288506 | 0.593630 | 0.741499 | 0 | 0 |

### Shared Policy target_original_normalized_position / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000053 |  | 0.000053 | 1.378337 | 0.474163 | 0.000053 | 1.937529 | 1 | 0.290660 | 0.585936 | 0.737036 | 0 | 0 |
| 1 | 0.000083 | 0.000053 | 0.000030 | 1.376592 | 0.265751 | 0.000083 | 1.924912 | 1 | 0.301123 | 0.594399 | 0.742730 | 0 | 0 |
| 2 | 0.000116 | 0.000056 | 0.000060 | 1.373179 | 0.532522 | 0.000116 | 1.950300 | 1 | 0.299277 | 0.587937 | 0.740114 | 0 | 0 |
| 3 | 0.000082 | 0.000062 | 0.000020 | 1.369699 | 0.179059 | 0.000082 | 1.935529 | 1 | 0.294815 | 0.592399 | 0.750423 | 0 | 0 |
| 4 | 0.000060 | 0.000064 | -0.000004 | 1.365787 | -0.035808 | 0.000060 | 1.924758 | 1 | 0.296046 | 0.599015 | 0.745499 | 0 | 0 |

### Shared Policy full_context_normalized_position / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000057 |  | 0.000057 | 1.370020 | 0.513146 | 0.000057 | 1.919064 | 1 | 0.306509 | 0.589783 | 0.738267 | 0 | 0 |
| 1 | 0.000083 | 0.000057 | 0.000025 | 1.364246 | 0.224640 | 0.000083 | 2.062163 | 1 | 0.253270 | 0.565010 | 0.723188 | 0 | 0 |
| 2 | 0.000126 | 0.000060 | 0.000066 | 1.347749 | 0.573695 | 0.000126 | 2.150639 | 1 | 0.219726 | 0.547007 | 0.717495 | 0 | 0 |
| 3 | 0.000081 | 0.000067 | 0.000014 | 1.325966 | 0.121737 | 0.000081 | 2.207417 | 1 | 0.198184 | 0.533467 | 0.716110 | 0 | 0 |
| 4 | 0.000071 | 0.000068 | 0.000003 | 1.309395 | 0.024089 | 0.000071 | 2.237267 | 1 | 0.179412 | 0.531466 | 0.707340 | 0 | 0 |

### PosOptMVP / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000058 |  | 0.000058 | 1.382516 | 0.519007 | 0.000058 | 1.924142 | 1 | 0.298969 | 0.588860 | 0.741499 | 0 | 0 |
| 1 | 0.000045 | 0.000058 | -0.000013 | 1.381571 | -0.117610 | 0.000045 | 1.940145 | 1 | 0.291891 | 0.596861 | 0.742114 | 0 | 0 |
| 2 | 0.000041 | 0.000056 | -0.000015 | 1.379910 | -0.138023 | 0.000041 | 1.947069 | 1 | 0.294353 | 0.591783 | 0.742576 | 0 | 0 |
| 3 | 0.000045 | 0.000055 | -0.000010 | 1.378099 | -0.086025 | 0.000045 | 1.949223 | 1 | 0.290968 | 0.587321 | 0.741653 | 0 | 0 |
| 4 | 0.000038 | 0.000054 | -0.000016 | 1.376237 | -0.143843 | 0.000038 | 1.937837 | 1 | 0.302046 | 0.588244 | 0.743345 | 0 | 0 |

### Shared Policy local_context / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000055 |  | 0.000055 | 1.369429 | 0.486449 | 0.000055 | 2.066164 | 1 | 0.280043 | 0.568395 | 0.723496 | 0 | 0 |
| 1 | 0.000044 | 0.000055 | -0.000011 | 1.364847 | -0.093292 | 0.000044 | 1.975227 | 1 | 0.312817 | 0.595938 | 0.736421 | 0 | 0 |
| 2 | 0.000048 | 0.000054 | -0.000006 | 1.353580 | -0.050853 | 0.000048 | 1.917526 | 1 | 0.331436 | 0.596861 | 0.739806 | 0 | 0 |
| 3 | 0.000046 | 0.000053 | -0.000007 | 1.338776 | -0.064414 | 0.000046 | 1.864441 | 1 | 0.348669 | 0.606555 | 0.748731 | 0 | 0 |
| 4 | 0.000035 | 0.000052 | -0.000017 | 1.322022 | -0.146957 | 0.000035 | 1.835975 | 1 | 0.373904 | 0.614248 | 0.752423 | 0 | 0 |

### Shared Policy normalized_position_only / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000058 |  | 0.000058 | 1.382267 | 0.519008 | 0.000058 | 1.944145 | 1 | 0.290814 | 0.584090 | 0.738883 | 0 | 0 |
| 1 | 0.000045 | 0.000058 | -0.000013 | 1.382495 | -0.113400 | 0.000045 | 1.931066 | 1 | 0.291737 | 0.599938 | 0.743191 | 0 | 0 |
| 2 | 0.000040 | 0.000057 | -0.000017 | 1.382280 | -0.148610 | 0.000040 | 1.920911 | 1 | 0.301277 | 0.594707 | 0.745961 | 0 | 0 |
| 3 | 0.000052 | 0.000055 | -0.000003 | 1.381950 | -0.024803 | 0.000052 | 1.895830 | 1 | 0.304201 | 0.599477 | 0.749962 | 0 | 0 |
| 4 | 0.000038 | 0.000055 | -0.000016 | 1.381566 | -0.144953 | 0.000038 | 1.887521 | 1 | 0.314048 | 0.598554 | 0.750115 | 0 | 0 |

### Shared Policy target_normalized_position / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000058 |  | 0.000058 | 1.382515 | 0.519003 | 0.000058 | 1.922757 | 1 | 0.298969 | 0.588860 | 0.741653 | 0 | 0 |
| 1 | 0.000045 | 0.000058 | -0.000013 | 1.382463 | -0.113403 | 0.000045 | 1.922911 | 1 | 0.295276 | 0.601323 | 0.743807 | 0 | 0 |
| 2 | 0.000040 | 0.000057 | -0.000017 | 1.382400 | -0.148622 | 0.000040 | 1.928758 | 1 | 0.300816 | 0.592707 | 0.744422 | 0 | 0 |
| 3 | 0.000052 | 0.000055 | -0.000003 | 1.382347 | -0.024810 | 0.000052 | 1.916141 | 1 | 0.299277 | 0.594399 | 0.746576 | 0 | 0 |
| 4 | 0.000038 | 0.000055 | -0.000017 | 1.382294 | -0.150777 | 0.000038 | 1.910294 | 1 | 0.308047 | 0.593938 | 0.747038 | 0 | 0 |

### Shared Policy target_original_normalized_position / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000062 |  | 0.000062 | 1.379546 | 0.552505 | 0.000062 | 1.931374 | 1 | 0.299585 | 0.587783 | 0.739498 | 0 | 0 |
| 1 | 0.000044 | 0.000062 | -0.000018 | 1.377178 | -0.158916 | 0.000044 | 1.943530 | 1 | 0.292199 | 0.599631 | 0.740575 | 0 | 0 |
| 2 | 0.000040 | 0.000060 | -0.000020 | 1.374967 | -0.179877 | 0.000040 | 1.949069 | 1 | 0.294353 | 0.591322 | 0.742576 | 0 | 0 |
| 3 | 0.000047 | 0.000058 | -0.000011 | 1.370787 | -0.093459 | 0.000047 | 1.938914 | 1 | 0.293891 | 0.586860 | 0.741191 | 0 | 0 |
| 4 | 0.000038 | 0.000057 | -0.000019 | 1.366176 | -0.167636 | 0.000038 | 1.938914 | 1 | 0.300969 | 0.586552 | 0.740729 | 0 | 0 |

### Shared Policy full_context_normalized_position / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000057 |  | 0.000057 | 1.363358 | 0.508663 | 0.000057 | 1.985228 | 1 | 0.283120 | 0.579474 | 0.735652 | 0 | 0 |
| 1 | 0.000054 | 0.000057 | -0.000004 | 1.356662 | -0.033980 | 0.000054 | 1.991845 | 1 | 0.291276 | 0.587475 | 0.733344 | 0 | 0 |
| 2 | 0.000043 | 0.000057 | -0.000014 | 1.347468 | -0.122609 | 0.000043 | 1.916449 | 1 | 0.311433 | 0.596553 | 0.743807 | 0 | 0 |
| 3 | 0.000038 | 0.000056 | -0.000017 | 1.337708 | -0.151802 | 0.000038 | 1.913525 | 1 | 0.313587 | 0.592553 | 0.745192 | 0 | 0 |
| 4 | 0.000038 | 0.000054 | -0.000016 | 1.327077 | -0.133714 | 0.000038 | 1.878443 | 1 | 0.346976 | 0.604862 | 0.747961 | 0 | 0 |
