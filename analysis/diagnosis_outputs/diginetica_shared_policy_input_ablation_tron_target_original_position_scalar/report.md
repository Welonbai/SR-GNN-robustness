# diginetica_shared_policy_input_ablation_tron_target_original_position_scalar

## Scope
- Dataset: `diginetica`
- Victim: `tron`
- Selected targets: 5334, 11103
- Blank cells indicate unavailable method-target artifacts.
- Authoritative final metrics come from `summary_current.json`.
- Reference method: `Shared Policy local_context`

## A. Final Metrics Comparison
| target_item | method | target_recall@10 | target_recall@20 | target_recall@30 | target_mrr@10 | target_mrr@20 | target_mrr@30 | gt_recall@10 | gt_recall@20 | gt_recall@30 | gt_mrr@10 | gt_mrr@20 | gt_mrr@30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | Clean | 0.000411 | 0.000723 | 0.000986 | 0.000155 | 0.000177 | 0.000188 | 0.285484 | 0.384321 | 0.441273 | 0.128554 | 0.135381 | 0.137674 |
| 5334 | Prefix-NZ | 0.189885 | 0.232985 | 0.262102 | 0.117392 | 0.120342 | 0.121509 | 0.299155 | 0.404483 | 0.467416 | 0.130772 | 0.138084 | 0.140624 |
| 5334 | PosOptMVP | 0.048358 | 0.073187 | 0.092708 | 0.018313 | 0.020018 | 0.020790 | 0.295787 | 0.396185 | 0.455306 | 0.135368 | 0.142293 | 0.144669 |
| 5334 | Shared Policy local_context | 0.220595 | 0.278106 | 0.317099 | 0.125449 | 0.129372 | 0.130938 | 0.289132 | 0.386490 | 0.445496 | 0.127137 | 0.133882 | 0.136257 |
| 5334 | Shared Policy normalized_position_only | 0.000016 | 0.000016 | 0.000016 | 0.000003 | 0.000003 | 0.000003 | 0.312087 | 0.420372 | 0.482402 | 0.143279 | 0.150742 | 0.153239 |
| 5334 | Shared Policy target_normalized_position | 0.532042 | 0.583161 | 0.615728 | 0.418477 | 0.422009 | 0.423317 | 0.269710 | 0.366542 | 0.423576 | 0.102857 | 0.109603 | 0.111905 |
| 5334 | Shared Policy target_original_normalized_position | 0.047241 | 0.074041 | 0.096750 | 0.018134 | 0.019943 | 0.020851 | 0.295918 | 0.395971 | 0.455372 | 0.135808 | 0.142744 | 0.145136 |
| 5334 | Shared Policy target_original_position_scalar | 0.141477 | 0.183296 | 0.211772 | 0.071517 | 0.074387 | 0.075526 | 0.289247 | 0.388774 | 0.447698 | 0.130126 | 0.137003 | 0.139371 |
| 5334 | Shared Policy full_context_normalized_position | 0.110240 | 0.154294 | 0.188866 | 0.051251 | 0.054247 | 0.055639 | 0.297824 | 0.398173 | 0.458346 | 0.134558 | 0.141486 | 0.143907 |
| 5334 | Shared Policy local_context_prefix_score_prob | 0.000033 | 0.000049 | 0.000148 | 0.000018 | 0.000019 | 0.000023 | 0.310986 | 0.417546 | 0.480512 | 0.142935 | 0.150290 | 0.152830 |
| 11103 | Clean | 0.000148 | 0.000411 | 0.000608 | 0.000054 | 0.000071 | 0.000079 | 0.285484 | 0.384321 | 0.441273 | 0.128554 | 0.135381 | 0.137674 |
| 11103 | Prefix-NZ | 0.013786 | 0.023826 | 0.033422 | 0.005495 | 0.006169 | 0.006553 | 0.311906 | 0.420684 | 0.481136 | 0.142588 | 0.150112 | 0.152555 |
| 11103 | PosOptMVP | 0.131766 | 0.177479 | 0.210211 | 0.056011 | 0.059138 | 0.060450 | 0.303181 | 0.403102 | 0.459052 | 0.138581 | 0.145508 | 0.147758 |
| 11103 | Shared Policy local_context | 0.025354 | 0.045976 | 0.063213 | 0.006838 | 0.008243 | 0.008927 | 0.306237 | 0.408360 | 0.465116 | 0.141877 | 0.148980 | 0.151267 |
| 11103 | Shared Policy normalized_position_only | 0 | 0 | 0 | 0 | 0 | 0 | 0.313484 | 0.418762 | 0.478984 | 0.143578 | 0.150872 | 0.153304 |
| 11103 | Shared Policy target_normalized_position | 0 | 0 | 0 | 0 | 0 | 0 | 0.313484 | 0.418762 | 0.478984 | 0.143578 | 0.150872 | 0.153304 |
| 11103 | Shared Policy target_original_normalized_position | 0.049427 | 0.080745 | 0.105952 | 0.014618 | 0.016754 | 0.017760 | 0.305761 | 0.408163 | 0.468468 | 0.140812 | 0.147883 | 0.150308 |
| 11103 | Shared Policy target_original_position_scalar | 0.015610 | 0.032157 | 0.047931 | 0.004387 | 0.005485 | 0.006115 | 0.309129 | 0.415229 | 0.474071 | 0.143114 | 0.150461 | 0.152842 |
| 11103 | Shared Policy full_context_normalized_position | 0.000263 | 0.000937 | 0.002415 | 0.000043 | 0.000088 | 0.000145 | 0.308406 | 0.410184 | 0.469749 | 0.142393 | 0.149460 | 0.151859 |
| 11103 | Shared Policy local_context_prefix_score_prob | 0.000016 | 0.000066 | 0.000197 | 0.000004 | 0.000008 | 0.000013 | 0.313714 | 0.419863 | 0.478967 | 0.144309 | 0.151666 | 0.154054 |

## B. Delta vs Reference
| target_item | method | reference_method | target_recall@10 | target_recall@20 | target_recall@30 | target_mrr@10 | target_mrr@20 | target_mrr@30 | gt_recall@10 | gt_recall@20 | gt_recall@30 | gt_mrr@10 | gt_mrr@20 | gt_mrr@30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | Clean | Shared Policy local_context | -0.220185 | -0.277383 | -0.316113 | -0.125294 | -0.129195 | -0.130751 | -0.003648 | -0.002169 | -0.004223 | 0.001417 | 0.001499 | 0.001417 |
| 5334 | Prefix-NZ | Shared Policy local_context | -0.030711 | -0.045121 | -0.054997 | -0.008057 | -0.009030 | -0.009429 | 0.010023 | 0.017993 | 0.021920 | 0.003635 | 0.004201 | 0.004367 |
| 5334 | PosOptMVP | Shared Policy local_context | -0.172237 | -0.204920 | -0.224391 | -0.107136 | -0.109354 | -0.110148 | 0.006655 | 0.009695 | 0.009810 | 0.008231 | 0.008410 | 0.008412 |
| 5334 | Shared Policy normalized_position_only | Shared Policy local_context | -0.220579 | -0.278090 | -0.317082 | -0.125446 | -0.129369 | -0.130935 | 0.022955 | 0.033882 | 0.036906 | 0.016143 | 0.016859 | 0.016983 |
| 5334 | Shared Policy target_normalized_position | Shared Policy local_context | 0.311446 | 0.305054 | 0.298630 | 0.293028 | 0.292637 | 0.292379 | -0.019422 | -0.019948 | -0.021920 | -0.024280 | -0.024279 | -0.024351 |
| 5334 | Shared Policy target_original_normalized_position | Shared Policy local_context | -0.173354 | -0.204065 | -0.220349 | -0.107315 | -0.109429 | -0.110088 | 0.006786 | 0.009481 | 0.009875 | 0.008671 | 0.008862 | 0.008880 |
| 5334 | Shared Policy target_original_position_scalar | Shared Policy local_context | -0.079119 | -0.094811 | -0.105327 | -0.053932 | -0.054985 | -0.055413 | 0.000115 | 0.002284 | 0.002202 | 0.002989 | 0.003120 | 0.003114 |
| 5334 | Shared Policy full_context_normalized_position | Shared Policy local_context | -0.110355 | -0.123813 | -0.128233 | -0.074198 | -0.075125 | -0.075300 | 0.008692 | 0.011683 | 0.012850 | 0.007421 | 0.007604 | 0.007651 |
| 5334 | Shared Policy local_context_prefix_score_prob | Shared Policy local_context | -0.220563 | -0.278057 | -0.316951 | -0.125431 | -0.129352 | -0.130915 | 0.021854 | 0.031056 | 0.035016 | 0.015798 | 0.016408 | 0.016573 |
| 11103 | Clean | Shared Policy local_context | -0.025206 | -0.045565 | -0.062605 | -0.006784 | -0.008171 | -0.008847 | -0.020753 | -0.024040 | -0.023842 | -0.013323 | -0.013599 | -0.013593 |
| 11103 | Prefix-NZ | Shared Policy local_context | -0.011568 | -0.022150 | -0.029791 | -0.001344 | -0.002073 | -0.002374 | 0.005669 | 0.012324 | 0.016021 | 0.000711 | 0.001132 | 0.001288 |
| 11103 | PosOptMVP | Shared Policy local_context | 0.106412 | 0.131503 | 0.146998 | 0.049173 | 0.050895 | 0.051523 | -0.003056 | -0.005258 | -0.006063 | -0.003296 | -0.003472 | -0.003509 |
| 11103 | Shared Policy normalized_position_only | Shared Policy local_context | -0.025354 | -0.045976 | -0.063213 | -0.006838 | -0.008243 | -0.008927 | 0.007246 | 0.010401 | 0.013868 | 0.001701 | 0.001892 | 0.002037 |
| 11103 | Shared Policy target_normalized_position | Shared Policy local_context | -0.025354 | -0.045976 | -0.063213 | -0.006838 | -0.008243 | -0.008927 | 0.007246 | 0.010401 | 0.013868 | 0.001701 | 0.001892 | 0.002037 |
| 11103 | Shared Policy target_original_normalized_position | Shared Policy local_context | 0.024072 | 0.034769 | 0.042739 | 0.007780 | 0.008511 | 0.008833 | -0.000477 | -0.000197 | 0.003352 | -0.001065 | -0.001097 | -0.000959 |
| 11103 | Shared Policy target_original_position_scalar | Shared Policy local_context | -0.009744 | -0.013819 | -0.015281 | -0.002452 | -0.002758 | -0.002811 | 0.002892 | 0.006868 | 0.008955 | 0.001237 | 0.001481 | 0.001575 |
| 11103 | Shared Policy full_context_normalized_position | Shared Policy local_context | -0.025091 | -0.045039 | -0.060797 | -0.006796 | -0.008155 | -0.008782 | 0.002169 | 0.001824 | 0.004634 | 0.000516 | 0.000480 | 0.000591 |
| 11103 | Shared Policy local_context_prefix_score_prob | Shared Policy local_context | -0.025338 | -0.045910 | -0.063016 | -0.006834 | -0.008235 | -0.008914 | 0.007476 | 0.011502 | 0.013852 | 0.002432 | 0.002686 | 0.002786 |

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
| 5334 | Shared Policy target_original_position_scalar | 6499 | 31 | 1 | 27.573473 | 20.326204 | 47.899677 | 64.625327 | 86.120942 | 1:27.57%, 0:20.33%, 2:16.73%, 3:10.14%, 4:6.82% |
| 5334 | Shared Policy full_context_normalized_position | 6499 | 26 | 1 | 31.851054 | 30.389291 | 62.240345 | 77.504231 | 93.737498 | 1:31.85%, 0:30.39%, 2:15.26%, 3:8.28%, 4:5.08% |
| 5334 | Shared Policy local_context_prefix_score_prob | 6499 | 12 | 0 | 97.984305 | 97.984305 | 98.815202 | 99.230651 | 99.846130 | 0:97.98%, 1:0.83%, 2:0.42%, 3:0.32%, 4:0.20% |
| 11103 | Clean |  |  |  |  |  |  |  |  |  |
| 11103 | Prefix-NZ | 6499 | 7 | 0 | 71.134021 | 71.134021 | 96.922603 | 99.230651 | 99.984613 | 0:71.13%, 1:25.79%, 2:2.31%, 3:0.58%, 4:0.14% |
| 11103 | PosOptMVP | 6499 | 23 | 0 | 29.973842 | 29.973842 | 59.039852 | 74.349900 | 92.229574 | 0:29.97%, 1:29.07%, 2:15.31%, 3:8.89%, 4:5.65% |
| 11103 | Shared Policy local_context | 6499 | 12 | 0 | 92.014156 | 92.014156 | 95.737806 | 97.414987 | 99.461456 | 0:92.01%, 1:3.72%, 2:1.68%, 3:1.03%, 4:0.62% |
| 11103 | Shared Policy normalized_position_only | 6499 | 1 | 0 | 100 | 100 | 100 | 100 | 100 | 0:100.00% |
| 11103 | Shared Policy target_normalized_position | 6499 | 1 | 0 | 100 | 100 | 100 | 100 | 100 | 0:100.00% |
| 11103 | Shared Policy target_original_normalized_position | 6499 | 24 | 0 | 34.543776 | 34.543776 | 62.517310 | 77.181105 | 93.968303 | 0:34.54%, 1:27.97%, 2:14.66%, 3:8.26%, 4:5.68% |
| 11103 | Shared Policy target_original_position_scalar | 6499 | 12 | 0 | 50.853978 | 50.853978 | 80.581628 | 92.352670 | 99.630712 | 0:50.85%, 1:29.73%, 2:11.77%, 3:4.80%, 4:1.82% |
| 11103 | Shared Policy full_context_normalized_position | 6499 | 23 | 0 | 81.104785 | 81.104785 | 87.505770 | 91.290968 | 96.861056 | 0:81.10%, 1:6.40%, 2:3.79%, 3:2.69%, 4:1.69% |
| 11103 | Shared Policy local_context_prefix_score_prob | 6499 | 8 | 0 | 95.507001 | 95.507001 | 98.107401 | 99.492230 | 99.923065 | 0:95.51%, 1:2.60%, 2:1.38%, 3:0.28%, 4:0.14% |

## D. Verification Summary
| target_item | method | policy_feature_set | active_item_features | active_scalar_features | policy_input_dim | policy_embedding_dim | policy_hidden_dim | prefix_score_enabled | run_metadata_present | training_history_present | learned_logits_present | expected_policy_feature_set | expected_active_item_features | expected_active_scalar_features | expected_input_dim_from_features | feature_set_matches_expected | active_features_match_expected | input_dim_matches_expected | prefix_flag_matches_expected | artifact_consistency_status | verification_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5334 | Shared Policy local_context |  |  |  |  | 16 | 32 |  | True | True | True | local_context | ['target_item', 'original_item', 'left_item', 'right_item'] | ['position_index', 'normalized_position', 'session_length'] | 67 |  |  |  |  | legacy_missing_fields | legacy_missing_fields |
| 5334 | Shared Policy normalized_position_only | normalized_position_only | [] | ['normalized_position'] | 1 | 16 | 32 | False | True | True | True | normalized_position_only | [] | ['normalized_position'] | 1 | True | True | True | True | ok | ok |
| 5334 | Shared Policy target_normalized_position | target_normalized_position | ['target_item'] | ['normalized_position'] | 17 | 16 | 32 | False | True | True | True | target_normalized_position | ['target_item'] | ['normalized_position'] | 17 | True | True | True | True | ok | ok |
| 5334 | Shared Policy target_original_normalized_position | target_original_normalized_position | ['target_item', 'original_item'] | ['normalized_position'] | 33 | 16 | 32 | False | True | True | True | target_original_normalized_position | ['target_item', 'original_item'] | ['normalized_position'] | 33 | True | True | True | True | ok | ok |
| 5334 | Shared Policy target_original_position_scalar | target_original_position_scalar | ['target_item', 'original_item'] | ['position_index', 'normalized_position', 'session_length'] | 35 | 16 | 32 | False | True | True | True | target_original_position_scalar | ['target_item', 'original_item'] | ['position_index', 'normalized_position', 'session_length'] | 35 | True | True | True | True | ok | ok |
| 5334 | Shared Policy full_context_normalized_position | full_context_normalized_position | ['target_item', 'original_item', 'left_item', 'right_item'] | ['normalized_position'] | 65 | 16 | 32 | False | True | True | True | full_context_normalized_position | ['target_item', 'original_item', 'left_item', 'right_item'] | ['normalized_position'] | 65 | True | True | True | True | ok | ok |
| 5334 | Shared Policy local_context_prefix_score_prob | local_context_prefix_score_prob |  | ['position_index', 'normalized_position', 'session_length', 'prefix_score', 'has_prefix'] |  | 16 | 32 | True | True | True | True | local_context_prefix_score_prob | ['target_item', 'original_item', 'left_item', 'right_item'] | ['position_index', 'normalized_position', 'session_length', 'prefix_score', 'has_prefix'] | 69 | True |  |  | True | legacy_missing_fields | legacy_missing_fields |
| 11103 | Shared Policy local_context |  |  |  |  | 16 | 32 |  | True | True | True | local_context | ['target_item', 'original_item', 'left_item', 'right_item'] | ['position_index', 'normalized_position', 'session_length'] | 67 |  |  |  |  | legacy_missing_fields | legacy_missing_fields |
| 11103 | Shared Policy normalized_position_only | normalized_position_only | [] | ['normalized_position'] | 1 | 16 | 32 | False | True | True | True | normalized_position_only | [] | ['normalized_position'] | 1 | True | True | True | True | ok | ok |
| 11103 | Shared Policy target_normalized_position | target_normalized_position | ['target_item'] | ['normalized_position'] | 17 | 16 | 32 | False | True | True | True | target_normalized_position | ['target_item'] | ['normalized_position'] | 17 | True | True | True | True | ok | ok |
| 11103 | Shared Policy target_original_normalized_position | target_original_normalized_position | ['target_item', 'original_item'] | ['normalized_position'] | 33 | 16 | 32 | False | True | True | True | target_original_normalized_position | ['target_item', 'original_item'] | ['normalized_position'] | 33 | True | True | True | True | ok | ok |
| 11103 | Shared Policy target_original_position_scalar | target_original_position_scalar | ['target_item', 'original_item'] | ['position_index', 'normalized_position', 'session_length'] | 35 | 16 | 32 | False | True | True | True | target_original_position_scalar | ['target_item', 'original_item'] | ['position_index', 'normalized_position', 'session_length'] | 35 | True | True | True | True | ok | ok |
| 11103 | Shared Policy full_context_normalized_position | full_context_normalized_position | ['target_item', 'original_item', 'left_item', 'right_item'] | ['normalized_position'] | 65 | 16 | 32 | False | True | True | True | full_context_normalized_position | ['target_item', 'original_item', 'left_item', 'right_item'] | ['normalized_position'] | 65 | True | True | True | True | ok | ok |
| 11103 | Shared Policy local_context_prefix_score_prob | local_context_prefix_score_prob |  | ['position_index', 'normalized_position', 'session_length', 'prefix_score', 'has_prefix'] |  | 16 | 32 | True | True | True | True | local_context_prefix_score_prob | ['target_item', 'original_item', 'left_item', 'right_item'] | ['position_index', 'normalized_position', 'session_length', 'prefix_score', 'has_prefix'] | 69 | True |  |  | True | legacy_missing_fields | legacy_missing_fields |

## E. Training Final-Step Summary
| method | target_item | outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty | entropy_drop | final_dominant_position | final_dominant_share_pct | final_unique_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PosOptMVP | 5334 | 29 | 0.000098 | 0.000075 | 0.000023 | 1.335112 | 0.196493 | 0.000098 | 1.963533 | 1 | 0.290814 | 0.580551 | 0.741806 | 0 | 0 | 0.047403 | 0 | 29.081397 | 25 |
| Shared Policy local_context | 5334 | 29 | 0.000112 | 0.000091 | 0.000021 | 1.151750 | 0.157789 | 0.000112 | 2.409140 | 2 | 0.138637 | 0.486229 | 0.679643 | 0 | 0 | 0.209923 | 1 | 34.759194 | 25 |
| Shared Policy normalized_position_only | 5334 | 29 | 0.000089 | 0.000070 | 0.000018 | 1.376627 | 0.162384 | 0.000089 | 1.825512 | 1 | 0.326358 | 0.612402 | 0.763964 | 0 | 0 | 0.005522 | 0 | 32.635790 | 24 |
| Shared Policy target_normalized_position | 5334 | 29 | 0.000095 | 0.000076 | 0.000020 | 1.382479 | 0.175859 | 0.000095 | 1.976612 | 1 | 0.286352 | 0.582551 | 0.742884 | 0 | 0 | 0.000035 | 1 | 29.619942 | 25 |
| Shared Policy target_original_normalized_position | 5334 | 29 | 0.000091 | 0.000075 | 0.000016 | 1.315922 | 0.133087 | 0.000091 | 1.934298 | 1 | 0.292968 | 0.589475 | 0.749038 | 0 | 0 | 0.062415 | 1 | 29.650715 | 25 |
| Shared Policy target_original_position_scalar | 5334 | 29 | 0.000123 | 0.000077 | 0.000046 | 1.307810 | 0.395061 | 0.000123 | 2.198492 | 1 | 0.260194 | 0.551008 | 0.714418 | 0 | 0 | 0.064681 | 1 | 29.081397 | 28 |
| Shared Policy full_context_normalized_position | 5334 | 29 | 0.000097 | 0.000081 | 0.000016 | 1.340021 | 0.137984 | 0.000097 | 1.941529 | 1 | 0.284044 | 0.589321 | 0.749038 | 0 | 0 | 0.029999 | 1 | 30.527774 | 25 |
| Shared Policy local_context_prefix_score_prob | 5334 | 29 | 0.000055 | 0.000049 | 0.000006 | 0.526493 | 0.019735 | 0.000055 | 0.473150 | 0 | 0.834282 | 0.897061 | 0.931220 | 0 | 0 | 0.837649 | 0 | 83.428220 | 18 |
| PosOptMVP | 11103 | 29 | 0.000034 | 0.000049 | -0.000014 | 1.341928 | -0.122388 | 0.000034 | 1.910602 | 1 | 0.296199 | 0.595476 | 0.741191 | 0 | 0 | 0.040588 | 1 | 29.927681 | 26 |
| Shared Policy local_context | 11103 | 29 | 0.000023 | 0.000032 | -0.000010 | 0.476913 | -0.031139 | 0.000023 | 0.470380 | 0 | 0.823050 | 0.896599 | 0.931990 | 0 | 0 | 0.892516 | 0 | 82.304970 | 17 |
| Shared Policy normalized_position_only | 11103 | 29 | 0.000027 | 0.000046 | -0.000020 | 1.364328 | -0.172527 | 0.000027 | 1.635329 | 1 | 0.360517 | 0.655485 | 0.784736 | 0 | 0 | 0.017940 | 0 | 36.051700 | 26 |
| Shared Policy target_normalized_position | 11103 | 29 | 0.000029 | 0.000048 | -0.000019 | 1.379221 | -0.173265 | 0.000029 | 1.785044 | 1 | 0.321742 | 0.624404 | 0.762733 | 0 | 0 | 0.003294 | 0 | 32.174181 | 26 |
| Shared Policy target_original_normalized_position | 11103 | 29 | 0.000035 | 0.000049 | -0.000014 | 1.287011 | -0.112411 | 0.000035 | 1.849361 | 1 | 0.311433 | 0.609940 | 0.751962 | 0 | 0 | 0.092535 | 0 | 31.143253 | 25 |
| Shared Policy target_original_position_scalar | 11103 | 29 | 0.000036 | 0.000044 | -0.000008 | 1.219212 | -0.063719 | 0.000036 | 1.371442 | 1 | 0.376519 | 0.688414 | 0.825358 | 0 | 0 | 0.158264 | 0 | 37.651946 | 19 |
| Shared Policy full_context_normalized_position | 11103 | 29 | 0.000030 | 0.000035 | -0.000005 | 0.867631 | -0.030143 | 0.000030 | 1.091706 | 0 | 0.638714 | 0.773965 | 0.846746 | 0 | 0 | 0.495727 | 0 | 63.871365 | 23 |
| Shared Policy local_context_prefix_score_prob | 11103 | 29 | 0.000025 | 0.000035 | -0.000010 | 0.830841 | -0.055348 | 0.000025 | 0.708417 | 0 | 0.706570 | 0.837821 | 0.902754 | 0 | 0 | 0.540764 | 0 | 70.657024 | 17 |

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

### Shared Policy target_original_position_scalar / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000053 |  | 0.000053 | 1.372491 | 0.468760 | 0.000053 | 1.949685 | 1 | 0.287737 | 0.583474 | 0.736113 | 0 | 0 |
| 1 | 0.000080 | 0.000053 | 0.000027 | 1.371253 | 0.243388 | 0.000080 | 1.974458 | 1 | 0.291122 | 0.585936 | 0.735806 | 0 | 0 |
| 2 | 0.000116 | 0.000055 | 0.000061 | 1.369662 | 0.540301 | 0.000116 | 2.020772 | 1 | 0.289275 | 0.577012 | 0.733344 | 0 | 0 |
| 3 | 0.000081 | 0.000061 | 0.000020 | 1.367054 | 0.175896 | 0.000081 | 2.049084 | 1 | 0.287121 | 0.578089 | 0.737190 | 0 | 0 |
| 4 | 0.000062 | 0.000063 | -0.000001 | 1.362821 | -0.007265 | 0.000062 | 2.071396 | 1 | 0.279735 | 0.580243 | 0.729343 | 0 | 0 |

### Shared Policy full_context_normalized_position / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000057 |  | 0.000057 | 1.370020 | 0.513146 | 0.000057 | 1.919064 | 1 | 0.306509 | 0.589783 | 0.738267 | 0 | 0 |
| 1 | 0.000083 | 0.000057 | 0.000025 | 1.364246 | 0.224640 | 0.000083 | 2.062163 | 1 | 0.253270 | 0.565010 | 0.723188 | 0 | 0 |
| 2 | 0.000126 | 0.000060 | 0.000066 | 1.347749 | 0.573695 | 0.000126 | 2.150639 | 1 | 0.219726 | 0.547007 | 0.717495 | 0 | 0 |
| 3 | 0.000081 | 0.000067 | 0.000014 | 1.325966 | 0.121737 | 0.000081 | 2.207417 | 1 | 0.198184 | 0.533467 | 0.716110 | 0 | 0 |
| 4 | 0.000071 | 0.000068 | 0.000003 | 1.309395 | 0.024089 | 0.000071 | 2.237267 | 1 | 0.179412 | 0.531466 | 0.707340 | 0 | 0 |

### Shared Policy local_context_prefix_score_prob / target 5334

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000057 |  | 0.000057 | 1.364142 | 0.502895 | 0.000057 | 1.823973 | 1 | 0.316356 | 0.609632 | 0.754578 | 0 | 0 |
| 1 | 0.000078 | 0.000057 | 0.000021 | 1.359651 | 0.187386 | 0.000078 | 1.871519 | 1 | 0.289737 | 0.599938 | 0.753039 | 0 | 0 |
| 2 | 0.000097 | 0.000059 | 0.000038 | 1.353837 | 0.337018 | 0.000097 | 1.953070 | 1 | 0.268964 | 0.579320 | 0.744114 | 0 | 0 |
| 3 | 0.000077 | 0.000063 | 0.000015 | 1.346873 | 0.127562 | 0.000077 | 2.013541 | 1 | 0.253424 | 0.575781 | 0.742576 | 0 | 0 |
| 4 | 0.000064 | 0.000064 | -0.000001 | 1.341045 | -0.004994 | 0.000064 | 2.017849 | 1 | 0.248038 | 0.574550 | 0.735806 | 0 | 0 |

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

### Shared Policy target_original_position_scalar / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000048 |  | 0.000048 | 1.377476 | 0.431853 | 0.000048 | 1.883982 | 1 | 0.308663 | 0.599477 | 0.747807 | 0 | 0 |
| 1 | 0.000041 | 0.000048 | -0.000007 | 1.374961 | -0.060064 | 0.000041 | 1.818741 | 1 | 0.310817 | 0.618711 | 0.759194 | 0 | 0 |
| 2 | 0.000043 | 0.000048 | -0.000004 | 1.370725 | -0.040135 | 0.000043 | 1.772888 | 1 | 0.322357 | 0.621019 | 0.769195 | 0 | 0 |
| 3 | 0.000040 | 0.000047 | -0.000007 | 1.365234 | -0.060708 | 0.000040 | 1.673950 | 1 | 0.332205 | 0.627789 | 0.778427 | 0 | 0 |
| 4 | 0.000038 | 0.000046 | -0.000008 | 1.362007 | -0.073837 | 0.000038 | 1.696876 | 1 | 0.340514 | 0.628558 | 0.776427 | 0 | 0 |

### Shared Policy full_context_normalized_position / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000057 |  | 0.000057 | 1.363358 | 0.508663 | 0.000057 | 1.985228 | 1 | 0.283120 | 0.579474 | 0.735652 | 0 | 0 |
| 1 | 0.000054 | 0.000057 | -0.000004 | 1.356662 | -0.033980 | 0.000054 | 1.991845 | 1 | 0.291276 | 0.587475 | 0.733344 | 0 | 0 |
| 2 | 0.000043 | 0.000057 | -0.000014 | 1.347468 | -0.122609 | 0.000043 | 1.916449 | 1 | 0.311433 | 0.596553 | 0.743807 | 0 | 0 |
| 3 | 0.000038 | 0.000056 | -0.000017 | 1.337708 | -0.151802 | 0.000038 | 1.913525 | 1 | 0.313587 | 0.592553 | 0.745192 | 0 | 0 |
| 4 | 0.000038 | 0.000054 | -0.000016 | 1.327077 | -0.133714 | 0.000038 | 1.878443 | 1 | 0.346976 | 0.604862 | 0.747961 | 0 | 0 |

### Shared Policy local_context_prefix_score_prob / target 11103

| outer_step | reward | baseline | advantage | mean_entropy | policy_loss | target_utility | avg_selected_position | median_selected_position | fraction_pos0 | fraction_pos<=1 | fraction_pos<=2 | gt_drop | gt_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000054 |  | 0.000054 | 1.371604 | 0.483141 | 0.000054 | 1.880135 | 1 | 0.309740 | 0.593322 | 0.745038 | 0 | 0 |
| 1 | 0.000044 | 0.000054 | -0.000010 | 1.364743 | -0.092580 | 0.000044 | 1.808740 | 1 | 0.330666 | 0.618403 | 0.759040 | 0 | 0 |
| 2 | 0.000043 | 0.000053 | -0.000010 | 1.348771 | -0.088592 | 0.000043 | 1.694876 | 1 | 0.362517 | 0.634251 | 0.772119 | 0 | 0 |
| 3 | 0.000045 | 0.000052 | -0.000007 | 1.331315 | -0.057818 | 0.000045 | 1.628866 | 1 | 0.375904 | 0.637175 | 0.778427 | 0 | 0 |
| 4 | 0.000033 | 0.000051 | -0.000019 | 1.305538 | -0.157884 | 0.000033 | 1.524388 | 1 | 0.423604 | 0.662256 | 0.790891 | 0 | 0 |
