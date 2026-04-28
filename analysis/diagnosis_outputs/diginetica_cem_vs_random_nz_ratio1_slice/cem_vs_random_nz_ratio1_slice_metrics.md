# CEM vs Random-NZ Ratio1 Slice Metrics

random_nz_run: `outputs\runs\diginetica\attack_random_nonzero_when_possible_ratio1\run_group_8679b974a1`  
cem_run: `outputs\runs\diginetica\attack_rank_bucket_cem\run_group_cadb73910d`  
overlap_targets: 5334, 11103  
victims: miasrec, srgnn, tron

| target_item | victim | scope | metric | k | random_nz_ratio1 | cem | delta | rel_delta_pct |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 5334 | miasrec | targeted | recall | 10 | 0.191610 | 0.351457 | 0.159848 | 83.423 |
| 5334 | miasrec | targeted | recall | 20 | 0.498636 | 0.462404 | -0.036232 | -7.266 |
| 5334 | miasrec | targeted | recall | 30 | 0.732180 | 0.480627 | -0.251553 | -34.357 |
| 5334 | miasrec | targeted | mrr | 10 | 0.039443 | 0.079922 | 0.040479 | 102.626 |
| 5334 | miasrec | targeted | mrr | 20 | 0.059959 | 0.088268 | 0.028309 | 47.214 |
| 5334 | miasrec | targeted | mrr | 30 | 0.069401 | 0.088997 | 0.019596 | 28.236 |
| 5334 | srgnn | targeted | recall | 10 | 0.128972 | 0.128594 | -0.000378 | -0.293 |
| 5334 | srgnn | targeted | recall | 20 | 0.214121 | 0.201140 | -0.012981 | -6.062 |
| 5334 | srgnn | targeted | recall | 30 | 0.276545 | 0.254708 | -0.021838 | -7.897 |
| 5334 | srgnn | targeted | mrr | 10 | 0.046666 | 0.045580 | -0.001087 | -2.328 |
| 5334 | srgnn | targeted | mrr | 20 | 0.052452 | 0.050595 | -0.001857 | -3.540 |
| 5334 | srgnn | targeted | mrr | 30 | 0.054943 | 0.052746 | -0.002197 | -3.999 |
| 5334 | tron | targeted | recall | 10 | 0.154524 | 0.362467 | 0.207943 | 134.570 |
| 5334 | tron | targeted | recall | 20 | 0.194272 | 0.430708 | 0.236436 | 121.703 |
| 5334 | tron | targeted | recall | 30 | 0.223586 | 0.474153 | 0.250567 | 112.067 |
| 5334 | tron | targeted | mrr | 10 | 0.085562 | 0.228930 | 0.143368 | 167.561 |
| 5334 | tron | targeted | mrr | 20 | 0.088277 | 0.233639 | 0.145361 | 164.664 |
| 5334 | tron | targeted | mrr | 30 | 0.089448 | 0.235388 | 0.145940 | 163.157 |
| 11103 | miasrec | targeted | recall | 10 | 0.218443 | 0.298170 | 0.079727 | 36.498 |
| 11103 | miasrec | targeted | recall | 20 | 0.519636 | 0.483552 | -0.036084 | -6.944 |
| 11103 | miasrec | targeted | recall | 30 | 0.708272 | 0.655312 | -0.052959 | -7.477 |
| 11103 | miasrec | targeted | mrr | 10 | 0.046527 | 0.068970 | 0.022443 | 48.235 |
| 11103 | miasrec | targeted | mrr | 20 | 0.066948 | 0.081396 | 0.014448 | 21.581 |
| 11103 | miasrec | targeted | mrr | 30 | 0.074587 | 0.088257 | 0.013670 | 18.328 |
| 11103 | srgnn | targeted | recall | 10 | 0.124174 | 0.129646 | 0.005472 | 4.407 |
| 11103 | srgnn | targeted | recall | 20 | 0.208945 | 0.211016 | 0.002070 | 0.991 |
| 11103 | srgnn | targeted | recall | 30 | 0.278468 | 0.270071 | -0.008397 | -3.015 |
| 11103 | srgnn | targeted | mrr | 10 | 0.047333 | 0.046036 | -0.001297 | -2.740 |
| 11103 | srgnn | targeted | mrr | 20 | 0.052994 | 0.051587 | -0.001407 | -2.656 |
| 11103 | srgnn | targeted | mrr | 30 | 0.055780 | 0.053945 | -0.001835 | -3.290 |
| 11103 | tron | targeted | recall | 10 | 0.199645 | 0.184166 | -0.015479 | -7.753 |
| 11103 | tron | targeted | recall | 20 | 0.241546 | 0.241595 | 0.000049 | 0.020 |
| 11103 | tron | targeted | recall | 30 | 0.269414 | 0.280177 | 0.010763 | 3.995 |
| 11103 | tron | targeted | mrr | 10 | 0.122735 | 0.090792 | -0.031943 | -26.026 |
| 11103 | tron | targeted | mrr | 20 | 0.125636 | 0.094790 | -0.030846 | -24.552 |
| 11103 | tron | targeted | mrr | 30 | 0.126755 | 0.096340 | -0.030415 | -23.995 |
