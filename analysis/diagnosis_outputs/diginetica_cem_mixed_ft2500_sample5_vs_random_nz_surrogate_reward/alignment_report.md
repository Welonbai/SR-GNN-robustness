# CEM ft2500 Sample5 Surrogate-Final Alignment

- status: complete
- elapsed_seconds: 708.4
- config: `attack\configs\diginetica_attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5.yaml`
- CEM run: `outputs\runs\diginetica\attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5\run_group_e4d7034404`
- Random-NZ run: `outputs\runs\diginetica\attack_random_nonzero_when_possible_ratio1_srgnn_sample5\run_group_720516397a`
- clean surrogate: `outputs\surrogates\diginetica\clean_srgnn_surrogate_from_attack_a7fd31f6af.pt`
- CEM reward source: stored best row in `cem_trace.jsonl`
- Random reward source: rescored with the same normal mixed ft2500 surrogate evaluator

## Four-Case Table

| target | CEM reward | Random reward | surrogate delta | surrogate judgement | final wins 6 | final wins @10/@20 4 | final judgement | case |
|---:|---:|---:|---:|---|---:|---:|---|---|
| 5334 | 0.0099420864 | 0.0080052932 | 0.0019367932 | CEM > Random | 6 | 4 | CEM > Random | reward_align |
| 5418 | 0.0046731429 | 0.0016956554 | 0.0029774875 | CEM > Random | 0 | 0 | CEM < Random | reward_final_misalignment |
| 11103 | 0.0099376747 | 0.0057575510 | 0.0041801237 | CEM > Random | 1 | 1 | CEM < Random | reward_final_misalignment |
| 14514 | 0.0067183576 | 0.0042547500 | 0.0024636075 | CEM > Random | 0 | 0 | CEM < Random | reward_final_misalignment |
| 39588 | 0.0084149744 | 0.0053703244 | 0.0030446500 | CEM > Random | 5 | 4 | CEM > Random | reward_align |

## Summary

- Case counts: `{"reward_align": 2, "reward_final_misalignment": 3}`
- `reward_final_misalignment` means CEM surrogate reward is higher than Random-NZ, but final SRGNN target metrics are worse.
- `search_or_reference_gap` means the surrogate evaluator itself prefers Random-NZ; CEM did not include Random-NZ as a candidate/reference.

## Output Files

- `surrogate_reward_comparison.csv`
- `surrogate_final_alignment.csv`
- `final_metric_overlap.csv`
- `manifest.json`
