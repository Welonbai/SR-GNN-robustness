# Diginetica CEM ft6500 Paired-Relative Diagnostic

## Scope

- Dataset: Diginetica
- Victim/model setting: SR-GNN matched surrogate/victim context from existing artifacts
- Compared candidates: existing CEM best vs reconstructed Random-NZ ratio1
- Targets: 11103, 5418, 14514
- Fine-tune: normal mixed surrogate evaluator, configured/effective 6500 optimizer steps
- Reward mode: paired_relative_mrr_recall_10_20

## Reward Definition

For each metric in targeted_mrr@10, targeted_mrr@20, targeted_recall@10, targeted_recall@20:

```text
norm_metric(candidate) = metric(candidate) / max(metric(CEM), metric(Random), eps)
paired_reward(candidate) = mean(norm_MRR10, norm_MRR20, norm_Recall10, norm_Recall20)
```

## Summary

| target | CEM reward | Random reward | surrogate judgement | final wins 6 | final judgement | case |
|---:|---:|---:|---|---:|---|---|
| 11103 | 0.8493 | 1.0000 | CEM < Random | 1 | CEM < Random | search_or_reference_gap |
| 5418 | 0.9120 | 1.0000 | CEM < Random | 0 | CEM < Random | search_or_reference_gap |
| 14514 | 0.8417 | 1.0000 | CEM < Random | 0 | CEM < Random | search_or_reference_gap |

## Interpretation

- All three failed targets have paired-relative surrogate reward Random-NZ > CEM best.
- Final target metrics also favor Random-NZ on these targets.
- This points away from pure surrogate-victim misalignment for these cases; the stronger explanation is that CEM search/best candidate did not beat the Random-NZ reference under a metric-aligned surrogate reward.
- For target 11103, earlier target_result.mean slightly preferred CEM, but paired-relative MRR/Recall@10/20 reverses that judgement, indicating target_result.mean was misleading for low-k target metrics.

## Files In This Folder

- `summary_table.csv`: compact per-target comparison.
- `candidate_scores.csv`: merged per-candidate surrogate scores and metrics.
- `alignment_cases.csv`: merged paired CEM-vs-Random alignment classification.
- `final_metric_overlap.csv`: merged final victim metric overlap/deltas.
- `manifest.json`: source paths and generated file list.

## Source Folders

- `analysis\diagnosis_outputs\diginetica_cem_mixed_ft6500_paired_relative_mrr_recall_10_20_target11103_vs_random_nz_surrogate_reward`
- `analysis\diagnosis_outputs\diginetica_cem_mixed_ft6500_paired_relative_mrr_recall_10_20_target5418_vs_random_nz_surrogate_reward`
- `analysis\diagnosis_outputs\diginetica_cem_mixed_ft6500_paired_relative_mrr_recall_10_20_target14514_vs_random_nz_surrogate_reward`
