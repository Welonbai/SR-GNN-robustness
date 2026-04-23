# Analysis Operator Playbook

Run all commands from the repository root.

This playbook is the default operating procedure for future analysis requests.
Use it before re-reading the full analysis implementation. Prefer reusing the
existing YAML configs listed here when the requested comparison matches their
scope.

## 1. Minimal Questions To Ask

Ask only what is needed to avoid running the wrong comparison.

Required:

- Which attack methods should be compared?
- Which victims should be included? Default: `srgnn`, `miasrec`, `tron`.
- Which target slice should be used? Default: `largest_complete_prefix` with `requested_target_count: 3`.

Ask only when relevant:

- Should results be split by `target_item`? Default: yes.
- Should a clean baseline relative transform be used? Default: no unless `clean` is part of the comparison.

Do not ask again if the user explicitly says to use the default analysis setup.

## 2. Default Example: Prefix vs Shared Policy

This is the current default example and can be rerun without regenerating YAML.

Comparison scope:

- Dataset: `diginetica`
- Attack methods: `prefix_nonzero_when_possible` vs `position_opt_shared_policy`
- Victims: `srgnn`, `miasrec`, `tron`
- Slice policy: `largest_complete_prefix`
- Requested target count: `3`
- Split: one report per `target_item`
- Metrics: `recall`, `mrr`
- K values: `10`, `20`, `30`
- Metric scopes: `ground_truth`, `targeted`
- Clean-relative transform: disabled

Existing YAML configs:

- `analysis/configs/long_csv/diginetica_prefix_vs_shared_policy.yaml`
- `analysis/configs/comparisons/diginetica_prefix_vs_shared_policy.yaml`
- `analysis/configs/views/prefix_vs_shared_policy_by_target_item.yaml`
- `analysis/configs/render/prefix_vs_shared_policy_by_target_item.yaml`

Commands:

```powershell
python analysis/pipeline/long_csv_generator.py --config analysis/configs/long_csv/diginetica_prefix_vs_shared_policy.yaml

python analysis/pipeline/compare_runs.py --config analysis/configs/comparisons/diginetica_prefix_vs_shared_policy.yaml

python analysis/pipeline/view_table_builder.py --config analysis/configs/views/prefix_vs_shared_policy_by_target_item.yaml

python analysis/pipeline/report_table_renderer.py --bundle-parent-dir results/comparisons/diginetica_prefix_vs_shared_policy --config analysis/configs/render/prefix_vs_shared_policy_by_target_item.yaml
```

Expected outputs:

- `results/runs/diginetica_prefix_nonzero_prefix_vs_shared_policy/`
- `results/runs/diginetica_position_opt_shared_policy_prefix_vs_shared_policy/`
- `results/comparisons/diginetica_prefix_vs_shared_policy/merged_long_table.csv`
- `results/comparisons/diginetica_prefix_vs_shared_policy/prefix_vs_shared_policy_by_target_item__target_item_*/render.png`

## 3. Reuse Existing YAML When Possible

Reuse existing YAML directly when:

- the same attack methods are being compared
- the same victim set is requested
- the same target slicing policy is requested
- the same metrics and render shape are acceptable
- the `summary_current.json` paths in the long CSV config point to the intended run groups

Before running, verify the run groups if there is any doubt:

```powershell
Get-ChildItem outputs\runs\diginetica\attack_prefix_nonzero_when_possible -Directory | Sort-Object LastWriteTime -Descending | Select-Object Name,LastWriteTime

Get-ChildItem outputs\runs\diginetica\attack_position_opt_shared_policy_ratio1 -Directory | Sort-Object LastWriteTime -Descending | Select-Object Name,LastWriteTime
```

Check completion if needed:

```powershell
$p='outputs\runs\diginetica\attack_position_opt_shared_policy_ratio1\run_group_c1835ab73f\run_coverage.json'
$j=Get-Content $p -Raw | ConvertFrom-Json
$rows=@()
foreach($t in $j.cells.PSObject.Properties){ foreach($v in $t.Value.PSObject.Properties){ $rows += [pscustomobject]@{ target=$t.Name; victim=$v.Name; status=$v.Value.status } } }
$rows | Group-Object status | Select-Object Name,Count
```

## 4. When To Create New YAML

Create a new four-file YAML set only when the requested comparison differs in a durable way:

- different attack methods
- different victim subset
- different target count or slice policy
- different output naming
- different view shape, metrics, or render style
- clean-relative GT transform is required or should be disabled

Use this naming pattern:

- `analysis/configs/long_csv/diginetica_<comparison_id>.yaml`
- `analysis/configs/comparisons/diginetica_<comparison_id>.yaml`
- `analysis/configs/views/<comparison_id>_by_target_item.yaml`
- `analysis/configs/render/<comparison_id>_by_target_item.yaml`

The comparison ID should also become the output directory:

- `results/comparisons/diginetica_<comparison_id>/`

## 5. Clean Baseline Rule

Enable `ground_truth_relative_to_clean` only when `clean` is included in the comparison input.

For attack-method-only comparisons such as prefix vs shared policy, keep it disabled:

```yaml
ground_truth_relative_to_clean:
  enabled: false
```

If clean-relative GT is enabled, the view builder needs clean rows available for pairing.

## 6. Batch-Size Fairness Rule

Do not mix formal MiaSRec comparisons across different batch-size regimes unless the user explicitly accepts that limitation.

Current practical rule:

- `prefix` and `shared_policy` can be compared together because they were rerun after the MiaSRec batch-size change.
- Older `clean`, `dpsbr`, and `random` MiaSRec results from the previous batch-size setting should not be included in the same formal MiaSRec comparison.
- `srgnn` and `tron` may still be compared if their own training settings did not change.

## 7. Standard Validation

After running analysis, report the important outputs and failures.

Minimum success checks:

- `long_csv_generator.py` writes all expected run bundles.
- `compare_runs.py` writes `merged_long_table.csv`.
- `view_table_builder.py` writes the expected number of report bundles.
- `report_table_renderer.py` writes `render.png` files.

If render fails due to `--bundle-parent-dir`, list the comparison directory and rerun with the parent directory that directly contains the split report bundle folders.
