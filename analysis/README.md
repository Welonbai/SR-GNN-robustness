# Analysis Pipeline

Run all commands from the repository root.

## 1. Quick Commands

### 1.1 Generate Slice-Aware Long Tables

```powershell
python analysis/pipeline/long_csv_generator.py `
  --config analysis/configs/long_csv/diginetica_attack_compare.yaml
```

### 1.2 Compare Compatible Bundles

```powershell
python analysis/pipeline/compare_runs.py `
  --config analysis/configs/comparisons/diginetica_attack_compare.yaml
```

### 1.3 Build View Bundles

```powershell
python analysis/pipeline/view_table_builder.py `
  --config analysis/configs/views/attack_vs_victim_metrics_split_by_target_item.yaml
```

### 1.4 Render From Propagated Metadata

Single bundle:

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-dir results/comparisons/diginetica_attack_compare/attack_vs_victim_metrics_split_by_target_item/<target_item_bundle> `
  --config analysis/configs/render/attack_vs_victim_metrics_split_by_target_item.yaml
```

Batch render all direct child bundles:

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-parent-dir results/comparisons/diginetica_attack_compare/attack_vs_victim_metrics_split_by_target_item `
  --config analysis/configs/render/attack_vs_victim_metrics_split_by_target_item.yaml
```

### 1.5 Resolve Diagnosis Run Bundles

Use this when you already know the completed `run_root` for each method and want one
sanity-checked manifest that points at the fixed artifact layout under each run group.

```powershell
python -m analysis.utils.run_bundle_loader `
  --config analysis/configs/diagnosis/diginetica_run_bundle_example.yaml `
  --output-json analysis/diagnosis_outputs/diginetica_run_bundle_example/resolved_manifest.json
```

Fill paths here:

- `analysis/configs/diagnosis/diginetica_run_bundle_example.yaml`
- replace only `methods.*.run_root` for the completed runs you want to compare
- the loader derives sibling artifacts like `summary_current.json`, `resolved_config.json`,
  `artifact_manifest.json`, `key_payloads.json`, `run_coverage.json`, and target-level
  artifacts under `targets/<target_item>/...`

## 2. Inputs and Trust Model

This pipeline now assumes the appendable run-group architecture:

- run-time state is stored under `outputs/runs/<dataset>/<experiment>/<run_group_key>/`
- analysis resolves a comparable slice before rendering
- comparison validates slice compatibility by default
- rendering consumes propagated metadata and stays presentation-only

Authoritative runtime inputs:

- `summary_current.json` for metric payload flattening
- `run_coverage.json` for cell completion truth
- `target_registry.json` for canonical target order

Authoritative analysis metadata:

- `slice_manifest.json` for one generated slice

Non-authoritative runtime/debug artifacts:

- `progress.json`
- legacy-style `summary_<run_type>.json`

Legacy analysis note:

- old `results/runs/diginetica__...__summary_*` bundles without `slice_manifest.json` are pre-appendable analysis outputs
- keep them for reference if you want, but do not feed them into strict comparison
- regenerate fresh slice-aware bundles from current run-group outputs before comparing

## 3. Long CSV Notes

The config supports shared defaults and multiple jobs:

- `defaults.slice_policy`
- `defaults.requested_victims`
- `defaults.requested_target_count`
- `jobs[*].summary`
- `jobs[*].output_name`
- `jobs[*].attack_method_override`
- `jobs[*].slice_policy`
- `jobs[*].requested_victims`
- `jobs[*].requested_target_count`

Default slice policy:

- `largest_complete_prefix`

Outputs under `results/runs/<bundle_name>/`:

- `long_table.csv`
- `inventory.json`
- `manifest.json`
- `slice_manifest.json`
- `source_resolved_config.json`

`long_table.csv` keeps the canonical row schema:

- `run_id`
- `dataset`
- `attack_method`
- `victim_model`
- `target_item`
- `target_type`
- `attack_size`
- `poison_model`
- `fake_session_generation_topk`
- `replacement_topk_ratio`
- `metric`
- `k`
- `value`

## 4. Comparison Notes

Comparison now loads each source bundle's sibling `slice_manifest.json`.

Default behavior:

- strict slice compatibility
- reject mismatched `slice_policy`
- reject mismatched `fairness_safe`
- reject mismatched `requested_victims`
- reject mismatched `selected_targets`
- reject mismatched `selected_target_count`

Relaxed comparison exists only for debug workflows and must be requested explicitly in the comparison spec.

Outputs under `results/comparisons/<comparison_id>/`:

- `merged_long_table.csv`
- `inventory.json`
- `manifest.json`
- `slice_manifest.json`

## 5. View Notes

The view builder:

- reads run or comparison bundles
- loads sibling `manifest.json` / `slice_manifest.json`
- propagates normalized slice metadata into bundle `meta.json`
- adds flattened `slice_context` for rendering
- does not recompute slice/fairness decisions

Typical outputs:

- `table.csv`
- `meta.json`

When `split_by` is configured, multiple sibling bundles are written.

## 6. Render Notes

The renderer:

- reads `table.csv` and `meta.json`
- can display slice metadata from `context` / `slice_context`
- can reorder row/column dimension values at render time with `table.dimension_value_orders`
- does not load `run_coverage.json` or `target_registry.json`
- does not recompute fairness

## 7. Recommended End-to-End Flow

1. Produce or append a run group with the attack pipeline.
2. Generate one slice-aware long-table bundle from `summary_current.json`.
3. Compare only strict-compatible bundles.
4. Build one or more view bundles from the comparison output.
5. Render final PNGs from the view bundles.

## 8. Related Docs

- [Analysis Operator Playbook](../docs/analysis_operator_playbook.md)
- [Appendable Experiment Operator Guide](../docs/operator_workflow_guide.md)
- [Analysis Slice Rules](../docs/analysis_slice_rules.md)
- [Legacy Migration Tool Usage](../docs/migration_tool_usage.md)
