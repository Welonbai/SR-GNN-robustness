# Analysis Pipeline

Run all commands from the repository root.

This pipeline now assumes the appendable run-group architecture:

- run-time state is stored under `outputs/runs/<dataset>/<experiment>/<run_group_key>/`
- analysis resolves a comparable slice before rendering
- comparison validates slice compatibility by default
- rendering consumes propagated metadata and stays presentation-only

## 1. Inputs and Trust Model

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

## 2. Generate A Slice-Aware Long Table

```powershell
python analysis/pipeline/long_csv_generator.py `
  --summary outputs/runs/<dataset>/<experiment>/<run_group_key>/summary_current.json
```

Optional flags:

- `--slice-policy largest_complete_prefix|intersection_complete|all_available`
- `--victim <victim_name>` repeated for explicit victim subsets
- `--target-count <N>`
- `--output-name <bundle_name>`

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

## 3. Compare Compatible Bundles

```powershell
python analysis/pipeline/compare_runs.py `
  --config analysis/configs/comparisons/<comparison>.yaml
```

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

## 4. Build View Bundles

```powershell
python analysis/pipeline/view_table_builder.py `
  --config analysis/configs/views/<view>.yaml
```

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

## 5. Render From Propagated Metadata

Single bundle:

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-dir results/<...>/<view_bundle> `
  --config analysis/configs/render/<render>.yaml
```

Batch render all direct child bundles:

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-parent-dir results/<...>/<parent_dir> `
  --config analysis/configs/render/<render>.yaml
```

The renderer:

- reads `table.csv` and `meta.json`
- can display slice metadata from `context` / `slice_context`
- does not load `run_coverage.json` or `target_registry.json`
- does not recompute fairness

## 6. Recommended End-to-End Flow

1. Produce or append a run group with the attack pipeline.
2. Generate one slice-aware long-table bundle from `summary_current.json`.
3. Compare only strict-compatible bundles.
4. Build one or more view bundles from the comparison output.
5. Render final PNGs from the view bundles.

## 7. Related Docs

- [Appendable Experiment Operator Guide](../docs/operator_workflow_guide.md)
- [Analysis Slice Rules](../docs/analysis_slice_rules.md)
- [Legacy Migration Tool Usage](../docs/migration_tool_usage.md)
