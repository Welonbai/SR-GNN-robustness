# Analysis Pipeline

Run all commands from the repository root.

## Quick Start

1. Generate one per-run long table.

```powershell
python analysis/pipeline/long_csv_generator.py `
  --summary outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json `
  --output-name diginetica_dpsbr_example
```

2. Merge multiple runs into one comparison bundle.

```powershell
python analysis/pipeline/compare_runs.py `
  --config analysis/configs/comparisons/diginetica_attack_compare.yaml
```

3. Build split view bundles from the comparison bundle.

```powershell
python analysis/pipeline/view_table_builder.py `
  --config analysis/configs/views/attack_vs_victim_metrics_split_by_target_item.yaml
```

4. Render all direct child view bundles to PNG.

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-parent-dir results/comparisons/clean_vs_dpsbr_vs_random_nonzero_vs_prefix_nonzero_diginetica_popular_0.1size `
  --config analysis/configs/render/attack_vs_victim_metrics_split_by_target_item.yaml
```

## Layout

- `analysis/pipeline/`: implementation files for the CLI stages
- `analysis/utils/`: shared helpers
- `analysis/configs/`: YAML configs

## Example Inputs

- `outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json`
- `outputs/runs/diginetica/attack_random_nonzero_when_possible/eval_1b8a0c10c9/summary_random_nonzero_when_possible.json`

## 1. Generate Per-Run Long Tables

`long_csv_generator.py` is not YAML-driven. It reads:

- one `summary*.json` under `outputs/`
- the sibling `resolved_config.json` in the same run directory

```powershell
python analysis/pipeline/long_csv_generator.py `
  --summary outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json
```

```powershell
python analysis/pipeline/long_csv_generator.py `
  --summary outputs/runs/diginetica/attack_random_nonzero_when_possible/eval_1b8a0c10c9/summary_random_nonzero_when_possible.json
```

Optional custom folder name:

```powershell
python analysis/pipeline/long_csv_generator.py `
  --summary outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json `
  --output-name my_custom_run_name
```

`--name` is kept as a backward-compatible alias for `--output-name`.
If `--output-name` is omitted, the folder name is derived from the summary path under `outputs/`.

Outputs are written to `results/runs/<output_name>/` and include:

- `long_table.csv`
- `inventory.json`
- `manifest.json`
- `source_resolved_config.json`

`long_table.csv` uses the canonical columns:

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

`inventory.json` records column-level availability and unique values for inspection, including fields
such as `target_item`, `metric`, and `k`. `manifest.json` records the source summary path, the
source `resolved_config.json` path, the generated files, and the row count.

## 2. Merge Runs For Comparison

```powershell
python analysis/pipeline/compare_runs.py `
  --config analysis/configs/comparisons/diginetica_attack_compare.yaml
```

This writes:

- `results/comparisons/<comparison_id>/merged_long_table.csv`
- `results/comparisons/<comparison_id>/manifest.json`
- `results/comparisons/<comparison_id>/inventory.json`

## 3. Build A View Table

```powershell
python analysis/pipeline/view_table_builder.py `
  --config analysis/configs/views/attack_vs_victim_metrics_11169.yaml
```

This writes:

- `results/comparisons/<comparison_id>/<view_name>/table.csv`
- `results/comparisons/<comparison_id>/<view_name>/meta.json`
- The view config uses one `output` field for the final bundle directory.
- When `split_by` is set, the builder writes multiple sibling bundles such as
  `results/comparisons/<comparison_id>/<view_name>__target_item_11169/`.

Optional view YAML fields:

- `auto_context: true|false`
  Adds singleton hidden columns to `meta.json["context"]` after filtering. Default: `false`.
- `require_unique_cells: true|false`
  Fails before pivoting if one output cell would aggregate multiple source rows. Default: `false`.
- `split_by: [<column_1>, <column_2>, ...]`
  After applying `filters`, splits the remaining rows by unique combinations of those columns and
  writes one bundle per split assignment.

For split-generated bundles, `meta.json` keeps both:

- `filters`
  The original filters from the view spec.
- `effective_filters`
  The actual per-bundle filters, i.e. `filters` plus the bundle's `split_values`.

Example:

```yaml
input: results/comparisons/<comparison_id>/merged_long_table.csv
output: results/comparisons/<comparison_id>/attack_vs_victim_metrics_target_11169

filters:
  target_item: 11169
  metric: [precision, recall, mrr, ndcg]
  k: [5, 10, 15, 20, 25, 30, 40, 50]

rows:
  - attack_method

cols:
  - victim_model
  - metric
  - k

value_col: value
agg: first
auto_context: true
require_unique_cells: true
```

Split example:

```yaml
input: results/comparisons/<comparison_id>/merged_long_table.csv
output: results/comparisons/<comparison_id>/attack_vs_victim_metrics

filters:
  metric: [precision, recall, mrr, ndcg]
  k: [5, 10, 15, 20, 25]

split_by:
  - target_item

rows:
  - victim_model

cols:
  - metric_name
  - metric_scope
  - attack_method

value_col: value
agg: first
auto_context: true
require_unique_cells: true
```

This produces bundles like:

- `results/comparisons/<comparison_id>/attack_vs_victim_metrics__target_item_11169/`
- `results/comparisons/<comparison_id>/attack_vs_victim_metrics__target_item_23467/`

Each bundle still contains:

- `table.csv`
- `meta.json`

## 4. Render The PNG

```powershell
python analysis/pipeline/report_table_renderer.py `
  --config analysis/configs/render/default_slide_png.yaml
```

This writes:

- `results/comparisons/<comparison_id>/<view_name>/render.png`

Single-bundle CLI:

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-dir results/comparisons/<comparison_id>/<view_name> `
  --config analysis/configs/render/default_slide_png.yaml
```

Batch render all direct child bundles under one parent directory:

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-parent-dir results/comparisons/<comparison_id> `
  --config analysis/configs/render/default_slide_png.yaml
```

Batch mode only inspects direct child directories. A valid bundle is any child directory containing
both `table.csv` and `meta.json`. Each rendered bundle writes its own `render.png`.

Example for `split_by: [target_item]` bundles:

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-parent-dir results/comparisons/clean_vs_dpsbr_vs_random_nonzero_diginetica_popular_0.1size `
  --config analysis/configs/render/attack_vs_victim_metrics_split_by_target_item.yaml
```

This renders sibling bundles such as:

- `attack_vs_victim_metrics_split_by_target_item__target_item_11169/`
- `attack_vs_victim_metrics_split_by_target_item__target_item_23467/`
- `attack_vs_victim_metrics_split_by_target_item__target_item_26471/`

Optional render YAML fields:

- `input_dir`
  Path to the view bundle directory containing `table.csv` and `meta.json`.
- `table.display_alias`
  Maps raw `table.csv` column names to display-only header labels in the rendered PNG.
- `table.value_alias`
  Maps raw cell values to display-only aliases per column in the rendered PNG.

Backward-compatible CLI override:

- `--bundle-dir`
  Overrides the single-bundle input directory from the render YAML when needed.
- `--input-dir`
  Legacy alias for `--bundle-dir`.
- `--bundle-parent-dir`
  Renders every valid direct child bundle under one parent directory with the same render config.

Example:

```yaml
input_dir: results/comparisons/<comparison_id>/<view_name>

table:
  font_size: 12
  round_digits: 6
  text_color: black
  show_grid: true
  auto_shrink: false
  wrap_text: false
  cell_align: center
  display_alias:
    "attack_method": "Attack"
    "miasrec | precision | 5": "MIA | P@5"
    "srgnn | mrr | 25": "SRGNN | MRR@25"
  value_alias:
    attack_method:
      dpsbr_baseline: "DP-SBR"
      random_nonzero_when_possible: "RND-NZ"
```
