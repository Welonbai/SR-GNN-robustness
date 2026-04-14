# Analysis Pipeline

Run all commands from the repository root.

## Quick Start

1. Generate one per-run long table.

```powershell
python analysis/pipeline/long_csv_generator.py `
  --summary outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json `
  --output-name diginetica_dpsbr_example
```

2. Merge multiple runs into one comparison table.

```powershell
python analysis/pipeline/compare_runs.py `
  --config analysis/configs/comparisons/diginetica_attack_compare.yaml
```

3. Build one filtered pivot table view.

```powershell
python analysis/pipeline/view_table_builder.py `
  --config analysis/configs/views/attack_vs_victim_metrics_11169.yaml
```

4. Render the view bundle to PNG.

```powershell
python analysis/pipeline/report_table_renderer.py `
  --config analysis/configs/render/default_slide_png.yaml
```

## Layout

- `analysis/pipeline/`: implementation files for the CLI stages
- `analysis/utils/`: shared helpers
- `analysis/configs/`: YAML configs

## Example Inputs

- `outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json`
- `outputs/runs/diginetica/attack_random_nonzero_when_possible/eval_1b8a0c10c9/summary_random_nonzero_when_possible.json`

## 1. Generate Per-Run Long Tables

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

Outputs are always written to `results/runs/<output_name>/`.
Each run bundle also includes `inventory.json` with columns, unique counts, and available values such as `target_item`, `metric`, and `k`.

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

Optional view YAML fields:

- `auto_context: true|false`
  Adds singleton hidden columns to `meta.json["context"]` after filtering. Default: `false`.
- `require_unique_cells: true|false`
  Fails before pivoting if one output cell would aggregate multiple source rows. Default: `false`.

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

## 4. Render The PNG

```powershell
python analysis/pipeline/report_table_renderer.py `
  --config analysis/configs/render/default_slide_png.yaml
```

This writes:

- `results/comparisons/<comparison_id>/<view_name>/render.png`

Optional render YAML fields:

- `input_dir`
  Path to the view bundle directory containing `table.csv` and `meta.json`.
- `table.display_alias`
  Maps raw `table.csv` column names to display-only header labels in the rendered PNG.
- `table.value_alias`
  Maps raw cell values to display-only aliases per column in the rendered PNG.

Backward-compatible CLI override:

- `--input-dir`
  Overrides `input_dir` from the render YAML when needed.

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
