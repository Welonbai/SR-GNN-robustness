# Analysis Pipeline

Run all commands from the repository root.

Internal layout:

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
  --config analysis/configs/views/attack_vs_victim_metrics.yaml
```

This writes:

- `results/comparisons/<comparison_id>/<view_name>/table.csv`
- `results/comparisons/<comparison_id>/<view_name>/meta.json`
- The view config uses one `output` field for the final bundle directory.

## 4. Render The PNG

```powershell
python analysis/pipeline/report_table_renderer.py `
  --input-dir results/comparisons/<comparison_id>/<view_name> `
  --config analysis/configs/render/default_slide_png.yaml
```

This writes:

- `results/comparisons/<comparison_id>/<view_name>/render.png`
