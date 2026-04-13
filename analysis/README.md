# Analysis Pipeline

Run all commands from the repository root.

## Example Inputs

- `outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json`
- `outputs/runs/diginetica/attack_random_nonzero_when_possible/eval_1b8a0c10c9/summary_random_nonzero_when_possible.json`

## 1. Generate Per-Run Long Tables

```powershell
python analysis/long_csv_generator.py `
  --summary outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json
```

```powershell
python analysis/long_csv_generator.py `
  --summary outputs/runs/diginetica/attack_random_nonzero_when_possible/eval_1b8a0c10c9/summary_random_nonzero_when_possible.json
```

Optional custom folder name:

```powershell
python analysis/long_csv_generator.py `
  --summary outputs/runs/diginetica/attack_dpsbr/eval_ea7308a8e0/summary_dpsbr_baseline.json `
  --output-name my_custom_run_name
```

Outputs are always written to `results/runs/<output_name>/`.

## 2. Merge Runs For Comparison

```powershell
python analysis/compare_runs.py `
  --config analysis/specs/comparisons/diginetica_attack_compare.yaml
```

This writes:

- `results/comparisons/diginetica_attack_compare/merged_long_table.csv`
- `results/comparisons/diginetica_attack_compare/manifest.json`

## 3. Build A View Table

```powershell
python analysis/view_table_builder.py `
  --config analysis/specs/views/attack_vs_victim_metrics.yaml
```

This writes:

- `results/comparisons/diginetica_attack_compare/attack_vs_victim_metrics/table.csv`
- `results/comparisons/diginetica_attack_compare/attack_vs_victim_metrics/meta.json`

## 4. Render The PNG

```powershell
python analysis/report_table_renderer.py `
  --input-dir results/comparisons/diginetica_attack_compare/attack_vs_victim_metrics `
  --config analysis/specs/render/default_slide_png.yaml
```

This writes:

- `results/comparisons/diginetica_attack_compare/attack_vs_victim_metrics/render.png`
