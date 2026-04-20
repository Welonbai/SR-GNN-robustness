# Appendable Experiment Operator Guide

Run all commands from the repository root.

## 1. Mental Model

The repository now uses an appendable experiment container model:

- `split_key`: canonical dataset split identity
- `target_cohort_key`: stable ordered target cohort identity
- `run_group_key`: durable experiment container identity
- cell: one `(target_item, victim_name)` result

A run group is not one batch invocation. It is the container that accumulates completed cells over time.

## 2. Artifact Trust Model

Authoritative artifacts:

| Artifact | Scope | Purpose |
| --- | --- | --- |
| `target_registry.json` | target cohort shared dir | authoritative ordered target cohort and current materialized prefix |
| `run_coverage.json` | run-group root | authoritative target × victim completion matrix |
| `execution_log.json` | run-group root | authoritative append/retry execution history |

Snapshot or debug-only artifacts:

| Artifact | Scope | Purpose |
| --- | --- | --- |
| `summary_current.json` | run-group root | current snapshot rebuilt from authoritative state |
| `progress.json` | run-group root | operator/debug visibility only; never scheduling authority |
| `summary_<run_type>.json` | run-group root | legacy-style compatibility snapshot |

Analysis-time artifacts:

| Artifact | Scope | Purpose |
| --- | --- | --- |
| `slice_manifest.json` | results bundle | explicit comparable slice selection used for one long-table/comparison output |

## 3. Native Execution Workflow

### 3.1 Choose the run entrypoint

Examples:

- Clean baseline: `python attack/pipeline/runs/run_clean.py --config attack/configs/diginetica_clean.yaml`
- DP-SBR baseline: `python attack/pipeline/runs/run_dp_sbr_baseline.py --config attack/configs/diginetica_attack_dpsbr.yaml`
- Position optimization MVP: `python attack/pipeline/runs/run_position_opt_mvp.py --config attack/configs/diginetica_attack_position_optimization_reward.yaml`

Other run entrypoints under `attack/pipeline/runs/` follow the same run-group model.

### 3.2 Initial run

Run once with the desired stable cohort and run-group settings.

For sampled targets:

- `targets.mode: sampled`
- `targets.bucket: <popular|unpopular|all>`
- `targets.count: N`

This creates or updates:

- `outputs/shared/<dataset>/target_cohorts/<target_cohort_key>/target_registry.json`
- `outputs/runs/<dataset>/<experiment>/<run_group_key>/...`

### 3.3 Target append

To append more targets for the same sampled cohort:

1. Keep the same split, attack, evaluation, and target cohort settings.
2. Increase only `targets.count`.
3. Re-run the same entrypoint.

Behavior:

- the run group stays the same
- only newly requested target-prefix cells are scheduled
- completed cells are skipped

### 3.4 Victim append

To append new victims to the same run group:

1. Keep the same split, attack, evaluation, and target cohort settings.
2. Extend `victims.enabled`.
3. Make sure `victims.params` and `victims.runtime` include the new victim entries.
4. Re-run the same entrypoint.

Behavior:

- the run group stays the same
- new victim cells are scheduled across the current requested target prefix
- completed old-victim cells are skipped

### 3.5 Resume and retry

If an execution is interrupted or some cells fail:

1. Re-run the same entrypoint with the same intended request.
2. The orchestrator resumes from `run_coverage.json` and `execution_log.json`.

Rules:

- `completed` cells are skipped
- `failed` cells are eligible
- `requested` / `pending` cells are eligible
- `summary_current.json` and `progress.json` are not used as scheduling authority

## 4. Analysis Workflow

### 4.1 Generate a slice-aware long table

```powershell
python analysis/pipeline/long_csv_generator.py `
  --config analysis/configs/long_csv/diginetica_attack_compare.yaml
```

The long_csv config supports shared defaults and multiple jobs:

- `defaults.slice_policy`
- `defaults.requested_victims`
- `defaults.requested_target_count`
- `jobs[*].summary`
- `jobs[*].output_name`
- `jobs[*].slice_policy`
- `jobs[*].requested_victims`
- `jobs[*].requested_target_count`

Default policy:

- `largest_complete_prefix`

Outputs under `results/runs/<bundle_name>/`:

- `long_table.csv`
- `manifest.json`
- `inventory.json`
- `slice_manifest.json`
- `source_resolved_config.json`

### 4.2 Compare compatible runs

```powershell
python analysis/pipeline/compare_runs.py `
  --config analysis/configs/comparisons/diginetica_attack_compare.yaml
```

Default comparison behavior:

- strict slice compatibility
- comparison fails if slice policy, fairness safety, requested victims, or selected targets differ

Relaxed comparison is debug-only and must be requested explicitly in the comparison spec.

### 4.3 Build a view bundle

```powershell
python analysis/pipeline/view_table_builder.py `
  --config analysis/configs/views/attack_vs_victim_metrics_split_by_target_item.yaml
```

The view builder:

- consumes run/comparison bundles
- propagates slice metadata into `meta.json`
- does not recompute fairness or slice selection

### 4.4 Render the final table

```powershell
python analysis/pipeline/report_table_renderer.py `
  --bundle-dir results/comparisons/diginetica_attack_compare/attack_vs_victim_metrics_split_by_target_item/<target_item_bundle> `
  --config analysis/configs/render/attack_vs_victim_metrics_split_by_target_item.yaml
```

The renderer:

- reads `table.csv` and `meta.json`
- displays already-propagated slice metadata
- does not recompute slice eligibility or fairness

## 5. Expected End-to-End Operator Sequence

1. Run a native experiment entrypoint.
2. Append target prefix by increasing `targets.count` when needed.
3. Append victims by extending `victims.enabled` plus matching params/runtime.
4. Re-run after interruption or failure; let authoritative state drive resume.
5. Generate a slice-aware long table from `summary_current.json`.
6. Compare only compatible bundles under strict mode by default.
7. Build view bundles.
8. Render from the propagated bundle metadata.

## 6. What Not To Do

- Do not treat `summary_current.json` as the only source of truth.
- Do not treat `progress.json` as authoritative.
- Do not compare long tables without their `slice_manifest.json`.
- Do not append victims by editing only `victims.enabled`; the new victim also needs config.
- Do not use legacy `evaluation_key` or `target_selection_key` semantics as the runtime identity model.

## 7. Related Docs

- [Appendable Experiment Architecture](./appendable_experiment_architecture.md)
- [Appendable Experiment Execution Plan](./appendable_experiment_execplan.md)
- [Analysis Slice Rules](./analysis_slice_rules.md)
- [Legacy Migration Plan](./legacy_migration.md)
- [Legacy Migration Tool Usage](./migration_tool_usage.md)
