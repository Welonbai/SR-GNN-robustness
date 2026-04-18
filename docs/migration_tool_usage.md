# Legacy Migration Tool Usage

Run all commands from the repository root.

## 1. Purpose

`attack/tools/migrate_legacy_runs.py` imports legacy batch-era run outputs into the new appendable run-group model.

The result is a real new-architecture run group with:

- `target_registry.json`
- `run_coverage.json`
- `execution_log.json`
- `summary_current.json`
- migrated local cell artifacts under the new run-group layout

It is not a runtime fallback shim.

## 2. CLI

Dry-run inspection:

```powershell
python attack/tools/migrate_legacy_runs.py `
  outputs/runs/<dataset>/<experiment>/<legacy_eval_key> `
  --artifacts-root outputs `
  --dry-run
```

Real migration:

```powershell
python attack/tools/migrate_legacy_runs.py `
  outputs/runs/<dataset>/<experiment>/<legacy_eval_key> `
  --artifacts-root outputs
```

You may also pass a specific legacy `summary_*.json` file instead of the run root.

## 3. Inputs The Tool Uses

The tool inspects as many of these legacy inputs as are available:

- legacy `summary_*.json`
- legacy `artifact_manifest.json`
- legacy `resolved_config.json`
- legacy `key_payloads.json`
- legacy local per-cell artifacts such as:
  - `metrics.json`
  - `predictions.json`
  - `train_history.json`
  - `poisoned_train.txt`
- legacy target-selection artifacts:
  - `selected_targets.json`
  - `target_selection_meta.json`
  - `target_info.json`

Not every optional legacy file is required, but missing metadata may limit what can be reconstructed safely.

## 4. Outputs The Tool Creates

For each successfully migrated legacy run, the tool writes:

- a new run-group root under `outputs/runs/<dataset>/<experiment>/<run_group_key>/`
- a new target-cohort registry under `outputs/shared/<dataset>/target_cohorts/<target_cohort_key>/`
- copied local cell artifacts under:
  - `targets/<target_id>/victims/<victim_name>/...`
- migrated metadata snapshots under:
  - `run_root/migration/legacy_*.json`

It also annotates migrated artifacts with metadata such as:

- `imported_from_legacy`
- migration version
- source legacy paths
- reconstruction notes

## 5. Reconstruction Behavior

### Explicit targets

If the legacy source clearly specifies an explicit target list, the migration preserves that order.

### Sampled targets when only selected targets are known

If the legacy source only proves the selected/materialized target set, the migration is conservative:

- it reconstructs a deterministic ordered target list from the known selected targets
- it records that the order was reconstructed
- it does not claim to know the original full sampled cohort order

This keeps the migrated state honest while still producing a usable run group.

### Coverage

`run_coverage.json` is reconstructed conservatively:

- `completed` requires persisted compatible local cell artifacts
- missing required local artifacts do not become `completed`
- partial cells may be marked `failed`
- absent cells stay absent/requested

### Execution log

The migrated `execution_log.json` contains at least one imported execution record:

- `mode: legacy_import`
- marked as imported/migrated
- records source legacy paths and imported cells

## 6. Conservative Rejections

The tool fails loudly instead of guessing when migration is ambiguous or incompatible.

Examples:

- multiple legacy inputs map to the same new run-group destination
- an existing destination `target_registry.json` conflicts with the migrated registry
- required legacy metadata is too incomplete to infer the new identities safely
- target identifiers are not supported by the current repository assumptions

## 7. After Migration

Once migration completes, treat the result like any native run group.

Typical next steps:

1. inspect `run_coverage.json` and `execution_log.json`
2. append victims or rerun failures using the normal run entrypoints
3. generate a slice-aware long table from `summary_current.json`
4. use strict comparison and normal view/render workflows

Example: native victim append after migration

1. use a config with the same split/cohort/run-group settings
2. extend `victims.enabled`
3. make sure the new victim has matching `victims.params` and `victims.runtime`
4. rerun the normal attack entrypoint

## 8. Related Docs

- [Legacy Migration Plan](./legacy_migration.md)
- [Appendable Experiment Operator Guide](./operator_workflow_guide.md)
