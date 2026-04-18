# AGENTS.md

## Purpose

This repository is being refactored from a **batch-style experiment runner** into an **appendable experiment container** for targeted poisoning experiments in Session-Based Recommendation (SBR).

The implementation must support:

1. **Target append**: grow a target cohort over time without creating a new run group.
2. **Victim append**: add new victim models to an existing run group without creating a new run group.
3. **Coverage-aware analysis**: downstream analysis must know which `(target_item, victim_name)` cells are complete.
4. **Slice-aware reporting**: tables and rendered reports must be generated from an explicitly selected comparable slice.
5. **Legacy migration**: previously completed results must remain reusable through formal import/migration, not by preserving the old run identity semantics.

## Read these files before changing code

Before modifying code, read the following documents in order:

1. `docs/appendable_experiment_architecture.md`
2. `docs/appendable_experiment_execplan.md`
3. `docs/analysis_slice_rules.md`
4. `docs/legacy_migration.md`

These documents define the target architecture, execution semantics, analysis rules, and migration strategy. They are the source of truth for this refactor.

## Core architectural rules

### 1) Separate identities by responsibility

The implementation must preserve the following separation of identities:

- **Split identity**: dataset split only.
- **Target cohort identity**: the stable identity of a target cohort.
- **Run group identity**: the appendable experiment container.
- **Cell identity**: one result cell for one `(target_item, victim_name)` pair.

Do **not** collapse these identities back into a single evaluation/run key.

### 2) Do not encode append progress into identity

The following values must **not** define the stable identity of a target cohort or run group:

- current target count / requested prefix length
- enabled victim set
- temporary resume/retry state
- execution batch boundaries

Append progress belongs in run-time artifacts such as coverage, execution logs, and registries.

### 3) Keep target and victim append semantics explicit

- Adding more targets to the same sampled cohort must extend the same stable ordered cohort.
- Adding more victim models must populate missing cells in the same run group.
- Re-running should skip already completed cells unless an explicit overwrite path is designed.

### 4) Analysis must be coverage-aware before rendering

Coverage selection logic must not be implemented inside the final report renderer.

Expected responsibility split:

- run-time pipeline stores append state and artifacts
- analysis resolves a comparable slice using coverage + registry artifacts
- renderer displays already-selected data and metadata

### 5) Preserve reusable lower-level caches where possible

The refactor should preserve compatibility for:

- split-level canonical artifacts
- shared attack-generation artifacts

These are expected to remain reusable across the architecture change when their underlying semantics are unchanged.

## Required new artifacts

The refactor must introduce and maintain the following artifacts:

- `target_registry.json`
- `run_coverage.json`
- `execution_log.json`
- `summary_current.json`

Analysis must also produce a slice-specific artifact:

- `slice_manifest.json`

See architecture and slice rule docs for schema expectations.

## Expected implementation areas

The following code areas are expected to change substantially:

- `attack/common/paths.py`
- `attack/common/artifact_io.py`
- `attack/pipeline/core/pipeline_utils.py`
- `attack/pipeline/core/orchestrator.py`
- `attack/pipeline/core/victim_execution.py`
- `analysis/pipeline/long_csv_generator.py`

The following areas are expected to change lightly or mainly for metadata propagation:

- `attack/pipeline/core/evaluator.py`
- `analysis/pipeline/compare_runs.py`
- `analysis/pipeline/view_table_builder.py`
- `analysis/pipeline/report_table_renderer.py`

A migration utility is also expected to be added.

## Guardrails

### Do not do these things

- Do not reintroduce a run identity that depends on the full enabled victim set.
- Do not reintroduce a target identity that depends on the current requested target count.
- Do not make the renderer infer comparable slices from raw summary data.
- Do not silently compare runs built from different slices without explicit metadata or validation.
- Do not implement legacy compatibility by scattering old-key fallback logic across the main execution path.

### Prefer these approaches

- Prefer explicit registry and coverage artifacts over implicit inference.
- Prefer deterministic ordered target cohorts for sampled target selection.
- Prefer cell-level completion tracking.
- Prefer manifest-based compatibility checks in analysis/comparison.
- Prefer a dedicated migration/import path for legacy results.

## Testing expectations

Every phase should leave the repository in a runnable, testable state.

At minimum, the implementation should eventually support validation for:

1. append targets into the same run group
2. append victims into the same run group
3. mixed target + victim append
4. resume after partial completion
5. slice-aware long table generation
6. comparison-time slice compatibility checks
7. legacy migration import

Where practical, add targeted tests or executable validation scripts instead of relying only on manual inspection.

## Working style for this refactor

When implementing a phase:

1. confirm the phase scope in `docs/appendable_experiment_execplan.md`
2. implement only that scope
3. keep the end-state architecture consistent with `docs/appendable_experiment_architecture.md`
4. update or add artifacts/schema helpers as needed
5. validate the phase-specific acceptance criteria

If a local design choice is unclear, choose the option that best preserves:

- appendable run groups
- deterministic target cohort ordering
- explicit cell coverage
- analysis-time slice correctness

## Final objective

The final system should behave like this:

- a **run group** is a durable experiment container
- a **target cohort** is a stable ordered pool of targets
- a **cell** is one `(target_item, victim_name)` result
- later executions add missing cells instead of creating unrelated runs
- analysis selects a fair comparable slice from coverage artifacts
- rendering displays those results with explicit slice metadata
