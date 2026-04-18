# Appendable Experiment Execution Plan

This document breaks the architecture refactor into implementation phases.

Each phase should be implemented and validated in sequence. The end of each phase must leave the repository in a coherent state that still points toward the final architecture.

The phase order matters because later phases depend on earlier identity and artifact decisions.

---

## Phase 0 — Preparation and guardrails

### Objective
Establish the design references and protect the refactor from drifting back to the old batch-style semantics.

### Scope
- add repository-level guidance documents
- confirm file ownership areas
- identify current key/path usage sites
- identify legacy assumptions in analysis and orchestration

### Expected files touched
- `AGENTS.md`
- `docs/appendable_experiment_architecture.md`
- `docs/analysis_slice_rules.md`
- `docs/legacy_migration.md`
- optionally local notes/tests scaffolding

### Deliverables
- reference docs committed to the repo
- a clear list of current identity entry points and path builders

### Acceptance criteria
- documentation exists in the repo
- the implementation team can point to the exact files that define current split, target, attack, victim, and run identities

### Out of scope
- no functional code changes required yet

---

## Phase 1 — Identity and path refactor

### Objective
Replace the old evaluation/run identity model with the new layered identity model.

### Scope
Implement the key/path foundation for:
- split identity (preserve existing semantics where possible)
- target cohort identity
- run group identity
- cell-oriented artifact paths

### Required changes

#### `attack/common/paths.py`
- add `target_cohort_key_payload(...)`
- add `target_cohort_key(...)`
- add `run_group_key_payload(...)`
- add `run_group_key(...)`
- add paths for:
  - target cohort directory
  - target registry artifact
  - run coverage artifact
  - execution log artifact
  - current summary artifact
- update run root resolution to use `run_group_key`

#### Path rules
- preserve shared split artifact paths if their semantics do not change
- preserve shared attack-generation artifact paths if their semantics do not change
- keep victim-specific prediction keys for victim cache paths only

### Deliverables
- new identity helper functions
- new path helper functions
- a single run group root path model aligned with the architecture doc

### Acceptance criteria
- code can derive:
  - `split_key`
  - `target_cohort_key`
  - `run_group_key`
  - `victim_prediction_key`
- changing target count alone does not change `target_cohort_key`
- changing enabled victim set alone does not change `run_group_key`

### Out of scope
- no append execution yet
- no slice-aware analysis yet

---

## Phase 2 — Registry, coverage, and execution-state artifacts

### Objective
Introduce the new run-time state artifacts required by appendable execution.

### Scope
Implement artifact IO and state construction for:
- `target_registry.json`
- `run_coverage.json`
- `execution_log.json`
- `summary_current.json`

### Required changes

#### `attack/common/artifact_io.py`
Add load/save helpers for the new artifacts.

#### `attack/pipeline/core/pipeline_utils.py`
Add helpers to:
- initialize target registry
- expand target registry to a requested prefix
- initialize run coverage
- initialize execution log
- rebuild current summary snapshot from state

### Deliverables
- artifact schemas implemented in code
- artifact load/save helpers
- deterministic target-registry generation logic for sampled cohorts

### Acceptance criteria
- the repository can create an empty run group state
- the repository can create a stable ordered target registry for sampled targets
- the repository can persist run coverage and execution logs without executing victims yet
- the repository can rebuild a valid `summary_current.json`

### Out of scope
- orchestrator still may not execute append plans fully
- analysis still may not consume coverage yet

---

## Phase 3 — Target append execution

### Objective
Support growing the target prefix within the same run group.

### Scope
Change the execution pipeline so that requested target count becomes a prefix request over a stable target cohort.

### Required changes

#### `attack/pipeline/core/pipeline_utils.py`
- replace old target resolution semantics with registry-aware prefix semantics
- compute newly requested targets versus already covered targets

#### `attack/pipeline/core/orchestrator.py`
- load or initialize run group state
- build a target-append execution plan
- execute only missing cells for the requested target prefix and current victims
- update coverage and summary incrementally
- append execution records

### Deliverables
- repeatable target append behavior for sampled target cohorts

### Acceptance criteria
- a first run with target count `3` creates cells for prefix `[:3]`
- a second run with target count `6` under the same cohort only adds cells for prefix `[3:6]`
- the run group key remains unchanged across those executions
- coverage reflects exactly which cells were added

### Out of scope
- adding new victims to an existing run group may still be incomplete

---

## Phase 4 — Victim append execution

### Objective
Support adding new victim models to an existing run group without changing run identity.

### Scope
Extend the orchestrator and victim execution logic so newly enabled victims populate missing cells across the current target prefix.

### Required changes

#### `attack/pipeline/core/orchestrator.py`
- extend planning logic to detect missing victim cells over the current requested target prefix
- support mixed plans that contain both target append and victim append work

#### `attack/pipeline/core/victim_execution.py`
- standardize per-cell outputs
- return sufficient metadata for coverage updates
- optionally standardize shared-cell manifest writing

### Deliverables
- victim append support within the same run group

### Acceptance criteria
- starting from a run group with victims `A, B`, enabling victim `C` appends only the missing `C` cells
- the run group key remains unchanged
- existing metrics are not recomputed unless explicitly required
- coverage marks the new victim cells as completed

### Out of scope
- slice-aware analysis may still be pending

---

## Phase 5 — Resume and failure handling

### Objective
Make appendable execution robust to interruption and partial completion.

### Scope
Ensure the new state model supports safe resume semantics.

### Required changes
- write coverage updates at cell granularity
- write enough execution metadata to distinguish completed and failed work
- ensure orchestrator skips completed cells on rerun

### Deliverables
- resumable append execution

### Acceptance criteria
- interrupting a run after partial cell completion does not corrupt the run group
- rerunning resumes only incomplete/failed cells
- summary snapshot can be rebuilt from coverage after interruption

### Out of scope
- no analysis-specific changes required yet

---

## Phase 6 — Slice-aware analysis

### Objective
Make analysis coverage-aware and explicitly slice-driven.

### Scope
Move fair-slice selection into analysis before rendering.

### Required changes

#### `analysis/pipeline/long_csv_generator.py`
Add support to read:
- `summary_current.json`
- `run_coverage.json`
- `target_registry.json`

Add slice resolution inputs:
- requested victims
- slice policy
- optional target count

Generate:
- slice-aware long table
- `slice_manifest.json`

### Deliverables
- coverage-aware long table generation
- explicit slice metadata artifact

### Acceptance criteria
- `largest_complete_prefix` works as specified
- `intersection_complete` works as specified
- `all_available` can be produced for debug mode
- long-table output changes correctly when victim completeness changes

### Out of scope
- final report rendering may still only need minor metadata updates

---

## Phase 7 — Comparison compatibility checks

### Objective
Prevent invalid comparisons across incompatible slices.

### Scope
Strengthen run comparison to validate slice compatibility before merging.

### Required changes

#### `analysis/pipeline/compare_runs.py`
- read `slice_manifest.json` for each source run
- validate compatibility by default
- record merged slice metadata in comparison outputs

### Deliverables
- safe comparison semantics for appendable experiments

### Acceptance criteria
- comparison fails by default when slice manifests are incompatible
- comparison succeeds when slice manifests match
- merged outputs record slice metadata

### Out of scope
- renderer remains mostly a presentation layer

---

## Phase 8 — View/render metadata propagation

### Objective
Expose slice metadata clearly in built views and rendered reports.

### Scope
Pass slice metadata through view-building and rendering.

### Required changes

#### `analysis/pipeline/view_table_builder.py`
- carry slice metadata into bundle meta/context

#### `analysis/pipeline/report_table_renderer.py`
- display slice metadata in titles, subtitles, captions, or footnotes
- preserve existing formatting logic where possible

### Deliverables
- rendered outputs that clearly state the slice basis

### Acceptance criteria
- rendered outputs can display:
  - slice policy
  - selected target count
  - requested victims
- best-value highlighting still works on already-selected comparable data

### Out of scope
- renderer should not perform slice selection itself

---

## Phase 9 — Legacy migration tooling

### Objective
Allow previously completed runs to be reused under the new architecture.

### Scope
Implement import/migration utilities for legacy runs.

### Required changes
Add a migration utility, for example:
- `attack/tools/migrate_legacy_runs.py`

The migration should:
- scan legacy runs
- infer the new split/cohort/run-group identities
- reconstruct registry/coverage/execution metadata
- import completed cells into the new state model

### Deliverables
- migration utility
- migration documentation aligned with `docs/legacy_migration.md`

### Acceptance criteria
- a legacy run can be imported into a new run group
- imported cells appear in coverage as completed
- slice-aware analysis can run on imported state

### Out of scope
- no need to preserve old run-key semantics as the final architecture

---

## Phase 10 — End-to-end validation

### Objective
Validate that the architecture behaves correctly as a whole.

### Scope
Run end-to-end tests covering the intended research workflow.

### Required validation scenarios

1. target append only
2. victim append only
3. mixed target + victim append
4. resume after interruption
5. slice-aware analysis over incomplete victim coverage
6. strict comparison across compatible slices
7. legacy migration followed by new append

### Deliverables
- test results or validation scripts
- confirmation that the repository now supports the intended workflow

### Acceptance criteria
- all core architecture invariants hold in practice

---

## Recommended implementation order

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8
10. Phase 9
11. Phase 10

This order is intentional:
- identity before state
- state before append execution
- append execution before slice-aware analysis
- analysis before metadata-rich rendering
- migration after the new architecture exists

---

## General acceptance rule

A phase is complete only when:

1. its intended behavior exists,
2. its acceptance criteria are satisfied, and
3. its implementation still aligns with the architecture document.

Do not “solve” a local phase by reintroducing old batch-style assumptions.
