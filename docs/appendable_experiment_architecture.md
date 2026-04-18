# Appendable Experiment Architecture

## 1. Problem statement

The current experiment pipeline behaves like a **batch-style run system**:

- a run is tightly coupled to one target selection result
- a run is tightly coupled to one enabled victim set
- append progress is implicitly encoded into keys and directory identity
- downstream analysis assumes that one summary corresponds to one complete comparable result set

This design does not match the intended research workflow.

The intended workflow is:

- gradually append more target items under the same experimental setting
- gradually append more victim models under the same experimental setting
- reuse previously completed target/victim cells without redefining the run identity
- generate fair comparisons only from coverage-aware slices

This document defines the target architecture for that workflow.

---

## 2. Design goals

### Required goals

1. **Target append**
   - A sampled target cohort must remain stable across repeated executions.
   - Increasing the requested target count must extend the same cohort rather than create a new run.

2. **Victim append**
   - Adding a new victim model to an existing experimental setting must populate missing cells in the same run group.

3. **Explicit coverage**
   - The system must explicitly record which `(target_item, victim_name)` cells are complete.

4. **Slice-aware analysis**
   - Analysis must construct comparable slices from explicit coverage metadata.

5. **Durable experiment container**
   - A run group must persist across multiple executions.

6. **Legacy migration**
   - Previously completed runs must be reusable through formal import/migration.

### Non-goals

1. Preserve old run key semantics as the long-term architecture.
2. Make the renderer infer experiment completeness from raw result files.
3. Encode append progress into the run identity.

---

## 3. Terminology

### Split identity
Identity of the dataset split only.

This should continue to represent the canonical train/validation/test partition and related split artifacts.

### Target cohort
A stable, ordered target pool used for one appendable target-selection universe.

Examples:
- popular sampled cohort under one seed
- unpopular sampled cohort under one seed
- fixed explicit target list

### Run group
A durable experiment container for one attack/evaluation setup applied to one target cohort.

A run group is not a single execution. It is the cumulative container that receives new cells over time.

### Cell
The minimal result unit:

`(target_item, victim_name)`

A cell may contain metrics, predictions, training history, poisoned train paths, and related metadata.

### Execution
One invocation of the pipeline that appends new work into an existing run group.

Examples:
- increase target count from 3 to 6
- add a new victim model over an existing target prefix
- rerun failed cells

### Slice
The selected comparable subset of completed cells used by analysis/reporting.

---

## 4. Identity model

The architecture must separate identity into four layers.

### 4.1 Split identity

**Purpose**: identify the canonical split.

**Should depend on**:
- dataset
- split configuration
- preprocessing choices that materially affect the split

**Should not depend on**:
- target count
- victim set
- execution state

**Decision**: preserve the existing split identity semantics when possible.

---

### 4.2 Target cohort identity

**Purpose**: identify the stable target cohort.

**Should depend on**:
- split identity
- target selection mode
- bucket/category (e.g. popular, unpopular)
- target selection seed
- selection policy version
- explicit target list if the mode is explicit

**Must not depend on**:
- current requested target count
- reuse/resume flags
- execution batch size

#### Sampled cohort semantics

For sampled cohorts, the cohort identity defines a **stable ordered pool**.

The requested target count is interpreted as:

> the desired prefix length of that stable ordered pool

So `count=3`, `count=6`, and `count=10` all refer to the same cohort, with different requested prefixes.

#### Explicit cohort semantics

For explicit target lists, the explicit list may define the cohort identity directly.

If appendable explicit cohorts are later supported, the append contract must be formalized separately.

---

### 4.3 Run group identity

**Purpose**: identify the durable experiment container.

**Should depend on**:
- run type
- split identity
- target cohort identity
- attack-generation identity
- attack-replacement identity
- evaluation metric schema

**Must not depend on**:
- current target count
- enabled victim set
- current execution batch contents
- temporary resume state

#### Interpretation

A run group means:

> “All results accumulated for this attack/evaluation setup over this target cohort.”

It is the parent container for all completed target/victim cells.

---

### 4.4 Cell identity

**Purpose**: identify one result cell.

A cell is defined by:

- `target_item`
- `victim_name`

Victim-specific caching may also use `victim_prediction_key`, but that key must not redefine the run group.

---

## 5. Artifact model

The new architecture introduces several first-class artifacts.

### 5.1 `target_registry.json`

**Location**: target cohort shared directory.

**Purpose**:
- store the stable ordered target cohort
- store the currently materialized prefix length
- document how the cohort was derived

**Expected fields**:
- `target_cohort_key`
- `split_key`
- `selection_policy_version`
- `mode`
- `bucket`
- `seed`
- `explicit_list` (nullable)
- `candidate_pool_hash`
- `candidate_pool_size`
- `ordered_targets`
- `current_count`
- timestamps

**Critical invariant**:
- `ordered_targets` must be deterministic and stable for a given sampled cohort.

---

### 5.2 `run_coverage.json`

**Location**: run group root.

**Purpose**:
- record which cells exist and their completion state
- support append planning
- support slice-aware analysis

**Expected fields**:
- `run_group_key`
- `target_cohort_key`
- `targets_order`
- `victims`
- `cells`
- timestamps

**`cells` structure**:
For each target item, store each victim cell status and artifact paths.

**Minimum useful statuses**:
- `pending`
- `completed`
- `failed`

---

### 5.3 `execution_log.json`

**Location**: run group root.

**Purpose**:
- record each execution batch appended into the run group
- support debugging and auditability
- distinguish target-append from victim-append activity

**Expected fields per execution**:
- `execution_id`
- timestamp
- mode
- requested target count
- executed targets
- victims
- status
- optional failure metadata

---

### 5.4 `summary_current.json`

**Location**: run group root.

**Purpose**:
- provide a current human-readable and tool-readable snapshot
- summarize the cumulative state of the run group

**Important rule**:
This file is a **snapshot**, not the only source of truth.

Coverage and registry artifacts remain authoritative for append and slice logic.

---

### 5.5 `slice_manifest.json`

**Location**: analysis output directory.

**Purpose**:
- record how one analysis slice was selected
- make comparisons reproducible and auditable

**Expected fields**:
- source run group
- slice policy
- requested victims
- selected targets
- selected target count
- excluded incomplete cells

---

## 6. Directory structure

### Shared artifacts

Recommended shape:

- `shared/canonical/<split_key>/...`
- `shared/target_cohorts/<target_cohort_key>/...`
- `shared/attack/<shared_attack_artifact_key>/...`
- `shared/victim_predictions/<victim_name>/<victim_prediction_key>/targets/<target_id>/...`

### Run group artifacts

Recommended shape:

- `runs/<dataset>/<experiment>/<run_group_key>/`
  - `resolved_config.json`
  - `key_payloads.json`
  - `artifact_manifest.json`
  - `run_coverage.json`
  - `execution_log.json`
  - `summary_current.json`
  - `targets/<target_id>/victims/<victim_name>/...`

---

## 7. Target append semantics

### 7.1 Sampled cohorts

For sampled target selection:

1. compute a candidate pool
2. sort the candidate pool deterministically
3. apply deterministic seeded ordering/permutation
4. store the full ordered target list in `target_registry.json`

Then:
- requested count = 3 means use the first 3 targets
- requested count = 6 means use the first 6 targets
- requested count = 10 means use the first 10 targets

The run group remains the same across these requests.

### 7.2 Append behavior

When the requested prefix grows:

1. load `target_registry.json`
2. ensure the registry is materialized to at least the requested count
3. compare the requested prefix against `run_coverage.json`
4. execute only missing target/victim cells

### 7.3 Explicit cohorts

For explicit target lists, the initial version may treat the list as a fixed cohort.

If appendable explicit cohorts are later added, the allowed update rule should be explicit rather than implicit.

---

## 8. Victim append semantics

### 8.1 Victims do not define the run group

Adding a new victim must not create a new run group.

Instead:
- the run group stays the same
- new victim cells are added to coverage for already-selected targets

### 8.2 Victim append behavior

When a new victim is enabled:

1. load the run group state
2. load the requested target prefix from the registry
3. find all targets in the requested prefix missing this victim
4. execute only those missing cells

### 8.3 Shared victim caches

Victim-specific caches may still use `victim_prediction_key`, but those are cache-level identities, not run-group identities.

---

## 9. Orchestrator semantics

The orchestrator must no longer assume that one invocation corresponds to a brand-new complete run.

### Required behavior

1. load or initialize run group state
2. load or initialize target registry
3. build an execution plan from requested prefix + enabled victims + coverage
4. execute cell-level jobs
5. update coverage incrementally after each completed cell
6. rebuild/update the current summary snapshot
7. append an execution record to `execution_log.json`

### Required property

The system must support safe resume after partial completion.

---

## 10. Analysis and rendering model

### 10.1 Analysis is responsible for slice resolution

Coverage-aware slice selection must happen **before** final rendering.

The analysis stage must read:
- `summary_current.json`
- `run_coverage.json`
- `target_registry.json`

and resolve a comparable slice.

### 10.2 Renderer is a presentation layer

The renderer should:
- consume already-selected tables/bundles
- display slice metadata
- not infer completeness or fairness by itself

### 10.3 Comparison is only safe on compatible slices

When comparing multiple runs, the comparison stage must validate slice compatibility.

---

## 11. Slice policies

The default architecture assumes at least these slice policies:

### `largest_complete_prefix`
Use the largest ordered target prefix for which all requested victims are complete.

This is the preferred default for the appendable target workflow.

### `intersection_complete`
Use the set intersection of targets complete for all requested victims.

### `all_available`
Use all available rows regardless of completeness.

This is for debugging only and is not recommended for formal comparisons.

---

## 12. Legacy migration principles

The new architecture should not depend on preserving the old run identity semantics.

Instead, migration should:

1. preserve reusable lower-level caches where valid
2. import legacy completed cells into the new run group representation
3. reconstruct coverage, registry, and execution metadata in the new schema

The old summary should not remain the main long-term source of truth.

---

## 13. Compatibility strategy

### Keep stable if semantics are unchanged

Where possible, preserve compatibility for:
- split-level canonical artifacts
- shared attack-generation artifacts

### Rebuild at higher levels

Higher-level run organization should move to the new architecture even if migration is needed.

This means:
- target-selection-level identity may change into target-cohort identity
- run-level identity may change into run-group identity
- old summaries may need import rather than direct reuse

---

## 14. Invariants

The final system should satisfy all of the following:

1. Increasing target count within the same sampled cohort does not create a new run group.
2. Adding a new victim does not create a new run group.
3. The target registry exposes a stable deterministic order.
4. Coverage explicitly records completion at the cell level.
5. Analysis can reconstruct a fair slice without guessing.
6. Comparison rejects incompatible slices by default.
7. Rendering consumes already-resolved slices and displays their metadata.
8. Legacy results can be imported into the new state model.

---

## 15. Suggested implementation hotspots

Expected primary change points:

- `attack/common/paths.py`
- `attack/common/artifact_io.py`
- `attack/pipeline/core/pipeline_utils.py`
- `attack/pipeline/core/orchestrator.py`
- `attack/pipeline/core/victim_execution.py`
- `analysis/pipeline/long_csv_generator.py`

Expected secondary change points:

- `attack/pipeline/core/evaluator.py`
- `analysis/pipeline/compare_runs.py`
- `analysis/pipeline/view_table_builder.py`
- `analysis/pipeline/report_table_renderer.py`

---

## 16. Final architectural summary

The repository is being redesigned from:

> one execution = one fixed run defined by one target selection + one victim set

into:

> one run group = one durable experiment container over one target cohort, gradually filled by completed `(target_item, victim_name)` cells across multiple executions.

Analysis then derives a fair comparable slice from explicit coverage metadata, and rendering presents those slice-aware results.
