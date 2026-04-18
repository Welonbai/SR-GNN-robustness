# Legacy Migration Plan

This document defines how previously completed results should be migrated into the new appendable experiment architecture.

The guiding principle is:

> reuse valid existing artifacts, but migrate old run results into the new state model instead of preserving the old run-identity semantics.

---

## 1. Why migration is necessary

The old architecture encoded append-sensitive information into run identity, including concepts like:
- the current target selection result
- the currently enabled victim set

The new architecture separates:
- target cohort identity
- run group identity
- cell-level completion state

Because of this change, legacy runs cannot simply be treated as if they were already stored under the new state model.

A formal migration/import path is required.

---

## 2. Migration goals

A valid migration must achieve all of the following:

1. preserve reusable lower-level artifacts when their semantics still match
2. reconstruct the new target cohort/run group state model
3. import completed legacy cells into `run_coverage.json`
4. rebuild `summary_current.json` from the migrated state
5. make migrated runs usable by slice-aware analysis

---

## 3. What should be preserved directly

Where semantics are unchanged, the migration should preserve and reuse:

### 3.1 Split-level canonical artifacts
Examples:
- canonical split outputs
- split metadata
- any artifacts keyed only by split identity

### 3.2 Shared attack-generation artifacts
Examples:
- poison model outputs
- fake session generation outputs
- attack-generation histories

These should remain reusable if the new architecture preserves the same underlying split and attack-generation semantics.

---

## 4. What should be imported rather than directly reused as-is

The following legacy artifacts should generally be imported into the new state model rather than treated as long-term authoritative structure:

### 4.1 Legacy target-selection artifacts
The old target-selection identity may have been coupled to:
- requested count
- reuse flags
- batch-specific settings

These should become part of the new target cohort representation only after normalization.

### 4.2 Legacy run summaries
Legacy summary files may reflect a single execution worldview rather than a durable appendable run group.

They are useful as source material, but they should not remain the final state authority.

### 4.3 Legacy run-directory identity
Old run keys may depend on victim sets or target-selection batches.

These must not remain the final architecture semantics.

---

## 5. Migration output targets

A migrated run group should contain at minimum:

- `target_registry.json`
- `run_coverage.json`
- `execution_log.json`
- `summary_current.json`
- migrated or referenced cell artifacts under the new run-group layout

The migrated run group should then behave like a native new-architecture run group.

---

## 6. Legacy source data to read

A migration utility should inspect as many of the following as are available:

- legacy summary files
- legacy local per-target/per-victim `metrics.json`
- legacy local per-target/per-victim `predictions.json`
- legacy train history outputs
- legacy poisoned train references
- legacy artifact manifests
- legacy key payloads / resolved config files
- legacy selected-target lists / target metadata

The utility should not assume every optional file exists.

---

## 7. Migration strategy

### Step 1 — Discover legacy runs

Scan legacy run roots and collect:
- config information
- target items present
- victim names present
- per-cell artifact availability

### Step 2 — Infer new identities

From the legacy configuration and artifact structure, infer:
- split identity
- target cohort identity
- run group identity

If some aspects are ambiguous, the migration should fail explicitly rather than silently guessing.

### Step 3 — Construct or recover target registry

Build `target_registry.json` using one of these strategies:

#### Preferred strategy
Recover the actual target order from legacy target-selection artifacts if the order is meaningful and deterministic.

#### Fallback strategy
If only a completed target set is known and append order cannot be recovered reliably, record a stable deterministic order for the migrated cohort and mark the registry as migrated.

The migration metadata should make this provenance explicit.

### Step 4 — Reconstruct coverage

Create `run_coverage.json` by iterating through every discovered legacy `(target_item, victim_name)` result cell.

For each cell, record:
- completion status
- metrics path
- predictions path
- optional train history path
- optional poisoned train path
- migration source metadata

### Step 5 — Create execution log entries

Add at least one imported execution record into `execution_log.json`.

This record should identify:
- that the state originated from legacy import
- which targets and victims were imported
- source legacy run identifiers if available

### Step 6 — Rebuild current summary

Generate `summary_current.json` from the migrated coverage and cell artifacts.

This ensures the summary is a fresh snapshot consistent with the new architecture.

---

## 8. Artifact placement strategies

Two migration styles are acceptable in principle.

### 8.1 Reference-style import

The new coverage/state files point to legacy artifact paths without moving or duplicating the files.

#### Advantages
- simpler initial migration
- minimal storage duplication

#### Disadvantages
- the new run group depends on legacy directory layout
- long-term maintenance is messier
- path durability depends on preserving old structures

### 8.2 Copy/symlink-style import

Cell artifacts are copied or linked into the new run-group structure.

#### Advantages
- the migrated run group becomes self-contained
- the new architecture is cleaner and easier to maintain
- future append operations interact with one consistent directory layout

#### Disadvantages
- initial migration is more involved
- storage costs may increase if hardlinks/symlinks are not available

### Recommendation
Prefer copy/symlink-style import for long-term maintainability.

---

## 9. Required migration metadata

Migrated artifacts should record their origin where practical.

Useful metadata fields include:
- `imported_from_legacy: true`
- `legacy_run_key`
- `legacy_summary_path`
- `legacy_cell_path`
- `imported_at`
- `migration_version`

This can be stored in:
- coverage cell metadata
- execution logs
- optional cell manifests

---

## 10. Validation after migration

A migrated run group should pass the following checks.

### 10.1 Registry validation
- target cohort key exists
- ordered target list exists
- current target count is coherent

### 10.2 Coverage validation
- every imported completed cell has a valid target and victim
- every completed cell points to the expected artifacts
- no imported cell is silently dropped

### 10.3 Summary validation
- summary snapshot reflects imported cells
- target and victim listings match coverage

### 10.4 Analysis validation
- slice-aware analysis can generate a long table from the migrated run group
- comparison readiness can be checked from the generated slice manifest

---

## 11. Handling ambiguity

Migration should fail loudly when it cannot safely infer:
- the correct split identity
- the target cohort membership/order
- the intended run-group mapping

Do not silently merge ambiguous legacy runs into one new run group.

If needed, provide manual override inputs to the migration tool for exceptional cases.

---

## 12. What migration should not do

The migration utility should **not**:
- preserve old run-group semantics as the new long-term architecture
- scatter legacy-key fallback logic throughout the main execution path
- silently reinterpret incompatible old runs as one appendable run group
- bypass coverage reconstruction and rely only on legacy summary files

---

## 13. Suggested migration utility behavior

A migration tool such as `attack/tools/migrate_legacy_runs.py` should support:

- selecting one or more legacy run roots
- preview/dry-run mode
- explicit import destination
- optional copy/symlink/reference mode
- validation-only mode
- explicit failure on ambiguous mapping

A dry run should show:
- inferred split identity
- inferred target cohort identity
- inferred run-group identity
- discovered targets
- discovered victims
- number of completed cells to import

---

## 14. Recommended workflow

1. finish the new architecture implementation first
2. validate native appendable runs
3. run migration on selected legacy runs
4. validate migrated run groups with slice-aware analysis
5. continue future appends only in the new architecture

---

## 15. Summary

Legacy migration is not about pretending old runs were already appendable run groups.

It is about:
- preserving valid lower-level artifacts,
- reconstructing the new explicit state model,
- importing old completed cells into that model,
- and making old results usable within the new appendable workflow.

The end state should be:

> migrated results behave like first-class citizens of the new architecture.
