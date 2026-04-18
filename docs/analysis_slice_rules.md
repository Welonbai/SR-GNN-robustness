# Analysis Slice Rules

This document defines how analysis must select comparable subsets of results from appendable experiment state.

The key principle is:

> comparability must be decided explicitly from coverage artifacts before final rendering.

The renderer should not guess comparability.

---

## 1. Inputs for slice resolution

Slice resolution must use the following run-time artifacts:

- `summary_current.json`
- `run_coverage.json`
- `target_registry.json`

It may also use analysis inputs such as:
- requested victim list
- requested target count
- chosen slice policy

---

## 2. Core entities

### Target registry order
The target registry provides the stable ordered list of targets for a cohort.

This order is authoritative for prefix-based slice policies.

### Coverage matrix
Coverage is the set of completion states over `(target_item, victim_name)` cells.

### Requested victims
The victim set requested by the analysis command.

Only these victims matter when resolving the slice.

### Requested target count
Optional cap on how many targets from the target registry may participate in the slice.

If omitted, analysis may consider the full currently materialized target prefix.

---

## 3. Required slice policies

Analysis should support at least the following policies.

### 3.1 `largest_complete_prefix`

#### Definition
Given:
- an ordered target list from the target registry
- a requested victim set
- an optional requested target count cap

Select the largest prefix of the ordered target list such that, for every target in the prefix, all requested victims have `completed` status.

#### Use case
This is the preferred default for appendable target-cohort experiments.

It matches the workflow where target items are gradually appended in registry order.

#### Example
Target registry order:
- `[t1, t2, t3, t4, t5, t6]`

Requested victims:
- `A, B, C`

Coverage:
- `A` completed for `t1..t6`
- `B` completed for `t1..t6`
- `C` completed for `t1..t4`

Selected slice under `largest_complete_prefix`:
- `[t1, t2, t3, t4]`

#### Notes
- If a later target is complete but an earlier target is incomplete, the prefix stops before the incomplete target.
- This policy preserves the semantics of ordered append growth.

---

### 3.2 `intersection_complete`

#### Definition
Given the requested victim set, select all targets for which every requested victim has `completed` status.

This does not require the selected targets to form a prefix.

#### Use case
Useful when completeness is sparse or when prefix semantics are not desired.

#### Example
Target registry order:
- `[t1, t2, t3, t4, t5, t6]`

Coverage:
- `A` completed for `t1..t6`
- `B` completed for `t1, t2, t4, t5`
- `C` completed for `t1, t2, t4`

Selected slice under `intersection_complete`:
- `[t1, t2, t4]`

#### Notes
This policy may produce non-prefix slices, so rendered outputs should clearly identify the policy.

---

### 3.3 `all_available`

#### Definition
Return all available completed rows without enforcing victim-complete comparability across the requested set.

#### Use case
Debugging only.

#### Warning
This policy is not appropriate for formal comparisons because different victims may be averaged over different target sets.

---

## 4. Required status interpretation

At minimum, slice resolution should interpret coverage statuses like this:

- `completed`: eligible for inclusion
- `pending`: not eligible
- `failed`: not eligible

Additional statuses may exist, but only `completed` should count as complete for strict slice selection.

---

## 5. Requested target count behavior

If analysis specifies a target count cap, slice resolution must not exceed that cap.

### Example
Target registry prefix currently materialized to 10 targets.

Analysis requests:
- victims = `A, B`
- policy = `largest_complete_prefix`
- target_count = 6

Then the slice must be resolved using only the first 6 ordered targets.

This allows controlled reporting at `N=3`, `N=6`, `N=10`, etc.

---

## 6. Empty-slice behavior

If no targets satisfy the slice policy for the requested victims:

- analysis should fail clearly, or
- produce an empty slice artifact only if an explicit debug flag allows it.

Do not silently render misleading empty comparisons.

---

## 7. Slice manifest requirements

Every slice-aware analysis output should produce a `slice_manifest.json`.

### Minimum recommended fields
- `source_run_group_key`
- `target_cohort_key`
- `slice_policy`
- `requested_victims`
- `requested_target_count` (nullable)
- `selected_targets`
- `selected_target_count`
- `excluded_targets`
- `excluded_incomplete_cells`
- timestamps

### Purpose
This artifact makes analysis reproducible and comparison-safe.

---

## 8. Long-table generation rules

`long_csv_generator.py` should apply the selected slice **before** writing the long table.

This means:
- only rows belonging to the selected slice are emitted
- incomplete rows outside the slice are excluded unless debug mode explicitly requests otherwise

The long table should not leave fairness decisions to later stages.

---

## 9. Comparison rules

Comparison should operate on slice-aware outputs, not on raw cumulative summaries.

### Default comparison rule
Two analysis outputs are comparable only if their slice manifests are compatible.

### Minimum compatibility checks
- same `slice_policy`
- same `requested_victims` (or a documented compatible equivalent)
- same `selected_targets`, or at minimum a clearly accepted same-slice guarantee
- same selected target count for formal average-based comparisons

### Recommended default
Use strict compatibility by default and require an explicit override for relaxed comparison.

---

## 10. Renderer expectations

The renderer should receive already-sliced data and metadata.

It should not:
- infer missing victims
- recompute intersections
- decide fairness policies

It may display:
- slice policy
- requested victims
- selected target count
- notes about completeness basis

---

## 11. Recommended CLI/analysis interface

A slice-aware analysis command should support inputs such as:

- summary path
- coverage path
- target registry path
- requested victims
- slice policy
- optional target count

It should produce:
- long table
- inventory/manifest
- `slice_manifest.json`

---

## 12. Recommended default policy

The repository should default to:

- `slice_policy = largest_complete_prefix`

for formal appendable target-cohort reporting.

This default best matches the intended workflow of gradually extending the target cohort over time.

---

## 13. Summary

The slice system exists to guarantee that:

- target append does not silently break comparability
- victim append does not silently break comparability
- reports and tables are based on explicit, auditable, coverage-aware target selection

The final rule is simple:

> analysis must decide the slice; rendering must only present it.
