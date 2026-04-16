# Multi-Victim Refactor Plan

## Goal

Refactor the current SBR robustness experiment framework so it can support:

1. one unified benchmark split protocol,
2. reusable shared attack artifacts,
3. one or multiple target items in one run,
4. one or multiple victim models in one run,
5. per-model exporters and runners,
6. centralized evaluation for target-oriented metrics,
7. future extension to additional victim models such as TRON.

At the current stage:

- the **poison model remains SRGNN**,
- the main new goal is to support **different victim models** cleanly,
- victim checkpoints are **not required** and should **not be saved**.

This refactor is intended to move the repository directly toward the new architecture. It is **not** a goal to preserve the old SRGNN-only pipeline as a compatibility layer.

---

## Confirmed Design Decisions

### 1. Unified benchmark protocol
Use one shared split policy for the framework.

- `original train -> train_sub + valid`
- `original test -> test`
- poisoning applies **only to `train_sub`**
- `valid` remains clean
- `test` remains clean

This unified split is the main protocol for all victim models.

### 2. Data source policy
Keep using the original dataset source under the existing `datasets/` location.

The framework should:

- read from the same original dataset source,
- perform one shared preprocessing and split step,
- freeze and save the resulting canonical split,
- let each victim model convert that canonical split into its own required format.

### 3. Poison model policy
At this stage, keep the poison model fixed to **SRGNN**.

This only refers to the **attack artifact generation stage**, such as:

- fake session generation support,
- position optimization support,
- poison-model-related shared artifacts.

Do **not** add multi-poison-model support yet.

### 4. Victim model policy
The framework must support running **one or multiple victim models** in a single experiment.

- if one victim is enabled, only one is run,
- if multiple victims are enabled, they are run sequentially on the same poisoned data.

Use a list-based configuration design.

### 5. Target item policy
Target selection must support both:

- `explicit_list`
- `sampled`

`sampled` mode is necessary because target items may need to be drawn from buckets such as:

- popular,
- unpopular,
- all.

The sampled result must be saved to `target_info.json` and be reusable.

### 6. Seed policy
At this stage, only split the previous global seed into two explicit seeds:

- `fake_session_seed`
- `target_selection_seed`

The purpose is:

- fake sessions can remain fixed,
- targets can change independently,
- target changes should not force fake sessions to change.

### 7. Evaluation policy
The centralized evaluator should support:

- `targeted_precision@K`
- `targeted_mrr@K`

These metrics should be reported for each victim model against the target item.

### 8. Checkpoint policy
- **Save shared attack artifacts**.
- **Do not save victim model checkpoints**.

Required shared artifacts include:

- `poison_model.pt`
- `fake_sessions.pkl`
- `target_info.json`

Victim-side outputs should focus on:

- metrics,
- logs,
- prediction outputs / top-k outputs.

---

## High-Level Target Architecture

```text
original datasets/
-> shared preprocessing
-> canonical split (train_sub / valid / test)
-> shared attack artifacts
-> target loop
   -> build target-specific poisoned train_sub
   -> victim loop
      -> per-model exporter
      -> per-model runner
      -> centralized evaluator
      -> outputs
```

---

## Planned Module Responsibilities

### A. Shared data layer
Responsible for:

- reading original datasets,
- performing unified filtering,
- performing unified time-based split,
- producing canonical `train_sub / valid / test`,
- freezing the canonical split for reuse.

This layer must be model-agnostic.

### B. Attack-side shared artifacts
Responsible for:

- poison model checkpoint reuse,
- fake session reuse,
- target selection result reuse.

These artifacts must not depend on the victim model.

### C. Target abstraction
Responsible for:

- explicit or sampled target selection,
- target loop orchestration,
- reusing shared fake sessions while varying targets,
- saving target metadata to `target_info.json`.

### D. Victim abstraction
Responsible for:

- hiding victim-specific training and evaluation details,
- providing a common runner interface,
- registering supported victim models,
- allowing one or many victims in one run.

### E. Exporter layer
Responsible for converting canonical split data into model-specific input formats.

Examples:

- SRGNN exporter
- MiaSRec exporter
- TRON exporter

### F. Centralized evaluator
Responsible for:

- computing framework-level metrics,
- aligning metric definitions across victim models,
- avoiding dependency on model-native printed metrics only.

### G. Output layer
Responsible for:

- separating shared attack artifacts from run outputs,
- separating outputs by target and victim,
- storing logs, metrics, and prediction outputs.

---

## File/Folder Refactor Direction

Refactor toward the following conceptual structure:

```text
attack/
├── common/
├── configs/
├── data/
│   ├── canonical_dataset.py
│   ├── unified_split.py
│   ├── poisoned_dataset_builder.py
│   └── exporters/
│       ├── base_exporter.py
│       ├── srgnn_exporter.py
│       ├── miasrec_exporter.py
│       └── tron_exporter.py
├── generation/
├── insertion/
├── models/
│   ├── poison/
│   │   └── srgnn_poison_runner.py
│   └── victim/
│       ├── base_runner.py
│       ├── registry.py
│       ├── srgnn_runner.py
│       ├── miasrec_runner.py
│       └── tron_runner.py
└── pipeline/
```

This is a design target, not a requirement to complete in one batch.

---

## Shared Split Requirements

The shared split must be defined once and reused by all victim models.

### Requirements

1. Use one common filtering pipeline.
2. Use one common time-based split rule.
3. Derive `valid` from the original training portion.
4. Keep `test` fixed and clean.
5. Do not let each model silently redefine its own split.

### Important note
Do **not** define the benchmark by fixed percentage alone.
The benchmark should be defined by:

- filtering rule,
- time split rule,
- valid split rule,
- then frozen and saved.

---

## MiaSRec Integration Notes

MiaSRec has already been independently verified to run.

### Confirmed input contract
MiaSRec benchmark mode expects three split files:

- `diginetica.train.inter`
- `diginetica.valid.inter`
- `diginetica.test.inter`

Core columns:

- `session_id`
- `item_id_list`
- `item_id`

Where:

- `item_id_list` is the prefix sequence,
- `item_id` is the next-item label.

### Integration boundary
MiaSRec must remain an external repository under `third_party/miasrec`.

The framework should integrate MiaSRec by:

- exporting canonical split data into benchmark `.inter` files,
- calling MiaSRec via subprocess on Windows,
- later adding a minimal evaluation-path patch only if needed for exporting per-sample top-k predictions.

### Important restriction
Do not merge MiaSRec code into the internal attack framework.

---

## TRON Integration Notes

TRON is planned as a later stage.

Before implementation:

- inspect TRON input contract,
- inspect TRON output contract,
- identify whether TRON needs exporter adaptation,
- then add `tron_exporter.py` and `tron_runner.py`.

TRON should follow the same framework rules:

- consume canonical split semantics,
- run as a victim model,
- report into centralized evaluation.

---

## Output and Saving Policy

### Shared attack artifacts
Shared artifacts should be reusable and saved separately from run outputs.

#### A. Fake-session / poison-model shared artifacts
These should be keyed primarily by:

- dataset,
- split protocol,
- fake session seed,
- fake-session-generation-relevant settings,
- poison-model-relevant settings.

These artifacts include at least:

- `poison_model.pt`
- `fake_sessions.pkl`

Target selection changes should **not** force these artifacts to move or regenerate.

Important separation:

Fake-session / poison-model artifacts must not depend on target selection.

Changing:
- target_selection_seed
- target bucket
- target count
- explicit target list

must NOT invalidate:
- poison_model.pt
- fake_sessions.pkl

Only target_info.json should depend on target selection.

#### B. Target-selection artifacts
These should be keyed primarily by:

- dataset,
- split protocol,
- target selection seed,
- target mode,
- target bucket,
- target count.

These artifacts include at least:

- `target_info.json`

### Run outputs
Run outputs should be stored separately from shared artifacts.

They should be organized primarily by:

- dataset,
- experiment name,
- target id,
- victim model.

The implementation must also prevent accidental overwrite across different seed/config runs.

Victim outputs should store only:

- logs,
- metrics,
- prediction outputs / top-k outputs,
- temporary exported datasets if needed.

### Recommended conceptual layout

```text
outputs/
├── shared/
│   └── <dataset>/...
└── runs/
    └── <dataset>/<experiment>/
        └── targets/
            └── <target_id>/
                └── victims/
                    ├── srgnn/
                    ├── miasrec/
                    └── tron/
```

This is a conceptual target and may be implemented incrementally.

---

## Config Direction

The config should remain primarily **attack-centric**, but must be extended with explicit blocks for:

- data / split protocol,
- seeds,
- targets,
- victims,
- evaluation,
- artifacts.

### Intended shape

```yaml
experiment:
  name: ...

data:
  dataset_name: diginetica
  split_protocol: unified
  poison_train_only: true

seeds:
  fake_session_seed: 42
  target_selection_seed: 123

attack:
  method: ...
  poison_model:
    name: srgnn

targets:
  mode: sampled   # or explicit_list
  explicit_list: []
  bucket: popular
  count: 5
  reuse_saved_targets: true

victims:
  enabled: [srgnn, miasrec]

evaluation:
  topk: 20
  targeted_metrics:
    - precision
    - mrr
  ground_truth_metrics:
    - precision
    - mrr

artifacts:
  root: outputs
  shared_dir: shared
  runs_dir: runs
```

This is a design target and does not require final naming to match exactly.

---

## Implementation Strategy

Implementation must be done in **small batches**.
Do **not** ask Codex to perform the full refactor in one turn.

---

## Batch Plan

### Batch 1 — Config and path foundation
Focus only on the new config schema and the saving/path foundation.

Scope:

1. refactor config loading to the new schema,
2. introduce explicit `data`, `seeds`, `targets`, `victims`, `evaluation`, `artifacts` blocks,
3. refactor path helpers for the new output architecture,
4. do not implement canonical split yet,
5. do not migrate SRGNN runtime yet.

Success condition:

- new config/path foundation exists and matches the new architecture.

### Batch 2 — Legacy pipeline alignment with new schema

This batch aligns the existing SRGNN-centric pipeline with the new config schema and path helpers.

Scope:

1. update pipeline to read the new config schema
2. update run scripts to use dataset_paths and new seed fields
3. align SRGNN runner with dataset_paths
4. ensure shared artifact paths follow the new structure
5. do NOT implement canonical split yet

Important:

This batch does not introduce the new canonical dataset layer.
It only aligns the existing pipeline with the new schema to prepare for later migration.

### Batch 3 — Canonical dataset and unified split
Focus only on the shared dataset layer.

Scope:

1. implement shared preprocessing / canonical split,
2. introduce canonical `train_sub / valid / test`,
3. apply one common filtering pipeline before splitting,
4. apply one common time-based split rule,
5. derive `valid` from the original training portion,
6. freeze and save the canonical split for reuse,
7. make this canonical split the future single source of truth for all exporters and victim models,
8. do not implement target loop yet,
9. do not implement victim loop yet,
10. do not integrate exporters or runners yet.

Important:

This batch introduces the real unified benchmark dataset layer.

It should replace the current temporary situation where the new config/path schema exists but the canonical split has not yet been truly constructed.

The result of this batch should be a shared, frozen, model-agnostic dataset representation that all later exporters will consume.

Success condition:

- canonical `train_sub / valid / test` exists,
- the unified split is frozen and reusable,
- the framework now has a true shared dataset layer,
- later batches can build exporters and new pipeline execution on top of this layer.

### Batch 4 — Poison/victim abstraction
Focus on refactoring the model layer.

Scope:

1. split model responsibilities into poison vs victim,
2. add victim base runner and registry,
3. create SRGNN poison-side and victim-side runner structure,
4. do not integrate exporters or pipeline execution yet.

Success condition:

- model-layer structure supports future multi-victim design.

### Batch 5 — Exporter scaffolding
Focus on per-model exporter structure.

Scope:

1. add exporter abstraction,
2. add SRGNN exporter,
3. add MiaSRec/TRON exporter scaffolds,
4. refactor old serializer responsibility into exporter-based design,
5. do not migrate pipeline execution yet.

Success condition:

- exporter layer exists and is aligned with canonical dataset input.

### Batch 6 — SRGNN migration into the new pipeline
Focus only on making SRGNN run end-to-end through the new architecture.

Scope:

1. connect canonical split,
2. connect shared attack artifacts,
3. connect target selection,
4. connect SRGNN exporter and SRGNN victim runner,
5. run single-target, single-victim SRGNN through the new pipeline.

Success condition:

- SRGNN runs end-to-end under the new architecture.

### Batch 7 — MiaSRec integration
Focus only on integrating MiaSRec cleanly.

Scope:

1. add concrete `miasrec_exporter.py`,
2. add concrete `miasrec_runner.py`,
3. connect MiaSRec to canonical split,
4. run MiaSRec via subprocess from Windows,
5. do not yet redesign full multi-victim automation if not necessary.

Success condition:

- MiaSRec can run from the unified framework on the canonical split.

### Batch 8 — Central evaluator and target/victim loops
Focus on orchestration and unified evaluation.

Scope:

1. add centralized targeted metrics,
2. implement target loop,
3. implement victim loop,
4. separate outputs by target and victim,
5. add minimal support for prediction output collection.

Success condition:

- one experiment can run multiple targets and multiple victim models and collect centralized metrics.

### Batch 9 — TRON integration
Focus only on adding TRON after the framework is already stable.

Scope:

1. inspect TRON input/output contract,
2. add concrete `tron_exporter.py`,
3. add concrete `tron_runner.py`,
4. plug TRON into victim loop.

Success condition:

- TRON runs as another victim model under the same benchmark protocol.

---

## Things Explicitly Out of Scope for Now

Do **not** include these unless specifically requested in a later batch:

1. multiple poison models,
2. saving victim checkpoints,
3. full metric expansion beyond targeted Precision/MRR,
4. large TRON integration before SRGNN and MiaSRec are stable,
5. one-shot full-repo rewrite.

---

## Recommended Codex Workflow

When using Codex, always instruct it to:

1. first read this document,
2. implement only the requested batch,
3. keep changes scoped to that batch,
4. avoid touching later batches unless strictly necessary,
5. explain how each modified file maps back to this plan.

Recommended instruction pattern:

```text
Please first read docs/multi_victim_refactor_plan.md.
Only implement Batch 1 in this turn.
Do not touch later batches unless strictly necessary.
Keep changes aligned with the plan.
```

---

## Final Notes

The key principle of this refactor is:

- one unified benchmark,
- one shared data split,
- one shared attack-artifact layer,
- target abstraction,
- victim abstraction,
- model-specific exporters and runners,
- centralized evaluation.

The framework should become easier to maintain, easier to extend, and easier to use for future victim-model comparisons.
