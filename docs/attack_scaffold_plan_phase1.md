# Attack Scaffold Plan for SBR Robustness (Phase 1)

## 1. Purpose

This document defines the implementation plan for the phase-1 attack scaffold in this repository.

The research topic is **Session-Based Recommendation (SBR) robustness**, with **SR-GNN** as the initial clean backbone. The immediate goal is **not** to build a full black-box or generalized attack framework. Instead, the goal is to build a **minimal, extensible, and non-disposable attack scaffold** that can support:

1. a reproducible clean SR-GNN run,
2. a DP-SBR-baseline-compatible attack pipeline,
3. a later replacement of the insertion policy with my own method.

The implementation priority is therefore:

- preserve a clean SR-GNN backbone,
- build the attack framework outside the backbone,
- maximize code reuse between the DP-SBR baseline and my later method,
- avoid introducing abstractions that are unnecessary for phase 1.

---

## 2. Research Positioning

### 2.1 Current phase objective

Phase 1 only aims to answer the following question:

> Under the same fake-session-generation backbone, if we later replace only the target insertion step, can insertion policy alone produce a stable improvement?

Before testing that later idea, phase 1 must first provide a trustworthy baseline pipeline.

### 2.2 What phase 1 includes

Phase 1 includes only:

- a clean SR-GNN run,
- a DP-SBR-baseline-compatible attack run,
- the shared scaffold pieces needed by both.

### 2.3 What phase 1 does not include

Phase 1 must **not** implement:

- best-position replacement,
- local context optimization,
- beam search,
- bilevel optimization,
- Gumbel-Softmax relaxation,
- stealth objective,
- black-box validation,
- generalized multi-model evaluation,
- future attack variants beyond the DP-SBR-compatible baseline.

These belong to later phases.

---

## 3. High-Level Design Principles

### 3.1 Clean backbone vs. external attack layer

The repository must be divided conceptually into two layers:

#### A. Clean SR-GNN backbone

Responsible for:

- dataset preprocessing from raw data to clean SR-GNN dataset,
- clean training,
- inference / recommendation,
- baseline evaluation already supported by the original implementation.

#### B. External attack framework

Responsible for:

- attack-level dataset statistics,
- fake-session parameter sampling,
- fake-session generation,
- target insertion policy,
- poisoned dataset construction,
- orchestration of training and evaluation runs.

The attack framework must sit **outside** the SR-GNN core.  
Attack logic should not be mixed into the SR-GNN model implementation unless absolutely necessary for integration.

### 3.2 Shared upper half, swappable lower half

The intended scaffold design is:

- **shared upper half**
  - session statistics,
  - fake-session parameter sampling,
  - poison-model-guided fake-session generation,
  - poisoned dataset assembly,
  - evaluation orchestration.

- **swappable lower half**
  - insertion policy.

This means the DP-SBR baseline and my later method should share as much code as possible, differing only where the method is genuinely different.

### 3.3 Each step must remain reusable later

Any phase-1 code should be written so that it can remain the base for later work.

The phase-1 scaffold should be easy to extend toward:

- best-position replacement,
- local context scoring,
- beam search,
- stealth-aware objectives,
- black-box validation.

But these should **not** be prematurely implemented now.

---

## 4. Repository Role Separation

### 4.1 Existing clean backbone area

The following existing parts belong to the clean SR-GNN backbone:

- `datasets/preprocess.py`
- `pytorch_code/`
- optionally `tensorflow_code/` if kept for reference

`datasets/preprocess.py` is the **clean data entry point** from raw dataset to SR-GNN-ready dataset.

It is **not** part of the attack scaffold.

### 4.2 New attack framework area

A new package should be introduced, tentatively:

- `attack/`

This package should contain the external robustness / attack scaffold.

---

## 5. Phase-1 Target File Structure

A minimal target structure for phase 1 is:

    attack/
    ├── common/
    │   ├── config.py
    │   ├── seed.py
    │   ├── paths.py
    │   └── logging_utils.py
    ├── configs/
    │   └── dp_sbr_diginetica.yaml
    ├── data/
    │   ├── session_stats.py
    │   ├── target_selector.py
    │   ├── poisoned_dataset_builder.py
    │   └── dataset_serializer.py
    ├── models/
    │   └── srg_nn_runner.py
    ├── generation/
    │   ├── fake_session_parameter_sampler.py
    │   ├── score_smoothing.py
    │   └── fake_session_generator.py
    ├── insertion/
    │   ├── base_policy.py
    │   └── random_topk_replace.py
    └── pipeline/
        ├── run_clean.py
        ├── run_dp_sbr_baseline.py
        └── evaluator.py

This structure is intentionally minimal.  
No later-phase modules should be added yet.

---

## 6. Module Responsibilities

### 6.1 `attack/common/`

Shared infrastructure only.

#### `config.py`

Responsible for:

- loading experiment config from YAML,
- validating only phase-1-required fields,
- exposing a structured config object to the pipeline.

#### `seed.py`

Responsible for:

- controlling random seed for reproducibility,
- synchronizing Python / NumPy / PyTorch randomness.

#### `paths.py`

Responsible for:

- building experiment output directories,
- defining artifact paths for checkpoints, fake sessions, poisoned datasets, logs, and metrics.

#### `logging_utils.py`

Responsible for:

- consistent console/file logging format across runs.

### 6.2 `attack/data/`

Attack-level data logic.

#### `session_stats.py`

Responsible for computing statistics from the clean training set, including:

- first-item distribution,
- session-length distribution,
- item frequency.

#### `target_selector.py`

Responsible for selecting target items for experiments, including:

- explicitly specified target item,
- popular target pool,
- unpopular target pool.

#### `poisoned_dataset_builder.py`

Responsible for constructing the poisoned training set from:

- clean training sessions,
- generated fake sessions.

#### `dataset_serializer.py`

Responsible for converting attack-level session data into the format required by SR-GNN training code.

This is necessary because attack modules will likely manipulate sessions in a simpler Python structure, while SR-GNN expects its own data format.

### 6.3 `attack/models/`

Model execution adapter layer.

#### `srg_nn_runner.py`

Responsible for wrapping the clean SR-GNN backbone so that the attack pipeline can call it as a reusable component.

Expected responsibilities:

- clean training,
- poison-model training,
- score inference for fake-session generation,
- evaluation invocation.

In phase 1, there should be only **one** SR-GNN runner abstraction, not separate victim and poison runners with different implementations. Their roles differ conceptually, but they use the same underlying model family and should share the same runner.

### 6.4 `attack/generation/`

Fake-session generation logic.

#### `fake_session_parameter_sampler.py`

Responsible for sampling:

- fake session initial item,
- fake session length,

based on clean training distribution.

#### `score_smoothing.py`

Responsible for score smoothing used during fake-session generation.

Phase 1 only needs the smoothing used by the DP-SBR-compatible baseline.

#### `fake_session_generator.py`

Responsible for generating fake sessions iteratively using the poison model.

This module should generate the template/filler session first, before target insertion.

### 6.5 `attack/insertion/`

Insertion-policy layer.

#### `base_policy.py`

Defines the common insertion-policy interface.

#### `random_topk_replace.py`

Implements the DP-SBR-compatible baseline insertion policy.

No later insertion variants should be added in phase 1.

### 6.6 `attack/pipeline/`

Top-level orchestration.

#### `run_clean.py`

Runs a clean SR-GNN experiment with no attack.

#### `run_dp_sbr_baseline.py`

Runs the phase-1 attack pipeline with the DP-SBR-compatible baseline insertion policy.

#### `evaluator.py`

Responsible for collecting and standardizing evaluation outputs needed by phase 1.

---

## 7. Phase-1 Implementation Order

Implementation must proceed incrementally.  
Do **not** create the whole scaffold in one step.

Recommended order:

### Step 1. Common infrastructure

Implement:

- `attack/common/config.py`
- `attack/common/seed.py`
- `attack/common/paths.py`

Goal:

- establish reproducible experiment execution,
- define config-driven behavior,
- avoid hard-coded paths and parameters.

### Step 2. SR-GNN runner

Implement:

- `attack/models/srg_nn_runner.py`

Goal:

- expose the clean SR-GNN backbone to the external attack layer without modifying the backbone heavily.

### Step 3. Attack data statistics

Implement:

- `attack/data/session_stats.py`
- `attack/data/target_selector.py`

Goal:

- compute attack-relevant statistics from clean train data.

### Step 4. Fake-session parameter sampling and generation

Implement:

- `attack/generation/fake_session_parameter_sampler.py`
- `attack/generation/score_smoothing.py`
- `attack/generation/fake_session_generator.py`

Goal:

- generate filler/template sessions before insertion.

### Step 5. Baseline insertion policy

Implement:

- `attack/insertion/base_policy.py`
- `attack/insertion/random_topk_replace.py`

Goal:

- complete the DP-SBR-compatible fake session.

### Step 6. Poisoned dataset building

Implement:

- `attack/data/poisoned_dataset_builder.py`
- `attack/data/dataset_serializer.py`

Goal:

- convert clean train + fake sessions into a trainable poisoned dataset.

### Step 7. Pipeline and evaluation

Implement:

- `attack/pipeline/run_clean.py`
- `attack/pipeline/run_dp_sbr_baseline.py`
- `attack/pipeline/evaluator.py`

Goal:

- make the whole phase-1 flow reproducible and comparable.

---

## 8. Required Phase-1 Outputs

After phase 1, the scaffold should support at least:

1. a clean SR-GNN run,
2. a DP-SBR-baseline-compatible attack run,
3. reproducible config-driven execution,
4. saved artifacts for:
   - config snapshot,
   - logs,
   - checkpoints,
   - fake sessions,
   - poisoned dataset,
   - evaluation metrics.

---

## 9. Constraints and Non-Negotiable Rules

The following rules must be respected in all implementation work.

### 9.1 Do not break the clean backbone

- Do not rewrite `pytorch_code/model.py` unless absolutely required for integration.
- Do not mix attack-specific logic into SR-GNN core training code.
- Do not modify `datasets/preprocess.py` for attack-specific purposes.

### 9.2 Do not over-abstract early

- Do not add interfaces for future phases unless phase 1 already needs them.
- Do not add black-box/generalized support now.
- Do not add future insertion policies now.

### 9.3 Keep every task small and verifiable

Implementation should be done as small, reviewable tasks.  
Each task should be independently testable before moving to the next one.

### 9.4 Config-first, not hard-coded

Experiment settings should live in YAML config files.  
The pipeline should load config and pass structured settings to modules.

### 9.5 Reproducibility is mandatory

All randomness involved in fake-session generation and training should be seed-controlled.

---

## 10. Recommended Codex Workflow

Codex should not be asked to implement the entire scaffold in one shot.

Recommended workflow:

1. ask Codex to read this plan and summarize the phase-1 file tree,
2. ask it to implement only one small task at a time,
3. validate each task before moving forward.

Each Codex task should explicitly specify:

- context,
- scope,
- non-goals,
- expected files,
- constraints,
- verification steps.