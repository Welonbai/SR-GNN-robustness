# Phase 2 Plan: Position-Only Insertion Prototype for SBR Robustness

## 1. Purpose

This document defines **phase 2** of the SBR robustness implementation plan.

Phase 1 already established a shared attack scaffold that supports:

1. a reproducible clean SR-GNN run,
2. a DP-SBR-baseline-compatible attack run,
3. a shared upper-half pipeline for later attack variants.

The purpose of phase 2 is **not** to redesign the scaffold and **not** to introduce a full optimization attack.

Instead, phase 2 only aims to test the following research question:

> Under the same fake-session-generation backbone, if we replace only the target insertion step, can a position-aware insertion policy produce a stable improvement over the DP-SBR-compatible random top-k% replacement baseline?

In other words, phase 2 isolates the contribution of the **insertion policy**.

---

## 2. Phase-2 Research Meaning

The DP-SBR-compatible baseline pipeline already does the following:

1. sample fake-session parameters from the clean training distribution,
2. train a poison/surrogate model,
3. iteratively generate fake sessions,
4. smooth generation scores,
5. randomly replace one item among the top-k% positions with the target item,
6. build the poisoned dataset,
7. retrain and evaluate the victim model.

Phase 2 keeps steps 1 to 7 unchanged **except for step 5**.

This phase is meant to test whether the target item's **position inside a generated fake session** is already an important optimization variable by itself.

If phase 2 shows a stable gain, then later phases can extend naturally toward:

- local context optimization,
- structured local search,
- stealth-aware insertion objectives,
- transfer / black-box validation.

---

## 3. What Phase 2 Includes

Phase 2 includes only:

- baseline-definition verification,
- a new insertion policy called `best_position_replace`,
- a new pipeline entry point for the position-only prototype,
- a new phase-2 config file,
- minimal evaluation and logging needed to compare baseline vs. phase-2 method.

---

## 4. What Phase 2 Does Not Include

Phase 2 must **not** implement:

- local context optimization,
- beam search,
- bilevel optimization,
- Gumbel-Softmax relaxation,
- stealth objectives,
- black-box validation,
- multi-model generalized evaluation,
- new fake-session-generation strategies,
- scaffold refactoring beyond what is strictly required.

The phase-1 scaffold should now be treated as **frozen**, except for small compatibility changes only if a real blocking bug is found.

---

## 5. Baseline Assumptions Already Locked

The following baseline semantics are already considered aligned and must remain unchanged in phase 2:

- targeted metric = `targeted_precision_at_k`, interpreted as whether the target item appears in top-K,
- K = 25 for the main phase-2 slice,
- popular target definition = count > average,
- unpopular target definition = count < 10,
- baseline insertion semantics = generate fake session first, then randomly replace one item among the top-k% positions with the target item,
- baseline protocol = generate -> insert -> build poisoned dataset -> retrain -> evaluate,
- dataset slice for phase 2 = Diginetica,
- attack size for phase 2 = 1%,
- target type for each run = one type only (popular or unpopular).

Do not change these baseline semantics during phase 2.

---

## 6. Phase-2 Design Principle

Phase 2 must preserve the existing design principle:

- **shared upper half**
  - session statistics,
  - fake-session parameter sampling,
  - poison-model-guided fake-session generation,
  - poisoned dataset construction,
  - training/evaluation orchestration.

- **swappable lower half**
  - insertion policy only.

That means:

- DP-SBR baseline and the phase-2 prototype must share the same scaffold,
- only the insertion policy should differ,
- the improvement, if any, should be attributable to insertion policy rather than changes elsewhere.

---

## 7. Confirmed Minimal File-Change Set

For the minimal phase-2 version, only the following files need to be **added**:

1. `attack/insertion/best_position_replace.py`
2. `attack/pipeline/run_position_opt.py`
3. `attack/configs/position_opt_diginetica_attack.yaml`

For the minimal phase-2 version, **no existing file must be modified** if the current scaffold remains compatible.

Optional convenience changes to `base_policy.py` or `evaluator.py` are explicitly **out of scope** unless a real blocker is found.

---

## 8. Files That Must Remain Unchanged

Do not change these files in the minimal phase-2 version:

- `attack/generation/fake_session_generator.py`
- `attack/generation/fake_session_parameter_sampler.py`
- `attack/generation/score_smoothing.py`
- `attack/data/poisoned_dataset_builder.py`
- `attack/data/dataset_serializer.py`
- `attack/models/srgnn_runner.py`
- `attack/pipeline/run_dp_sbr_baseline.py`
- `attack/pipeline/evaluator.py`
- `attack/common/config.py`
- `attack/data/target_selector.py`
- `datasets/preprocess.py`
- `pytorch_code/`

Reason:

Phase 2 is specifically meant to isolate the effect of **insertion policy**.  
Changing the shared upper-half generation backbone or the clean SR-GNN backbone would confound the comparison.

---

## 9. `best_position_replace` Method Definition

### 9.1 High-level idea

`best_position_replace` is a **position-only** insertion policy.

It does **not** change:

- fake-session generation,
- session length,
- local context items,
- candidate target set,
- dataset-building procedure.

It only changes **how the target item is inserted into an already generated fake session**.

### 9.2 Candidate positions

To preserve fairness relative to the DP-SBR-compatible baseline, the phase-2 method must use the **same top-k% candidate position set** as the baseline.

That means:

- baseline: random choice within the candidate position set,
- phase 2: best-score choice within the **same** candidate position set.

### 9.3 Scoring rule

For phase 2, best-position selection uses the existing poison/surrogate model signal only.

The initial scoring rule is:

1. enumerate valid candidate replacement positions,
2. for each candidate position:
   - replace the item at that position with the target item,
   - score the resulting session using the poison model,
3. choose the position that maximizes the **target item's score** under the poison model.

This is a **greedy position-only selection rule**.  
It is not yet a full optimization formulation.

### 9.4 What it is not

`best_position_replace` is **not**:

- local-context optimization,
- beam search,
- sequence extension,
- joint generation-and-insertion optimization,
- black-box scoring.

---

## 10. Files to Add

### 10.1 `attack/insertion/best_position_replace.py`

Responsibilities:

- accept a generated fake session and the target item,
- identify the valid candidate replacement positions under the baseline top-k% semantics,
- create candidate sessions by replacing each valid position with the target item,
- evaluate each candidate session using the existing poison/surrogate model score,
- choose the best position,
- return the final poisoned session,
- optionally store the selected position for logging.

### 10.2 `attack/pipeline/run_position_opt.py`

Responsibilities:

- reuse the same upper-half pipeline as `run_dp_sbr_baseline.py`,
- switch only the insertion policy from `random_topk_replace` to `best_position_replace`,
- save outputs using the same experiment-artifact structure as the baseline,
- produce directly comparable evaluation outputs.

### 10.3 `attack/configs/position_opt_diginetica_attack.yaml`

Responsibilities:

- define the phase-2 experiment slice,
- keep baseline-aligned settings,
- introduce only the minimum additional fields needed by `best_position_replace`.

---

## 11. Expected Control Flow of `run_position_opt.py`

The phase-2 pipeline should conceptually follow this order:

1. load config,
2. load clean train/test data,
3. compute or load attack-relevant session statistics,
4. select target item(s),
5. train poison/surrogate model,
6. sample fake-session parameters,
7. generate fake sessions,
8. apply `best_position_replace`,
9. build poisoned dataset,
10. retrain victim model,
11. evaluate targeted P@25,
12. save artifacts.

The structure should remain parallel to `run_dp_sbr_baseline.py`.  
The only intended method difference is the insertion policy.

---

## 12. Minimal Experiment Slice for Phase 2

To keep phase 2 small and interpretable, the first experiment slice should be fixed as:

- dataset: Diginetica,
- metric: P@25,
- attack size: 1%,
- one target type per run,
- identical clean data, fake-session-generation backbone, and target pool across compared methods.

The minimal comparison should be:

1. no attack,
2. DP-SBR-compatible random insertion baseline,
3. best-position insertion prototype.

---

## 13. Required Verification for Phase 2

Before expanding phase 2 further, verify the following:

### 13.1 Baseline reproducibility sanity check

Run the DP-SBR-compatible baseline under fixed seeds and confirm:

- output schema is stable,
- target sampling is consistent,
- targeted P@25 meaning is unchanged.

### 13.2 Position-only method sanity check

Run the phase-2 prototype and verify:

- it uses the same fake-session-generation backbone as the baseline,
- it uses the same candidate position semantics as the baseline,
- only the insertion policy differs,
- the selected best position is actually used in the final session.

### 13.3 Minimal comparison check

Confirm that the evaluation artifacts clearly support comparison among:

- clean,
- random insertion baseline,
- best-position insertion.

---

## 14. Success Criterion for Phase 2

Phase 2 is successful if it provides a trustworthy answer to the following question:

> Under the current shared attack scaffold, does replacing random top-k% insertion with position-aware best-position insertion produce a stable gain in targeted P@25?

A successful phase 2 does **not** need to prove a final optimized attack.  
Its purpose is to establish whether **target position itself** is an attack-relevant variable worth optimizing further.

---

## 15. Codex Constraints

When implementing phase 2, follow these rules strictly:

1. do not refactor the scaffold broadly,
2. do not implement future-phase ideas,
3. do not touch the generation backbone unless a real blocking bug is found,
4. keep every change small and reviewable,
5. preserve config-driven reproducibility,
6. maximize code reuse between baseline and phase-2 pipeline,
7. keep the phase-2 method difference limited to insertion policy.

---

## 16. Recommended Codex Task Order

### Task 1
Implement `attack/insertion/best_position_replace.py`.

### Task 2
Implement `attack/pipeline/run_position_opt.py`.

### Task 3
Add `attack/configs/position_opt_diginetica_attack.yaml`.

### Task 4
Run a minimal verification pass and confirm:
- clean run still works,
- DP-SBR baseline still works,
- phase-2 position-only prototype works,
- the only intended method difference is insertion policy.

### Task 5
Run the minimal comparison slice:
- clean,
- baseline random insertion,
- phase-2 best-position insertion.

---

## 17. Deliverable of Phase 2

At the end of phase 2, the repository should support:

1. clean SR-GNN run,
2. DP-SBR-compatible baseline attack run,
3. position-only insertion prototype run,
4. reproducible comparison among the three under the same scaffold.

Phase 2 is successful if it gives a clear initial answer to whether **position-aware target insertion alone** already improves attack effectiveness under the existing fake-session-generation backbone.
