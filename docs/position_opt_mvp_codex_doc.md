# MVP Design Doc for Codex: Learnable Position Optimization (SBR Poisoning)

## 1. Goal

Implement the **first MVP** of a learnable replacement-position optimization module for the existing DPSBR-based attack scaffold.

This MVP must be implemented as an **external add-on module**, not as a rewrite of the current pipeline.

The core idea is:

- reuse existing **saved fake sessions**
- reuse existing `replacement_topk_ratio` candidate semantics
- keep the current fake-session generator unchanged
- add a new **learnable position optimizer**
- use **SR-GNN** as the first surrogate backend
- use **truncated fine-tuning from a clean surrogate checkpoint**
- support **GT penalty hook**, but keep it **disabled by default** in MVP-v1

---

## 2. Scope

### In scope for MVP-v1
1. Load existing saved fake sessions.
2. Build candidate positions using current `replacement_topk_ratio` semantics.
3. Add a **per-session learnable logits policy**.
4. Use **ST-Gumbel** during training and **argmax** during final export/eval.
5. Use **hard replacement** to insert the target item.
6. Add an **SR-GNN surrogate backend wrapper** with a replaceable interface.
7. Add an **inner trainer** using **truncated fine-tuning** from a clean checkpoint.
8. Add an **outer objective** based on target-side soft utility.
9. Add **GT penalty support**, but keep it **off by default**.
10. Export final optimized poisoned sessions that can be fed into the existing evaluation pipeline.

### Explicitly out of scope for MVP-v1
1. No contextual policy network yet.
2. No multi-surrogate training.
3. No generator-position joint optimization.
4. No realism penalty yet.
5. No full refactor of the current attack pipeline.
6. No forced YAML wiring in this first step (use Python defaults first).
7. No change to current heuristic baselines' behavior.

---

## 3. Existing assumptions to preserve

These assumptions must remain true in MVP-v1:

1. **Fake sessions already exist** and should be reused directly.
2. **`replacement_topk_ratio` already exists in YAML** and its current semantics should be preserved.
3. Candidate positions should follow current top-k ratio behavior:
   - `topk_count = ceil(session_length * replacement_topk_ratio)`
   - candidates are positions within this prefix range
   - if ratio becomes `1.0`, this naturally becomes full-session candidate coverage
4. Existing baseline attack scripts and evaluation scripts must continue to work unchanged.
5. Existing heuristic policies (`dpsbr_baseline`, `random_nonzero_when_possible`, `prefix_nonzero_when_possible`) must not be broken.

---

## 4. Architecture requirements

The new MVP must be modular from the start.

### Required module separation
Do **not** implement everything inside one script.

Use these separated modules:

```text
attack/
  position_opt/
    __init__.py
    candidate_builder.py
    policy.py
    selector.py
    poison_builder.py
    objective.py
    trainer.py
    artifacts.py
    types.py

  surrogate/
    __init__.py
    base.py
    srgnn_backend.py

  inner_train/
    __init__.py
    base.py
    truncated_finetune.py

  pipeline/
    runs/
      run_position_opt_mvp.py
```

If the repository structure requires minor path adjustment, preserve the same logical separation.

### Key principle
The trainer must depend on **interfaces / abstractions**, not directly on SR-GNN-specific code internals.

---

## 5. Core design

### 5.1 Candidate builder
**Purpose:** build candidate replacement positions for one fake session.

#### Behavior
- Input: session, `replacement_topk_ratio`
- Output: list of candidate indices
- No heuristic ranking
- No hardcoded exclusion of position 0
- Reuse current semantics

#### Suggested function
```python
def build_candidate_positions(session: list[int], replacement_topk_ratio: float) -> list[int]:
    ...
```

---

### 5.2 Position policy
**MVP policy type:** per-session learnable logits

For each fake session, maintain learnable logits over its candidate positions.

#### Behavior
- one learnable vector per fake session
- vector size = number of candidate positions for that session
- no shared network yet
- no context features yet

#### Suggested API
```python
class PerSessionLogitPolicy(nn.Module):
    def __init__(self, candidate_sizes: list[int]):
        ...
    def get_logits(self, session_idx: int) -> torch.Tensor:
        ...
```

---

### 5.3 Selector
**Training selector:** Straight-Through Gumbel-Softmax  
**Eval selector:** argmax

#### Behavior
- during training: differentiable discrete selection
- during export/eval: deterministic argmax

#### Suggested API
```python
def select_position_train(logits: torch.Tensor, temperature: float) -> tuple[int, torch.Tensor]:
    ...
```

```python
def select_position_eval(logits: torch.Tensor) -> int:
    ...
```

The training function may return:
- selected local candidate index
- one-hot / relaxed one-hot tensor for backprop support

---

### 5.4 Poison builder
**Purpose:** hard replace the selected position with the target item.

#### Behavior
- pure hard replacement
- no soft session mixing

#### Suggested API
```python
def replace_item_at_position(session: list[int], position: int, target_item: int) -> list[int]:
    ...
```

---

### 5.5 Surrogate backend
**MVP surrogate backend:** SR-GNN only  
**But must be replaceable by design.**

#### Required abstraction
Create a surrogate backend interface/protocol.

#### Suggested API
```python
class SurrogateBackend(Protocol):
    def load_clean_checkpoint(self, path: str): ...
    def clone_clean_model(self): ...
    def fine_tune(self, model, poisoned_train_data, steps: int): ...
    def score_target(self, model, eval_sessions, target_item: int): ...
    def score_gt(self, model, eval_sessions): ...
```

#### SR-GNN backend implementation
Create `srgnn_backend.py` that wraps the current SR-GNN runner/utilities.

Do **not** hardwire SR-GNN logic inside the outer trainer.

---

### 5.6 Inner trainer
**MVP inner trainer:** truncated fine-tuning from a clean surrogate checkpoint

#### Required abstraction
Create an inner trainer interface/protocol.

#### Suggested API
```python
class InnerTrainer(Protocol):
    def run(
        self,
        surrogate_backend,
        clean_checkpoint_path: str,
        poisoned_train_data,
        config,
    ):
        ...
```

#### Required MVP implementation
`truncated_finetune.py`

#### Behavior
- start from a clean SR-GNN checkpoint
- clone the clean model
- fine-tune for only a small number of steps / batches / short epochs
- do not fully retrain from scratch every outer step

---

### 5.7 Objective
Use target-side soft utility as the primary objective.

#### MVP loss
```python
loss = -target_utility + lambda_gt * gt_penalty
```

#### Requirements
- support target-side soft objective
- support GT penalty hook
- GT penalty must be **disabled by default**

#### Important GT penalty rule
Use **asymmetric GT penalty** only.

This means:
- if GT does not drop, penalty = 0
- if GT improves, penalty = 0
- only penalize when GT drops beyond tolerance

#### Suggested GT penalty
```python
gt_drop = gt_clean - gt_poison
gt_penalty = max(0, gt_drop - gt_tolerance)
```

#### Important note
Do **not** use a symmetric GT reward/penalty term that encourages GT increase.  
GT is a secondary constraint, not a co-primary objective.

---

### 5.8 Trainer
This is the core controller for the MVP.

#### Responsibilities
1. load fake sessions
2. build candidate positions
3. initialize per-session logits
4. run outer optimization loop
5. select positions
6. build poisoned sessions
7. invoke inner trainer
8. compute target utility and optional GT penalty
9. update policy parameters
10. export final optimized poisoned sessions

#### Suggested API
```python
class PositionOptMVPTrainer:
    def train(self, fake_sessions, target_item, shared_artifacts, config):
        ...
    def export_final_poisoned_sessions(self) -> list[list[int]]:
        ...
```

---

### 5.9 Artifacts
Add a helper module for new artifacts.

#### Should support
- clean surrogate checkpoint path lookup
- optimized poisoned sessions output path
- learned logits dump (optional)
- training history JSON
- selected positions dump (optional)

---

## 6. New run entry point

Add a new run script:

```text
attack/pipeline/runs/run_position_opt_mvp.py
```

### Responsibilities
1. prepare / load shared artifacts
2. load existing fake sessions
3. load clean surrogate checkpoint path
4. run `PositionOptMVPTrainer`
5. export optimized poisoned sessions
6. optionally call existing evaluation pipeline afterward

### Important
Do **not** rewrite current run scripts for heuristic baselines.

---

## 7. Clean surrogate checkpoint requirement

This is required for MVP-v1.

There must be a reusable **clean SR-GNN checkpoint** available for the inner truncated fine-tuning loop.

If such artifact does not already exist, add support for generating or locating it.

### Required outcome
The MVP trainer should always start inner optimization from the same clean surrogate state, instead of training SR-GNN from scratch on every outer step.

---

## 8. Python defaults first, YAML later

### MVP-v1 implementation rule
Use **Python-side defaults first** for new position optimization configs.

Do **not** block MVP implementation on YAML/config parser changes.

### But do not forget YAML integration later
A follow-up step must be explicitly planned for moving these defaults into YAML/config parsing.

### Suggested Python defaults for MVP
```python
POSITION_OPT_DEFAULTS = {
    "enabled": True,
    "training_selector": "st_gumbel",
    "eval_selector": "argmax",
    "outer_steps": 30,
    "policy_lr": 0.05,
    "gumbel_temperature": 1.0,
    "fine_tune_steps": 20,
    "enable_gt_penalty": False,
    "gt_penalty_weight": 0.0,
    "gt_tolerance": 0.0,
}
```

### Mandatory TODO for later step
Add a later implementation step to support a YAML section such as:

```yaml
position_opt:
  enabled: true
  training_selector: st_gumbel
  eval_selector: argmax
  outer_steps: 30
  policy_lr: 0.05
  gumbel_temperature: 1.0
  fine_tune_steps: 20
  enable_gt_penalty: false
  gt_penalty_weight: 0.0
  gt_tolerance: 0.0
```

This YAML/config integration is **not required in the first coding step**, but **must be included in the follow-up plan**.

---

## 9. Data flow

### Training flow
```text
shared fake_sessions.pkl
    -> candidate_builder
    -> per-session logits policy
    -> ST-Gumbel selector
    -> poison_builder
    -> optimized poisoned sessions
    -> surrogate backend
    -> truncated fine-tuning
    -> target utility (+ optional GT penalty)
    -> update policy
```

### Final evaluation flow
```text
trained policy / argmax selection
    -> final poisoned sessions
    -> existing poisoned dataset builder / evaluation pipeline
    -> victim training + final metrics
```

---

## 10. Implementation phases for Codex

### Phase 1: module skeleton + reusable interfaces
Implement:
- new folders/files
- `candidate_builder.py`
- `poison_builder.py`
- `surrogate/base.py`
- `surrogate/srgnn_backend.py`
- `inner_train/base.py`
- `inner_train/truncated_finetune.py`
- `artifacts.py`

Acceptance:
- code imports cleanly
- fake sessions can be loaded
- candidate positions can be built
- SR-GNN backend wrapper exists
- truncated fine-tuning entry exists

---

### Phase 2: learnable policy + outer loop
Implement:
- `policy.py`
- `selector.py`
- `objective.py`
- `trainer.py`

Acceptance:
- per-session logits exist and are trainable
- ST-Gumbel selection works
- outer loop runs
- final poisoned sessions can be exported

---

### Phase 3: run script + pipeline hookup
Implement:
- `run_position_opt_mvp.py`
- wiring to shared artifacts
- export path handling
- optional integration with existing evaluation pipeline

Acceptance:
- the new run script can execute end-to-end
- it does not break existing heuristic runs

---

### Phase 4: follow-up (not part of first coding step, but must be remembered)
Implement later:
- YAML/config support for `position_opt`
- cleaner artifact handling for clean surrogate checkpoint
- possibly decouple fake-session cache key from replacement policy key if needed

---

## 11. Invariants / non-negotiable constraints

These are strict requirements:

1. Do not rewrite the fake-session generator.
2. Do not modify baseline heuristic behavior.
3. Do not hardcode SR-GNN logic directly into the trainer.
4. Do not hardcode truncated fine-tuning directly into the trainer.
5. Do not implement policy network yet.
6. Do not make GT a symmetric co-optimization target.
7. Do not force YAML integration before MVP core works.
8. Keep the new architecture modular from the start.

---

## 12. Acceptance criteria

The MVP is considered successful when all of the following are true:

1. Existing saved fake sessions can be reused directly.
2. Candidate positions are built from current `replacement_topk_ratio` semantics.
3. A per-session learnable logits policy is trainable.
4. ST-Gumbel is used during training and argmax during final selection.
5. Target item can be hard-replaced into selected positions.
6. An SR-GNN surrogate backend can be invoked through a replaceable interface.
7. Truncated fine-tuning from a clean checkpoint works through a replaceable inner trainer interface.
8. Target-side soft utility can be computed.
9. GT penalty hook exists and is disabled by default.
10. Final optimized poisoned sessions can be exported and passed into the existing evaluation pipeline.
11. Existing baseline scripts still work.

---

## 13. Notes for Codex

This task is **not** a request for a perfect final research system.

It is a request for a **clean MVP foundation** that:
- solves only the first research step
- stays modular
- does not over-engineer
- leaves room for later upgrades:
  - YAML config support
  - contextual policy network
  - multi-surrogate
  - full-session candidate experiments
  - realism penalty
