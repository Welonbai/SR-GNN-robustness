# SeqPoison-SBR

SeqPoison-SBR is the SBR adaptation of the Poisoning-SSL / Seq-poison fake user
sequence generation attack. The original method is from "Poisoning
Self-supervised Learning Based Sequential Recommendations".

Upstream reference:

- Repository: https://github.com/yanling02/Poisoning-Self-supervised-Learning-Based-Sequential-Recommendations
- Commit: `dc0a43821c36462528ec1eecb77ffaf0cd3cb1d8`
- Reference path: `Seq-poison/`

This package is the formal location for the migrated attack method. The runtime
pipeline must not import from `external_repos`, and this attack is not a
`third_party` victim model.

## Phase 1 Status

Phase 1 is interface+mock only and is not a reportable baseline. Real
Poisoning-SSL classifier/generator/discriminator training is intentionally not
implemented yet. Mock candidate sequences may be used only by tests or explicit
dependency injection. Experiment YAML execution must fail clearly until real
generation is implemented.

Phase 1 constraints:

- `max_seq_len = min(50, train_sub p99)` by nearest-rank percentile unless an
  explicit override is configured.
- Target at position 0 is allowed and diagnosed.
- No nonzero constrained decoding.
- No length-distribution matching.
- No target movement, insertion repair, or target-preserving crop.
- Final injected fake session count must equal the requested poisoning budget.

`candidate_multiplier=1` is a Phase 1 placeholder. Real generation may require
oversampling to guarantee `n_fake` valid sessions after filtering.

## Adaptation

The upstream method emits fake user sequences such as:

```text
[user_id, item1, item2, target, itemk]
```

SeqPoison-SBR removes the synthetic user id and injects the remaining sequence
as an anonymous SBR fake session:

```text
[item1, item2, target, itemk]
```

Padding id `0` is removed during postprocessing. Canonical internal item ids are
used directly; no runtime remap is applied in Phase 1.

`CandidateGenerator.generate()` must return upstream-style fake user sequences
with a synthetic user id as the first token:

```text
[user_id, item1, item2, ...]
```

The first token is always removed by Phase 1 pipeline postprocessing. Generator
implementations must not return item-only SBR sessions.

## Upstream Migration Map

- `classify.py` -> `model.py` classifier model interface.
- `train_classify.py` -> `trainer.py` classifier training interface.
- `generator.py` -> `model.py` / `generator.py` generator interface.
- `discriminator.py` -> `model.py` discriminator interface.
- `dataloader.py` -> `dataset_bridge.py` training data interface.
- `helpers.py` -> `trainer.py` / `generator.py` helper interfaces.
- `main.py` -> `pipeline.py` / `trainer.py` orchestration reference.
- `generate_data.py` -> `generator.py` / `postprocess.py` candidate generation
  and filtering reference.

The formal victim training and evaluation pipeline remains unchanged.
