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

## Phase Status

Phase 2 connects a local real Seq-poison generation path. Phase 1 was
interface+mock only and is not a reportable baseline. Mock candidate sequences
remain allowed only by tests or explicit dependency injection; experiment YAML
uses `generation_backend: real`.

Phase 1 constraints:

- `max_seq_len = min(50, train_sub p99)` by nearest-rank percentile unless an
  explicit override is configured.
- Target at position 0 is allowed and diagnosed.
- No nonzero constrained decoding.
- No length-distribution matching.
- No target movement, insertion repair, or target-preserving crop.
- Final injected fake session count must equal the requested poisoning budget.

`candidate_multiplier=1` is a diagnostic placeholder. Real generation may
require oversampling to guarantee `n_fake` valid sessions after filtering.

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

Padding id `0` is also the upstream start token and is removed during
postprocessing. Canonical internal item ids are used directly when they are
positive and contiguous. If the canonical IDs are sparse, SeqPoison-SBR trains
with a reversible dense item-id mapping and converts generated items back to
canonical IDs before postprocessing.

`CandidateGenerator.generate()` must return upstream-style fake user sequences
with a synthetic user id as the first token:

```text
[user_id, item1, item2, ...]
```

The first token is always removed by Phase 1 pipeline postprocessing. Generator
implementations must not return item-only SBR sessions.

Training uses `train_sub` only. Sessions longer than `max_seq_len` are excluded
from Seq-poison training rather than cropped, and the bridge records
`train_session_count_before_length_filter`,
`train_session_count_after_length_filter`, `excluded_train_session_count`,
`excluded_train_session_ratio`, and `max_seq_len_value`.
The optional `max_train_sequences` field is a diagnostic-only cap for count1
smoke configs; leave it unset for formal runs. The count1 smoke configs also
use a tiny `attack.size` and are generation diagnostics only, not reportable
1% baselines.

Phase 2B adds target acceptance diagnostics:

- `target_containing_candidate_count_before_single_target_filter`
- `target_containing_candidate_ratio_before_single_target_filter`

Use `attack/configs/diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_acceptance_diag.yaml`
only for generation calibration. It strengthens faithful Seq-poison generation
parameters (`candidate_multiplier`, `max_generation_rounds`,
`adversarial_epochs`, `reward_target_weight`, `target_probability`, and
`max_train_sequences`) without inserting, moving, cropping, or repairing target
items.

Candidate storage is controlled by `candidate_save_policy`:

- `summary_only`: do not save rejected raw candidates; save diagnostics only.
- `valid_only`: save final `raw_fake_sessions.pkl` only.
- `sample`: save at most `max_saved_candidates` raw candidates for debugging.
- `all`: save every raw candidate; debugging only, not a safe default.

Older configs with `save_generated_candidates: true` and no explicit
`candidate_save_policy` map to bounded `sample`, not `all`. `generated_candidates.pkl`
contains raw generated candidates only when the policy is `sample` or `all`;
otherwise only summaries and `raw_fake_sessions.pkl` are written.

Version A diagnostic command:

```powershell
python -m attack.poisoning_ssl.run_generation_diagnostic --config attack/configs/diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_versionA_diag.yaml
```

Optional PowerShell timing wrapper:

```powershell
Measure-Command { python -m attack.poisoning_ssl.run_generation_diagnostic --config attack/configs/diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_versionA_diag.yaml }
```

The local Phase 2 trainer preserves the upstream structure: classifier
pretraining, generator MLE pretraining, discriminator training, and adversarial
generator updates with target-related, bi-classifier, and GAN discriminator
reward components. One numeric adaptation is used for classifier training:
the local loss applies negative log likelihood to clamped classifier
probabilities for stability.

## Upstream Migration Map

- `classify.py` -> `model.py` `Classify` CNN bi-classifier.
- `train_classify.py` -> `trainer.py` classifier pretraining loop.
- `generator.py` -> `model.py` `Generator` and policy-gradient losses.
- `discriminator.py` -> `model.py` `Discriminator` adversarial reward model.
- `dataloader.py` -> `dataset_bridge.py` export plus `trainer.py` dataset
  adapters.
- `helpers.py` -> `model.py` / `trainer.py` batch helpers.
- `main.py` -> `trainer.py` real training orchestration.
- `generate_data.py` -> `generator.py` candidate extraction and synthetic
  user-id prepending.
- `process.py` -> `dataset_bridge.py` preprocessing assumptions only.

Upstream assumptions preserved locally:

- Item IDs start at 1 and 0 is padding/start.
- Generated samples are item-only tensors internally; the real generator
  prepends synthetic user IDs before returning `[user_id, item1, ...]`.
- Target item is represented by canonical ID externally and by remapped dense ID
  only inside Seq-poison training if remapping is required.
- Candidate count per round is `candidate_multiplier * n_fake_requested`.

The formal victim training and evaluation pipeline remains unchanged.
