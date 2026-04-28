# RankBucket-CEM Surrogate Reward Audit

Scope: code and existing artifacts only. No method behavior was changed and no expensive experiment was run.

Runs audited for target `5334`, victim `srgnn`:

- `mean_ft20`: `outputs/runs/diginetica/attack_rank_bucket_cem/run_group_cadb73910d`
- `mrr10_ft20`: `outputs/runs/diginetica/attack_rank_bucket_cem_mrr10_ft20_srgnn_target5334/run_group_36f28e71f3`
- `mean_ft100`: `outputs/runs/diginetica/attack_rank_bucket_cem_mean_ft100_srgnn_target5334/run_group_11d9031bfd`
- Random-NZ baseline: `outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_8679b974a1`

## 1. Candidate Fine-Tune Protocol

Primary path:

- `attack/pipeline/runs/run_rank_bucket_cem.py::run_rank_bucket_cem`
- `attack/position_opt/cem/trainer.py::RankBucketCEMTrainer.train`
- `attack/inner_train/truncated_finetune.py::TruncatedFineTuneInnerTrainer.run`
- `attack/surrogate/srgnn_backend.py::SRGNNBackend.fine_tune`

Findings:

- Each CEM candidate starts from the same clean surrogate checkpoint.
  - `TruncatedFineTuneInnerTrainer.run` calls `surrogate_backend.load_clean_checkpoint(clean_checkpoint_path)` and `surrogate_backend.clone_clean_model()` for every candidate (`attack/inner_train/truncated_finetune.py:22`, `:38`, `:39`).
  - The checkpoint used in all audited runs is `outputs/surrogates/diginetica/clean_srgnn_surrogate_from_attack_a7fd31f6af.pt`.
- The model is reloaded/cloned for every candidate.
  - `SRGNNBackend.clone_clean_model()` creates a fresh runner, loads the clean state dict, and returns a new model handle (`attack/surrogate/srgnn_backend.py:80`).
- Optimizer state is reset for every candidate.
  - A fresh SR-GNN model is built in `SRGNNBackend._new_runner()`.
  - SR-GNN model construction creates a new Adam optimizer and scheduler (`pytorch_code/model.py:70`, `:71`).
  - Only the model state dict is loaded, not optimizer state.
- Dataloader object is rebuilt for every candidate.
  - `SRGNNBackend.fine_tune()` constructs `Data((sessions, labels), shuffle=True)` for each call (`attack/surrogate/srgnn_backend.py:116`).
- Each candidate uses the same derived surrogate train seed.
  - `RankBucketCEMTrainer.train()` derives `shared_surrogate_train_seed` once per target (`attack/position_opt/cem/trainer.py:180`).
  - It passes the same seed to every candidate (`attack/position_opt/cem/trainer.py:268`, `:273`).
  - Current target `5334` uses `shared_surrogate_train_seed = 148798948`.
- Each candidate uses the same clean train and validation data. The only candidate-varying component is `poisoned_fake_sessions`, produced from that candidate's sampled positions.

## 2. Candidate Fine-Tune Training Data

Relevant code:

- Clean train prefixes are expanded from canonical train sessions in `build_clean_pairs()` (`attack/pipeline/core/pipeline_utils.py:86`).
- Candidate poisoned data is built in `build_poisoned_dataset()` (`attack/data/poisoned_dataset_builder.py:28`).
- CEM calls `build_poisoned_dataset(clean_sessions, clean_labels, poisoned_fake_sessions)` per candidate (`attack/position_opt/cem/trainer.py:263`).

For target `5334`:

| Quantity | Value |
|---|---:|
| Clean train prefix examples | 649,932 |
| Shared fake sessions | 6,499 |
| Poison prefix examples from fake sessions | 25,248 |
| Fine-tune dataset examples | 675,180 |
| Example-level poison ratio | 3.7394% |

The fine-tune dataset is exactly:

```text
clean_train_prefixes + expanded poisoned_fake_sessions
```

Poison sessions are appended once. They are not duplicated, oversampled, or weighted.

`poison_train_only` is relevant and required:

- `run_rank_bucket_cem.py::_validate_rank_bucket_cem_run_config()` rejects CEM if `data.poison_train_only` is false (`attack/pipeline/runs/run_rank_bucket_cem.py:265`, `:266`).
- The split key also includes `poison_train_only` (`attack/common/paths.py:124`, `:141`).

Clean/poison source is not distinguishable once inside the SR-GNN `Data` object:

- `PoisonedDataset` stores `clean_count` and `fake_count`, but only session/label arrays are passed into SR-GNN data (`attack/data/poisoned_dataset_builder.py:10`, `:49`).
- `_coerce_poisoned_train_data()` discards source counts and returns only normalized sessions and labels (`attack/surrogate/srgnn_backend.py:273`, `:286`).

## 3. Does `fine_tune_steps` Mean Optimizer Steps?

Relevant code:

- `fine_tune_steps` is read from `attack.position_opt.fine_tune_steps`.
- CEM creates `TruncatedFineTuneConfig(steps=fine_tune_steps, epochs=1)` (`attack/position_opt/cem/trainer.py:189`).
- `SRGNNBackend.fine_tune()` reads `step_limit = int(config.steps)` and `epoch_limit = int(config.epochs)` (`attack/surrogate/srgnn_backend.py:118`, `:119`).
- It increments `completed_steps` once per optimizer update (`attack/surrogate/srgnn_backend.py:138` to `:151`).

For the audited runs, yes: `20` and `100` mean actual optimizer update steps, because one full epoch has about `ceil(675180 / 100) = 6752` batches, and both step limits are far below that cap.

Important cap:

- Because CEM hard-codes `epochs=1`, `fine_tune_steps` is actually `min(configured_steps, number_of_batches_in_one_epoch)`.
- For this dataset, the cap is about `6752` steps, so `20`, `100`, and `300` would all be honored.

No early stopping exists in this fine-tune loop. Gradient updates do happen each step:

- `optimizer.zero_grad()`
- forward pass
- loss backward
- `optimizer.step()`

The current CEM artifacts do not store `inner_result.history`, so actual completed steps are not visible post-hoc in `cem_trace.jsonl`. The backend returns them, but CEM does not currently persist them.

Minimal debug fields to add per candidate:

- `configured_fine_tune_steps`
- `actual_optimizer_steps`
- `actual_batches_seen`
- `fine_tune_seconds`
- `score_target_seconds`
- `candidate_total_seconds`

Best location: around `inner_trainer.run()` and `surrogate_backend.score_target()` in `RankBucketCEMTrainer.train()` (`attack/position_opt/cem/trainer.py:268`, `:276`).

## 4. Poison Signal Seen During Fine-Tune

Batch size is `100`, inherited from SR-GNN poison model train config.

Dataset:

- Clean examples: `649,932`
- Poison prefix examples: `25,248`
- Total examples: `675,180`
- Poison ratio: `3.7394%`

Expected poison exposure:

| Fine-tune steps | Rows seen | Expected poison examples |
|---:|---:|---:|
| 20 | 2,000 | 74.79 |
| 100 | 10,000 | 373.94 |
| 300 | 30,000 | 1,121.83 |

Reconstructing the deterministic shuffle with `shared_surrogate_train_seed = 148798948` gives these actual counts in the current environment:

| Fine-tune steps | Clean examples seen | Poison prefix examples seen | Unique poison sessions seen | Unique poison session coverage |
|---:|---:|---:|---:|---:|
| 20 | 1,922 | 78 | 77 | 1.18% |
| 100 | 9,633 | 367 | 346 | 5.32% |
| 300 | 28,930 | 1,070 | 933 | 14.36% |

Because CEM resets the same seed for every candidate, the same shuffled row positions are used for every candidate. The poisoned content differs by candidate, but the clean/poison row exposure pattern is the same.

All poison sessions cannot be seen within 20 or 100 steps. One full epoch would require about `6752` batches.

Minimal instrumentation for exact future tracking:

- Add optional source flags before converting to SR-GNN `Data`, for example `0=clean`, `1=poison`.
- Preserve source flags through shuffle, or log the shuffled source flags before constructing batches.
- Record per candidate:
  - `clean_examples_seen`
  - `poison_examples_seen`
  - `poison_ratio_seen`
  - `unique_poison_sessions_seen`
  - `actual_optimizer_steps`

This can be done without changing sampling or training behavior.

## 5. Timing and Validation Cost

Current validation:

- `run_metadata.validation_session_count = 18,667` means canonical valid sessions.
- `trainer_result.validation_subset_effective_size = 69,538` means expanded validation prefixes used by `score_target`.
- Strategy is `full_validation_set`, seed is `null`.
- Selection is implemented in `_select_validation_subset()` (`attack/position_opt/cem/trainer.py:741`).

Approximate CEM timing from single-target runs:

| Run | CEM/pre-victim gap | Candidate count | Approx seconds per candidate |
|---|---:|---:|---:|
| `mrr10_ft20` | 570.15 sec | 24 | 23.76 sec |
| `mean_ft100` | 632.32 sec | 24 | 26.35 sec |

The extra 80 steps per candidate added only about 62 sec total, or about 2.6 sec per candidate.

Batch count estimate:

- One `score_target` over full validation: `ceil(69538 / 100) = 696` eval batches.
- CEM does 1 clean score plus 24 candidate scores: about `17,400` eval batches.
- Fine-tune update batches:
  - ft20: `24 * 20 = 480`
  - ft100: `24 * 100 = 2,400`
  - ft300: `24 * 300 = 7,200`

This strongly suggests validation scoring and fixed per-candidate overhead dominate ft20, and still likely dominate ft100. Dataset construction also repeatedly materializes about `675,180` examples per candidate.

Minimal timing insertion points:

- Dataset build: around `build_poisoned_dataset()` (`attack/position_opt/cem/trainer.py:263`).
- Fine-tune: around `inner_trainer.run()` (`attack/position_opt/cem/trainer.py:268`).
- Target scoring: around `score_target()` (`attack/position_opt/cem/trainer.py:276`).
- GT penalty scoring if enabled: around `score_gt()`.
- Candidate total: outer candidate loop.

## 6. Reward Behavior

Relevant code:

- `_resolve_reward_value()` (`attack/position_opt/cem/trainer.py:787`)
- `_resolve_reward_target_utility()` (`attack/position_opt/cem/trainer.py:805`)
- `_selected_reward_metric_name()` (`attack/position_opt/cem/trainer.py:673`)

Confirmed behavior:

- If `rank_bucket_cem.reward_metric = null`, reward value is `target_result.mean`.
- If `rank_bucket_cem.reward_metric` is set, reward value is `target_result.metrics[key]`.
- The CEM resolver has no low-k whitelist. It accepts any key present in `target_result.metrics`.
- The current SR-GNN backend only computes these target metric keys for surrogate scoring:
  - `targeted_mrr@10`
  - `targeted_recall@10`
  - `targeted_recall@20`
- `delta_lowk_rank_utility` is not used by RankBucket-CEM.
- CEM restricts `position_opt.reward_mode` to `poisoned_target_utility` or `delta_target_utility` (`attack/position_opt/cem/trainer.py:54`, `:805`; `attack/pipeline/runs/run_rank_bucket_cem.py:287`).

Current configs:

| Run | fine_tune_steps | reward_metric | selected reward |
|---|---:|---|---|
| `mean_ft20` | 20 | `null` | `target_result.mean` |
| `mrr10_ft20` | 20 | `targeted_mrr@10` | `targeted_mrr@10` |
| `mean_ft100` | 100 | `null` | `target_result.mean` |

`target_result.mean` is mean target softmax probability over validation prefixes, not a ranking metric. Metrics are computed separately from top-k ranking in `SRGNNBackend._score_item_probabilities()` (`attack/surrogate/srgnn_backend.py:218`, `:244`).

## 7. Delta Reward

For current runs, `delta_target_utility` would not change candidate ranking.

Reason:

- `clean_reward_baseline` is computed once per target before the candidate loop (`attack/position_opt/cem/trainer.py:193`, `:199`).
- Candidate delta is `candidate_reward_value - clean_reward_baseline` (`attack/position_opt/cem/trainer.py:805`).
- Subtracting a constant preserves ranking.

Artifact baselines:

| Run | clean_reward_baseline |
|---|---:|
| `mean_ft20` | 0.000037360649 |
| `mrr10_ft20` | 0.000110742237 |
| `mean_ft100` | 0.000037360649 |

If `enable_gt_penalty` is enabled, the final objective can change due to candidate-dependent poisoned GT utility, but that is not active in these runs.

## 8. Final Output Replay

Confirmed for target `5334`:

| Run | final seed equals best seed | Rebuilt sessions match `optimized_poisoned_sessions.pkl` |
|---|---:|---:|
| `mean_ft20` | yes, `166061354` | yes |
| `mrr10_ft20` | yes, `166061354` | yes |
| `mean_ft100` | yes, `1805458957` | yes |

Relevant code:

- During training, the best candidate stores `_best_selection_seed` and `_best_poisoned_sessions` (`attack/position_opt/cem/trainer.py:351` to `:359`).
- `export_final_poisoned_sessions()` returns `_best_poisoned_sessions` (`attack/position_opt/cem/trainer.py:454`).
- `save_artifacts()` writes `optimized_poisoned_sessions.pkl` from the exported best sessions (`attack/position_opt/cem/trainer.py:459` to `:475`).
- Replay metadata saves `final_selection_seed`, best `pi_g2`, and best `pi_g3` (`attack/position_opt/cem/trainer.py:622` to `:670`).

No mismatch found.

## 9. CEM vs Random-NZ Fairness Check

Confirmed matches:

- Same canonical split key: `split_diginetica_unified_trainonly1_minitems5_minsess2_testdays7_valid0p1`
- Same shared fake-session key: `attack_shared_1c4345bfa3`
- Same shared fake sessions path for CEM: `outputs/shared/diginetica/attack/attack_shared_1c4345bfa3/fake_sessions.pkl`
- Same target item for the audited slice: `5334`
- Same `attack.size = 0.01`
- Same fake session count: `6,499`
- Same clean train prefix count: `649,932`
- Same `replacement_topk_ratio = 1.0`
- Same `poison_train_only = true`
- Same fake session seed: `20260405`
- Same poison model: SR-GNN with train batch size `100`, epochs `8`, hidden size `100`

Main methodological mismatch:

- Random-NZ samples uniformly from nonzero positions in the top-k candidate range (`attack/insertion/random_nonzero_when_possible.py:27`).
- CEM uses the same top-k/nonzero candidate semantics but optimizes a global rank-bucket policy:
  - `G1`: rank1 only
  - `G2`: rank1/rank2
  - `G3`: rank1/rank2/tail, with tail uniform inside tail positions
- Therefore the available positions are aligned, but the policy family is not identical to Random-NZ's per-session uniform distribution.

## 10. Poison-Balanced Fine-Tune Feasibility

Current dataloader cannot distinguish clean vs poisoned examples once `Data` is built.

Least invasive options:

### Option A: Oversample poisoned examples before `Data`

Implementation:

- Modify `build_poisoned_dataset()` or add a CEM-only wrapper to repeat fake prefix examples until a target clean:poison ratio is reached.
- Keep final victim training unchanged by applying this only inside CEM surrogate evaluation.

Pros:

- Lowest code surface.
- No sampler or batch code refactor.

Cons:

- Duplicates poison examples.
- Still relies on shuffle; per-batch ratio is approximate.

### Option B: Fixed clean:poison batches

Implementation:

- Preserve source flags and build batches by sampling clean and poison subsets separately.
- Requires changing or bypassing `pytorch_code.utils.Data.generate_batch()`.

Pros:

- Strong control over exposure.

Cons:

- More invasive.
- Higher risk of accidentally changing SR-GNN batch semantics.

### Option C: Two-phase fine-tune

Implementation:

- First run poison-heavy or poison-only short fine-tune.
- Then run normal mixed fine-tune.

Pros:

- Less invasive than fixed-ratio batches.
- More direct poison exposure.

Cons:

- Adds another training knob and may overfit surrogate reward.

Recommended first implementation if needed:

- Option A, CEM-only oversampling, behind config:

```yaml
rank_bucket_cem:
  surrogate_eval_poison_balance:
    enabled: true
    mode: oversample_poison
    clean_poison_ratio: 4.0
```

Files likely touched:

- `attack/common/config.py` for config schema.
- `attack/position_opt/cem/trainer.py` to apply only before CEM surrogate evaluation.
- Possibly `attack/data/poisoned_dataset_builder.py` if source-aware helper is added.
- Tests in `attack/tests/test_rank_bucket_cem.py`.

Final victim training should remain unchanged by only applying this transformation before `inner_trainer.run()` in CEM, not in `run_targets_and_victims()`.

## 11. Actionable Conclusions

A. Is `fine_tune_steps=100` actually doing 100 optimizer steps?

Yes for the audited Diginetica runs. The code does one optimizer step per batch and stops at `fine_tune_steps`. The only cap is one epoch, about `6752` batches here.

B. How many poisoned examples are likely seen at ft20 and ft100?

With the current deterministic shuffle:

- ft20 sees 78 poison prefix examples from 77 unique fake sessions.
- ft100 sees 367 poison prefix examples from 346 unique fake sessions.
- ft300 would see about 1,070 poison prefix examples from 933 unique fake sessions.

C. Is surrogate reward likely dominated by clean-data updates or validation scoring?

The fine-tune updates are dominated by clean rows because poison exposure is only about 3.7% of examples. Wall time is likely dominated by validation scoring and fixed candidate overhead, especially at ft20 and still substantially at ft100.

D. Is `target_result.mean` likely a faithful proxy?

Not fully. It is mean target probability, not final ranking utility. It can increase while final victim MRR/Recall does not transfer.

E. Is delta reward likely to help?

No for current settings. It subtracts a per-target constant baseline and preserves candidate ranking.

F. Is final export replaying the best candidate correctly?

Yes. Rebuilding final sessions from best `pi_g2`, `pi_g3`, and `final_selection_seed` exactly matches `optimized_poisoned_sessions.pkl` for target `5334`.

G. Is poison-balanced fine-tune feasible with minimal changes?

Yes. The least invasive first version is CEM-only poison oversampling before `SRGNNBackend.fine_tune()`, gated by config. This should not affect final victim training.

H. Top 3 likely causes of proxy misalignment from this audit:

1. Poison exposure is very low in surrogate fine-tune: ft20 sees only 77 unique fake sessions and ft100 sees only 346 out of 6,499.
2. `target_result.mean` optimizes target probability, which is not the same as final ranking metrics like MRR/Recall under full victim retraining.
3. The surrogate evaluator is a short, deterministic fine-tune from a clean SR-GNN checkpoint, while final victim training is full retraining; this creates a large dynamics mismatch even when using the same SR-GNN architecture.
