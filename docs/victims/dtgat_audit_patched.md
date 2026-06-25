# DT-GAT Victim Format Audit

Phase: input/output audit only. No runtime integration has been implemented.

Source audited:

- `third_party/dtgat/main.py`
- `third_party/dtgat/train.py`
- `third_party/dtgat/Util.py`
- `third_party/dtgat/model.py`
- `third_party/dtgat/README.md`

## Original DT-GAT Input Format

DT-GAT reads binary pickle files despite the `.txt` suffix. The original entry point changes the process working directory to `third_party/dtgat`, then loads fixed relative paths:

```text
datasets/<dataset>/processed_data/train.txt
datasets/<dataset>/processed_data/test.txt
datasets/<dataset>/processed_data/all_train_seq.txt
```

Only `third_party/dtgat/datasets/load.md` is present in the repo; no sample processed pickles are bundled.

### `train.txt` and `test.txt`

Both files are expected to unpickle to a sequence of length 4:

```python
[
    item_prefixes,
    interval_prefixes,
    targets,
    session_stamps,
]
```

Observed requirements from `Data.__init__` and `Data.get_slice`:

- `item_prefixes`: `list[list[int]]`, one input prefix per training/evaluation example.
- `interval_prefixes`: `list[list[number]]`, one interval row per prefix.
- `targets`: `list[int]`, one next-item label per prefix.
- `session_stamps`: `list[number]`, one timestamp per prefix/example.
- All four top-level lists must have equal length.
- Item ID `0` is padding. Real items are expected to be positive IDs.
- Prefixes are padded internally to `max(len(seq) for seq in all_train_seq)`.
- Intervals are padded with `0` to the same padded length.
- `main.py` asserts only `len(train_seqs) == 4`; it does not validate `test.txt`.

The interval row is position-aligned with the item prefix. In local graph construction, interval `i` is used as the transition time for `seq[i] -> seq[i + 1]`; the last interval value is not used for that adjacency edge loop, but the full padded interval vector is used by the model's temporal attention. `Data.get_slice` divides intervals by `1e3`, so the original pickles appear to store item-transition intervals in milliseconds and the model consumes seconds.

`session_stamps` are normalized per batch by subtracting the minimum stamp in that batch, then converting to integer days via `/ 86400`. Unlike intervals, stamps are not divided by `1e3`, so the code assumes session stamps are already in seconds. The session-wise graph uses these relative day stamps to classify pairwise temporal order and to encode pairwise time differences.

### `all_train_seq.txt`

`all_train_seq.txt` is expected to unpickle to:

```python
list[list[int]]
```

The name suggests complete raw training sessions for global train graph construction. In the audited code, it is not used to build a global item or session graph. Its actual uses are:

- `data_masks()` computes `alllen_max = max(len(seq) for seq in all_train_seq)`.
- Train/test prefixes and intervals are padded to that max length.
- `self.all_train_seq` is stored and `self.session_id` is initialized from its row count, but neither is used later in active code.

This is still a required file. It must be non-empty, and its maximum session length must be at least as large as every exported train/test prefix length, otherwise NumPy array conversion can become ragged or fail.

## CLI, Paths, and Runtime Assumptions

Original CLI arguments:

```text
--dataset          default diginetica
--epoch            default 30
--batchSize        default 100
--seq_len          default 100
--embSize          default 100
--time_dims        default 100
--l2               default 1e-4
--lr               default 0.001
--lr_dc            default 0.1
--lr_dc_step       default 10
--layer            default 1
--beta             default 0.005
--filter           default False
--cuda             default True
--gpu              default 0
--intent_num       default 4
--random_seed      default 0
--dropout          default 0
```

Important caveats:

- `--data_dir`, `--n_node`, `--gpu_id`, `--seed`, `--topk`, and `--prediction_output_path` do not exist yet.
- `--cuda` and `--gpu` are effectively ignored.
- `CUDA_VISIBLE_DEVICES` is hardcoded to `0,1`.
- `torch.cuda.set_device(0)` is called unconditionally at import/runtime setup. A CPU-only environment will fail before `trans_to_cuda()` can fall back to CPU.
- Dataset paths are fixed under `datasets/<dataset>/processed_data`.
- `data_directory = 'processed_data'` is dead code.

Hardcoded item counts:

```text
diginetica:    43098
Retailrocket:  50020
Nowplaying:    60418
Yoochoose64:   37485
Yoochoose:     37485
Yoochoose4:    37485
```

If `--dataset` is not one of these names, `n_items` is never assigned. `n_sess` is computed as `len(train_seqs[1]) + len(test_seqs[1])`, which is effectively the number of train and test interval rows.

Seed behavior:

- `setup_seed(args.random_seed)` seeds PyTorch CPU, all CUDA devices, NumPy, and Python `random`.
- It does not set `PYTHONHASHSEED`.
- It does not configure deterministic cuDNN behavior.
- Evaluation uses `Data(..., shuffle=True)` in `main.py`, so NumPy's seeded RNG also shuffles test examples every epoch.

## Training and Evaluation Behavior

`train_test()` performs one train pass and one test pass per epoch.

Training:

- The scheduler is stepped at the start of each epoch, before optimizer updates.
- Batches are generated by `Data.generate_batch(model.batch_size)`.
- `train_batch()` constructs batch-local item transition adjacency, interval adjacency, pairwise session overlap types, and pairwise session-overlap weights.
- The model returns `(ssl_loss * beta, scores)`.
- Supervised loss is `CrossEntropyLoss(scores, targets - 1)`.
- Total loss is supervised cross entropy plus SSL loss.

Evaluation:

- The same `train_batch()` path is used under `model.eval()`.
- Metrics are computed on the test set every epoch; there is no validation split, checkpoint save, early stopping, or final-model export.
- The code records best Hit/MRR by test metrics, so the printed best epoch is test-selected.
- Fixed metric cutoffs are `[5, 10, 20]`.
- Ranking depth is fixed at 20.
- Top-k predictions are not written to disk.

`generate_batch()` has a tail-batch behavior that matters for export. If the split size is not divisible by `batchSize`, it replaces the final slice with the last full `batchSize` examples. This duplicates some examples and omits the natural short final batch. For original aggregate metrics this changes weighting; for project prediction export it must be patched to preserve one prediction per canonical test example in original order.

## Metric and Top-K Logic

`find_k_largest(K, candidates)` returns zero-based score-column indices for the largest `K` scores. It does not return scores. Evaluation always calls `find_k_largest(20, scores[row])`, then truncates that index array for K in `[5, 10, 20]`.

Metric target shift:

- Training label: `targets - 1`
- Evaluation match: `target - 1`
- Score column `0` corresponds to external item ID `1`.
- Score column `i` corresponds to external item ID `i + 1`.

Hit@K is `target - 1 in prediction[:K]`. MRR@K is reciprocal rank within `prediction[:K]`, or `0` if absent. `main.py` converts each metric list to a percentage with `np.mean(...) * 100`.

Tie behavior is not explicitly specified. The heap code sorts by score descending only, so equal-score ordering should not be treated as a stable contract for evaluator output.

## Output Scores and Item ID Mapping

`EnHSG` creates `nn.Embedding(n_node + 1, emb_size, padding_idx=0)`. Prediction scores are computed against:

```python
b = self.embedding.weight[1:]
score = torch.matmul(TD_session, b.transpose(1, 0))
```

Therefore the score tensor shape is:

```text
[batch_size, n_node]
```

The columns exclude padding ID `0`. Conversion for project predictions must be:

```text
external_item_id = score_column_index + 1
```

No candidate masking is applied for items already present in the prefix. The model can rank previously seen items unless a later integration patch explicitly changes this, which would alter original behavior.

## Internal Item ID Convention

DT-GAT is internally one-based for real item IDs, with `0` reserved for padding:

- Input prefixes and targets should contain real IDs in `1..n_node`.
- Padding ID is `0`.
- The embedding has row `0` for padding and rows `1..n_node` for real items.
- Loss and metrics shift labels by `-1` only because score columns are zero-based after removing the padding embedding row.

For the future `dtgat` victim, the exporter should preserve canonical project item IDs as dense one-based IDs. If the canonical dataset item universe differs from DT-GAT's hardcoded original counts, `--n_node` must override the original table.

## Original DT-GAT Output Behavior

Original DT-GAT only prints:

- parsed args,
- train/test row counts,
- epoch number,
- periodic training loss,
- raw metric lists before aggregation,
- best Hit/MRR percentages for K in `[5, 10, 20]`.

It does not export:

- model checkpoints,
- train history JSON,
- metrics JSON,
- per-example predictions,
- top-k score values,
- top-k item IDs.

Any project evaluator integration will need a new prediction export path.

## Required Exporter Format

The future project exporter should create a DT-GAT data directory containing:

```text
processed_data/train.txt
processed_data/test.txt
processed_data/all_train_seq.txt
```

Recommended pickle payloads:

```python
# processed_data/train.txt
[
    train_prefixes,       # list[list[int]]
    train_intervals,      # list[list[float|int]]
    train_labels,         # list[int]
    train_session_stamps, # list[int|float]
]

# processed_data/test.txt
[
    test_prefixes,        # list[list[int]], canonical evaluation order
    test_intervals,       # list[list[float|int]]
    test_labels,          # list[int]
    test_session_stamps,  # list[int|float]
]

# processed_data/all_train_seq.txt
train_full_sessions       # list[list[int]]
```

Exporter requirements:

- Use dense one-based item IDs in `1..item_count`; reserve `0` for DT-GAT padding only.
- Export explicit prefix-label examples. Do not rely on DT-GAT to expand full sessions.
- Preserve canonical test example order in `test.txt`.
- Ensure every prefix and label item ID is within `1..item_count`.
- Ensure `all_train_seq.txt` is non-empty and has max length at least every train/test prefix length.
- If authoritative timestamps exist, export transition intervals in milliseconds and session stamps in seconds.
- If no authoritative timestamps exist in the canonical dataset, decide and document a deterministic synthetic timestamp policy before running comparisons. Zero intervals/stamps are format-valid but may materially change DT-GAT behavior because temporal encoders and session-wise future-blocking depend on these fields.

## Project Integration Note: Poison Budget Semantics

This project uses a 1% expanded-prefix-pair poisoning budget, not a 1% raw-session budget. The DT-GAT exporter must not recompute `attack.size` or generate fake sessions. It must consume the already resolved clean or poisoned training sessions from the existing attack pipeline and only convert them into DT-GAT's required pickle format.

Because DT-GAT `train.txt` contains explicit prefix-label examples, any poisoned fake session should be expanded into supervised training rows:

```text
fake session of length L -> L - 1 prefix-label rows
```

For poisoned runs, export semantics should be:

```text
processed_data/train.txt:
  prefix-label examples expanded from clean training sessions plus poisoned fake sessions

processed_data/all_train_seq.txt:
  full clean training sessions plus full poisoned fake sessions
```

Do not export expanded prefix rows into `all_train_seq.txt`; it should remain a list of full training sessions.

## Required Third-Party Patches

Later integration should patch DT-GAT, not the project runtime first.

Required CLI/runtime patches:

- Add `--data_dir`: explicit data root. It should either point directly to a directory containing `processed_data/*.txt`, or to the `processed_data` directory itself; choose one convention and validate it.
- Add `--n_node`: required with `--data_dir`; replaces hardcoded `n_items`.
- Add `--gpu_id`: controls `CUDA_VISIBLE_DEVICES`; use logical CUDA device `0` after filtering.
- Honor CPU/no-CUDA behavior or fail early with a clear message. Remove unconditional `torch.cuda.set_device(0)` unless CUDA is available.
- Add `--seed` as the project-facing seed name, while optionally keeping `--random_seed` as an alias.
- Set `PYTHONHASHSEED` in the runner environment and consider deterministic cuDNN flags if reproducibility is required.
- Add `--topk`, defaulting to project needs such as `50`.
- Add `--prediction_output_path` for final top-k export.

Required data/evaluation patches:

- Load `train.txt`, `test.txt`, and `all_train_seq.txt` from `--data_dir`.
- Validate pickle structure, counts, item ID bounds, interval lengths, and timestamp counts.
- Use `shuffle=True` for training but `shuffle=False` for final prediction export.
- Replace `generate_batch()` tail handling for evaluation/export so each test example is scored exactly once.
- Preserve original test example indices through batching if any sorting/shuffling remains.
- Use `min(args.topk, n_node)` as effective export depth.
- Export item IDs, not zero-based score columns.
- Optionally include scores if useful for diagnostics, but the current project runners generally consume rankings.

Recommended prediction JSON shape for consistency with patched victims:

```json
{
  "topk": 50,
  "requested_topk": 50,
  "n_node": 43098,
  "rankings": [
    [123, 456, 789]
  ]
}
```

Each ranking row should contain exactly `topk` external one-based item IDs and there should be exactly one row per canonical test example. If using a row-object schema instead, include `example_id` and keep rows sorted by ascending `example_id`; the runner can normalize to `list[list[int]]`.

## Risks and Pitfalls

- Timestamp semantics are underspecified. Intervals are divided by `1e3`, but session stamps are divided by `86400` without a millisecond conversion. Exporting millisecond session stamps would create incorrect day offsets.
- `all_train_seq.txt` sounds like graph input, but active code only uses it for padding length. If future DT-GAT revisions use it for graph construction, the exporter should already provide complete training sessions, not expanded prefixes.
- Original evaluation shuffles test data and duplicates tail examples when the split size is not divisible by batch size. This is incompatible with canonical per-example prediction export.
- Top-k export is absent. The current `find_k_largest()` drops scores and returns zero-based columns, so export code must convert each index with `+1`.
- Hardcoded `n_items` values may not match this project's canonical item universe. `--n_node` must be authoritative for victim runs.
- GPU arguments are misleading in the original code. `--gpu` and `--cuda` do not control execution.
- CPU fallback is not actually safe because `torch.cuda.set_device(0)` is unconditional.
- Test metrics are used for best-epoch reporting. Project integration should define fixed-epoch/last-model behavior or add a real validation protocol before comparing runs.
- `model.batch_size` is embedded in `SessionHCov` and some tensor reshapes assume full batches. Evaluation/export patches should handle smaller final batches carefully or keep full batches while tracking and removing padded/duplicate rows.
- Score ties have no stable item-ID tie-break contract. If deterministic ranking under ties matters, use `torch.topk` plus an explicit ascending-item tie policy.
