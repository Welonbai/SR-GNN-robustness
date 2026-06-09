# Diginetica copy-CEM vs generated-CEM: why TRON transfers poorly

## Result

The copy-CEM TRON gap is reproducible in the fairer unpopular comparison, so it is not only a popular-run epoch artifact. The main source-specific mechanism is TRON's training-data semantics: the pipeline expands every fake session into all prefix-label pairs, then the TRON exporter reconstructs each pair as a sequence, and TRON again trains every next-item transition in that sequence. A length-L fake session therefore contributes L(L-1)/2 transition losses instead of L-1.

Copy templates preserve clean train prefixes and transitions. Under TRON's second expansion, those already-clean transitions are repeatedly reinforced, while generated templates contribute mostly novel transitions. The target-positive weight and target-position distributions are nearly the same, so target placement is not the primary explanation.

## Mean metrics over 10 targets

| Bucket | Source | TRON R@20 | SR-GNN R@20 | MiaSRec R@20 | Final TRON loss | Clean bigram overlap | Non-target clean overlap | Pre-target clean overlap | Exact clean prefixes | Target-positive weight |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| popular | copy | 0.308 | 0.322 | 0.592 | 3.775 | 68.2% | 82.6% | 100.0% | 37.4% | 17.6% |
| popular | generated | 0.509 | 0.290 | 0.548 | 4.509 | 15.6% | 18.9% | 21.3% | 1.9% | 17.9% |
| unpopular | copy | 0.347 | 0.324 | 0.631 | 3.267 | 67.6% | 81.9% | 100.0% | 37.4% | 17.5% |
| unpopular | generated | 0.486 | 0.288 | 0.615 | 4.345 | 14.7% | 17.8% | 21.1% | 1.9% | 17.8% |

## Interpretation

- Unpopular is the clean source ablation: both copy and generated TRON runs use the same seed, fixed-epoch protocol, and 7 epochs. TRON targeted Recall@20 is still lower for copy (0.347 vs 0.486), while copy is slightly better on SR-GNN and MiaSRec.
- Copy's pre-target transition overlap with clean train is 100%; generated is about 21%. Copy's non-target clean-transition overlap is about 82%; generated is about 18%.
- Copy reaches much lower TRON train loss, consistent with training on many repeated/easy clean transitions rather than stronger target transfer.
- CEM is optimized with an SR-GNN surrogate. Copy has equal or higher surrogate reward, but that reward does not model TRON's repeated transition weighting.
- Popular is additionally confounded: copy uses TRON epoch 3 and fixed-last SR-GNN surrogate training, while generated uses TRON epoch 4 and validation-best surrogate training. Do not attribute the full popular gap to source alone.

## Recommended confirmation experiment

Export TRON from raw clean and raw fake sequences exactly once, instead of passing already-expanded prefix-label pairs through `_pairs_to_sequences`. Re-run the unpopular copy/generated pair with the same seed and 7 epochs. If the diagnosis is correct, copy TRON performance should move substantially closer to generated, and the copy train-loss advantage should shrink.

A second controlled experiment is to keep the current exporter but reweight each reconstructed prefix so every original fake-session transition has total weight one. This isolates transition duplication from source novelty.
