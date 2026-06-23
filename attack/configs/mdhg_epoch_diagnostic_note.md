# MDHG fixed-epoch diagnostics

MDHG does not support validation-best checkpoint export. Use this path only for
fixed-epoch diagnostics; do not describe the output as valbest.

Minimal MDHG diagnostic block:

```yaml
victims:
  enabled: [mdhg]
  params:
    mdhg:
      train:
        epochs: 10
        batch_size: 100
        lr: 0.001
        checkpoint_protocol: fixed_epoch
        validation_enabled: false
        export_model: last
  runtime:
    mdhg:
      diagnostics:
        epoch_metrics: true
        per_epoch_predictions: true
```

Run clean diagnostics with a Diginetica or Yoochoose1_64 clean MDHG config:

```powershell
python -m attack.pipeline.runs.run_clean --config attack/configs/<diginetica-mdhg-diagnostic>.yaml
python -m attack.pipeline.runs.run_clean --config attack/configs/<yoochoose1_64-mdhg-diagnostic>.yaml
```

Diagnostic artifacts stay under the victim run directory:

```text
victims/mdhg/diagnostics/per_epoch_predictions/epoch_001_predictions.json
victims/mdhg/diagnostics/mdhg_epoch_diagnostic.json
victims/mdhg/diagnostics/mdhg_epoch_diagnostic.csv
```

These files are not formal shared predictions and must not be written to
`outputs/shared`.
