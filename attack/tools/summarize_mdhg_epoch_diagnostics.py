from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.models.victim.mdhg_diagnostics import (
    summarize_mdhg_epoch_diagnostics_from_run_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute official pipeline metrics for MDHG per-epoch predictions."
    )
    parser.add_argument(
        "--victim-run-dir",
        required=True,
        help="MDHG victim run directory containing resolved_config.json.",
    )
    args = parser.parse_args()
    rows = summarize_mdhg_epoch_diagnostics_from_run_dir(args.victim_run_dir)
    output_path = Path(args.victim_run_dir) / "mdhg_epoch_pipeline_metrics.jsonl"
    print(f"Wrote {len(rows)} epoch rows to {output_path}")


if __name__ == "__main__":
    main()
