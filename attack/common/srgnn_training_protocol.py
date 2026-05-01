from __future__ import annotations

from typing import Any, Mapping


SRGNN_FIXED_LAST_PROTOCOL = "fixed_last"
SRGNN_VALIDATION_BEST_PROTOCOL = "validation_best"
SRGNN_VALIDATION_BEST_METRIC = "valid_ground_truth_mrr@20"
SRGNN_VALIDATION_PATIENCE_METRIC = "recall20_or_mrr20"
SRGNN_VALIDATION_PROTOCOL_NAME = "mrr20_selected_recall20_or_mrr20_patience"

ALLOWED_SRGNN_CHECKPOINT_PROTOCOLS = {
    SRGNN_FIXED_LAST_PROTOCOL,
    SRGNN_VALIDATION_BEST_PROTOCOL,
}
ALLOWED_SRGNN_BEST_METRICS = {SRGNN_VALIDATION_BEST_METRIC}
ALLOWED_SRGNN_PATIENCE_METRICS = {SRGNN_VALIDATION_PATIENCE_METRIC}


def srgnn_checkpoint_protocol(train_config: Mapping[str, Any]) -> str:
    return str(train_config.get("checkpoint_protocol", SRGNN_FIXED_LAST_PROTOCOL)).strip().lower()


def srgnn_best_metric(train_config: Mapping[str, Any]) -> str:
    return str(train_config.get("best_metric", SRGNN_VALIDATION_BEST_METRIC)).strip().lower()


def srgnn_patience_metric(train_config: Mapping[str, Any]) -> str:
    return str(train_config.get("patience_metric", SRGNN_VALIDATION_PATIENCE_METRIC)).strip().lower()


def srgnn_validation_best_enabled(train_config: Mapping[str, Any]) -> bool:
    return srgnn_checkpoint_protocol(train_config) == SRGNN_VALIDATION_BEST_PROTOCOL


def srgnn_validation_protocol_identity(
    train_config: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}_training_protocol": srgnn_checkpoint_protocol(train_config),
        f"{prefix}_best_metric": srgnn_best_metric(train_config),
        f"{prefix}_patience_metric": srgnn_patience_metric(train_config),
        f"{prefix}_max_epochs": int(train_config["epochs"]),
        f"{prefix}_patience": int(train_config["patience"]),
    }


__all__ = [
    "ALLOWED_SRGNN_BEST_METRICS",
    "ALLOWED_SRGNN_CHECKPOINT_PROTOCOLS",
    "ALLOWED_SRGNN_PATIENCE_METRICS",
    "SRGNN_FIXED_LAST_PROTOCOL",
    "SRGNN_VALIDATION_BEST_METRIC",
    "SRGNN_VALIDATION_BEST_PROTOCOL",
    "SRGNN_VALIDATION_PATIENCE_METRIC",
    "SRGNN_VALIDATION_PROTOCOL_NAME",
    "srgnn_best_metric",
    "srgnn_checkpoint_protocol",
    "srgnn_patience_metric",
    "srgnn_validation_best_enabled",
    "srgnn_validation_protocol_identity",
]
