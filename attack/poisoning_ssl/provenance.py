from __future__ import annotations

UPSTREAM_URL = (
    "https://github.com/yanling02/"
    "Poisoning-Self-supervised-Learning-Based-Sequential-Recommendations"
)
UPSTREAM_COMMIT = "dc0a43821c36462528ec1eecb77ffaf0cd3cb1d8"

UPSTREAM_MIGRATION_MAP: dict[str, str] = {
    "classify.py": "model.py classifier interface",
    "train_classify.py": "trainer.py classifier training interface",
    "generator.py": "model.py/generator.py generator interface",
    "discriminator.py": "model.py discriminator interface",
    "dataloader.py": "dataset_bridge.py training data interface",
    "helpers.py": "trainer.py/generator.py helper interfaces",
    "main.py": "pipeline.py/trainer.py orchestration reference",
    "generate_data.py": "generator.py/postprocess.py candidate generation reference",
}


def provenance_payload() -> dict[str, object]:
    return {
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_migration_map": dict(UPSTREAM_MIGRATION_MAP),
        "runtime_dependency_on_external_repos": False,
        "phase": "phase1_interface_mock_only",
    }


__all__ = ["UPSTREAM_COMMIT", "UPSTREAM_MIGRATION_MAP", "UPSTREAM_URL", "provenance_payload"]
