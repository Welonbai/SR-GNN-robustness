from __future__ import annotations

UPSTREAM_URL = (
    "https://github.com/yanling02/"
    "Poisoning-Self-supervised-Learning-Based-Sequential-Recommendations"
)
UPSTREAM_COMMIT = "dc0a43821c36462528ec1eecb77ffaf0cd3cb1d8"

UPSTREAM_MIGRATION_MAP: dict[str, str] = {
    "classify.py": "model.py Classify CNN bi-classifier",
    "train_classify.py": "trainer.py classifier pretraining loop",
    "generator.py": "model.py Generator and policy-gradient losses",
    "discriminator.py": "model.py Discriminator adversarial reward model",
    "dataloader.py": "dataset_bridge.py export plus trainer.py Dataset adapters",
    "helpers.py": "model.py/trainer.py batch helpers",
    "main.py": "trainer.py real training orchestration",
    "generate_data.py": "generator.py candidate extraction and user-id prepending",
    "process.py": "dataset_bridge.py preprocessing assumptions only",
}

UPSTREAM_ASSUMPTIONS: dict[str, object] = {
    "item_ids_start_from": 1,
    "padding_id": 0,
    "start_letter": 0,
    "generated_output_contract": "[user_id, item1, item2, ...]",
    "target_representation": "canonical item id, or reversible dense seqpoison id during training",
    "max_seq_len_enforcement": "training sessions longer than max_seq_len are excluded; generated non-padding tokens are not cropped",
    "candidate_count_basis": "candidate_multiplier * n_fake_requested per generation round",
    "reward_components": [
        "target_related_reward",
        "bi_classifier_reward",
        "gan_discriminator_reward",
    ],
}


def provenance_payload() -> dict[str, object]:
    return {
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_migration_map": dict(UPSTREAM_MIGRATION_MAP),
        "upstream_assumptions": dict(UPSTREAM_ASSUMPTIONS),
        "runtime_dependency_on_external_repos": False,
        "phase": "phase2_real_generation",
    }


__all__ = [
    "UPSTREAM_ASSUMPTIONS",
    "UPSTREAM_COMMIT",
    "UPSTREAM_MIGRATION_MAP",
    "UPSTREAM_URL",
    "provenance_payload",
]
