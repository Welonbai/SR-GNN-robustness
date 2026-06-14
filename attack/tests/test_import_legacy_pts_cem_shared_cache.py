from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from attack.tools import import_legacy_pts_cem_shared_cache as importer


def test_direct_action_import_uses_resolved_run_type(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    run_type = "pts_construction_direct_action_mlp_cem"
    source_dir = tmp_path / "pts_construction_cem"
    fake_sessions_path = tmp_path / "fake_sessions.pkl"
    fake_sessions_path.write_bytes(b"fake sessions")
    shared_cache_dir = tmp_path / "shared_cache"
    calls: dict[str, object] = {}

    monkeypatch.setattr(importer, "load_config", lambda _path: object())
    monkeypatch.setattr(importer, "_pts_construction_run_type", lambda _config: run_type)
    monkeypatch.setattr(
        importer,
        "build_pts_construction_attack_identity_context",
        lambda _config: {"pts_construction": {"method": "direct_action_mlp_cem"}},
    )

    def load_cached(**kwargs):
        calls["source_identity"] = kwargs["current_identity"]
        return SimpleNamespace(metadata={"target_item": 11103})

    monkeypatch.setattr(importer, "_try_load_cached_pts_best_candidate", load_cached)

    def resolve_shared_paths(_config, *, run_type: str):
        calls["shared_run_type"] = run_type
        return {
            "fake_sessions": fake_sessions_path,
            "poison_model": tmp_path / "poison_model.pt",
        }

    monkeypatch.setattr(importer, "shared_artifact_paths", resolve_shared_paths)
    monkeypatch.setattr(
        importer,
        "build_pts_cem_shared_cache_identity",
        lambda *_args, **_kwargs: {"run_type": run_type},
    )
    monkeypatch.setattr(
        importer,
        "pts_cem_shared_cache_key",
        lambda _identity: "pts_cem_shared_direct",
    )
    monkeypatch.setattr(
        importer,
        "pts_cem_shared_cache_dir",
        lambda *_args: shared_cache_dir,
    )
    monkeypatch.setattr(
        importer,
        "_validate_shared_cache_available",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(importer, "_existing_pts_artifact_paths", lambda _path: {})
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_legacy_pts_cem_shared_cache.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--target-item",
            "11103",
            "--source",
            str(source_dir),
            "--dry-run",
        ],
    )

    assert importer.main() == 0
    assert calls["source_identity"] == {"run_type": run_type}
    assert calls["shared_run_type"] == run_type
    assert "status=dry_run_valid" in capsys.readouterr().out
