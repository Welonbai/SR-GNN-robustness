from __future__ import annotations

import subprocess

import pytest

from attack.data.canonical_fingerprints import resolve_wearec_repository_provenance


class FakeGit:
    def __init__(self, *, parent_status="", wearec_status="", gitlink="b", head="b"):
        self.parent_status = parent_status
        self.wearec_status = wearec_status
        self.gitlink = gitlink
        self.head = head

    def __call__(self, cmd, *, cwd, **kwargs):
        args = cmd[1:]
        if args == ["rev-parse", "HEAD:third_party/wearec"]:
            value = self.gitlink
        elif args == ["rev-parse", "HEAD"]:
            value = "parent" if str(cwd).endswith("parent") else self.head
        elif args == [
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--ignore-submodules=dirty",
        ]:
            if not str(cwd).endswith("parent"):
                raise AssertionError(args)
            value = self.parent_status
        elif args == ["status", "--porcelain", "--untracked-files=no"]:
            if str(cwd).endswith("parent"):
                raise AssertionError(args)
            value = self.wearec_status
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(cmd, 0, stdout=value + ("\n" if value else ""), stderr="")


def test_clean_matching_provenance_is_recorded(tmp_path):
    parent = tmp_path / "parent"
    wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    result = resolve_wearec_repository_provenance(
        parent, wearec, command_runner=FakeGit()
    )
    assert result["parent_repository_commit"] == "parent"
    assert result["wearec_gitlink_commit"] == result["wearec_submodule_commit"] == "b"


@pytest.mark.parametrize(
    "fake,message",
    [
        (FakeGit(gitlink="a", head="b"), "gitlink"),
        (FakeGit(parent_status=" M attack/x.py"), "Parent tracked"),
        (FakeGit(wearec_status="M  src/main.py"), "WEARec tracked"),
    ],
)
def test_dirty_or_mismatched_provenance_is_rejected(tmp_path, fake, message):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    with pytest.raises(RuntimeError, match=message):
        resolve_wearec_repository_provenance(parent, wearec, command_runner=fake)


def test_parent_python_runtime_cache_dirty_status_is_ignored(tmp_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    fake = FakeGit(
        parent_status="\n".join(
            [
                " M third_party/miasrec/recbole/__pycache__/__init__.cpython-38.pyc",
                " M third_party/miasrec/recbole/config/__pycache__/configurator.cpython-38.pyc",
                "?? third_party/tron/src/__pycache__/main.cpython-38.pyc",
                " M third_party/tron/src/cache.pyo",
            ]
        )
    )
    result = resolve_wearec_repository_provenance(
        parent, wearec, command_runner=fake
    )
    assert result["parent_tracked_worktree_clean"] is True
    assert result["parent_ignored_runtime_cache_dirty_paths"] == [
        "third_party/miasrec/recbole/__pycache__/__init__.cpython-38.pyc",
        "third_party/miasrec/recbole/config/__pycache__/configurator.cpython-38.pyc",
        "third_party/tron/src/__pycache__/main.cpython-38.pyc",
        "third_party/tron/src/cache.pyo",
    ]


@pytest.mark.parametrize("submodule_path", ["third_party/miasrec", "third_party/tron"])
def test_parent_unrelated_submodule_dirty_status_is_ignored(tmp_path, submodule_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    fake = FakeGit(parent_status=f" m {submodule_path}")
    result = resolve_wearec_repository_provenance(
        parent, wearec, command_runner=fake
    )
    assert result["parent_tracked_worktree_clean"] is True
    assert result["parent_ignored_runtime_cache_dirty_paths"] == [submodule_path]


def test_parent_real_source_dirty_status_is_rejected(tmp_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    fake = FakeGit(parent_status=" M attack/pipeline/runs/run_wearec.py")
    with pytest.raises(RuntimeError, match="run_wearec.py"):
        resolve_wearec_repository_provenance(parent, wearec, command_runner=fake)


def test_mixed_parent_status_ignores_cache_but_rejects_real_changes(tmp_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    fake = FakeGit(
        parent_status="\n".join(
            [
                " M third_party/miasrec/recbole/__pycache__/__init__.cpython-38.pyc",
                " M attack/data/canonical_fingerprints.py",
            ]
        )
    )
    with pytest.raises(RuntimeError) as exc_info:
        resolve_wearec_repository_provenance(parent, wearec, command_runner=fake)
    message = str(exc_info.value)
    assert "attack/data/canonical_fingerprints.py" in message
    assert "__pycache__" not in message


def test_wearec_python_runtime_cache_dirty_status_is_ignored(tmp_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    fake = FakeGit(wearec_status=" M src/__pycache__/train.cpython-38.pyc")
    result = resolve_wearec_repository_provenance(
        parent, wearec, command_runner=fake
    )
    assert result["wearec_tracked_worktree_clean"] is True
    assert result["wearec_ignored_runtime_cache_dirty_paths"] == [
        "src/__pycache__/train.cpython-38.pyc"
    ]


def test_wearec_real_source_dirty_status_is_rejected(tmp_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    fake = FakeGit(wearec_status=" M src/train.py")
    with pytest.raises(RuntimeError, match="src/train.py"):
        resolve_wearec_repository_provenance(parent, wearec, command_runner=fake)


def test_dirty_error_lists_only_non_ignored_paths(tmp_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    fake = FakeGit(
        parent_status="\n".join(
            [
                " M third_party/miasrec/recbole/__pycache__/configurator.cpython-38.pyc",
                " M attack/configs/some_formal_config.yaml",
            ]
        )
    )
    with pytest.raises(RuntimeError) as exc_info:
        resolve_wearec_repository_provenance(parent, wearec, command_runner=fake)
    message = str(exc_info.value)
    assert "attack/configs/some_formal_config.yaml" in message
    assert "configurator.cpython-38.pyc" not in message


def test_staged_gitlink_is_rejected_as_dirty_parent_not_used_as_revision(tmp_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    fake = FakeGit(parent_status="M  third_party/wearec", gitlink="committed", head="committed")
    with pytest.raises(RuntimeError, match="Parent tracked"):
        resolve_wearec_repository_provenance(parent, wearec, command_runner=fake)


def test_untracked_files_are_excluded_from_provenance_status(tmp_path):
    parent = tmp_path / "parent"; wearec = tmp_path / "wearec"
    parent.mkdir(); wearec.mkdir()
    calls = []
    fake = FakeGit()

    def recording_runner(cmd, **kwargs):
        calls.append(tuple(cmd))
        return fake(cmd, **kwargs)

    result = resolve_wearec_repository_provenance(
        parent, wearec, command_runner=recording_runner
    )
    assert result["wearec_submodule_commit"] == "b"
    assert ("git", "rev-parse", ":third_party/wearec") not in calls
    assert (
        "git",
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--ignore-submodules=dirty",
    ) in calls
    assert calls.count(("git", "status", "--porcelain", "--untracked-files=no")) == 1
