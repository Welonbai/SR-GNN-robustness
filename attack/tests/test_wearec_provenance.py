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
        elif args == ["status", "--porcelain", "--untracked-files=no"]:
            value = self.parent_status if str(cwd).endswith("parent") else self.wearec_status
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
    assert calls.count(("git", "status", "--porcelain", "--untracked-files=no")) == 2
