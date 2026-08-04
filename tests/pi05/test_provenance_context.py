from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

import scripts.pi05.compute_norm_stats as norm_module
import scripts.pi05.convert_openpi as convert_module
import scripts.pi05.smoke_lerobot_loader as loader_module
from scripts.pi05.provenance_context import (
    ProvenanceContext,
    resolve_provenance_context,
    validate_provenance_context,
)


OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def _context() -> ProvenanceContext:
    return ProvenanceContext(
        project_git_sha="1" * 40,
        project_worktree_dirty=True,
        project_worktree_diff_sha256="2" * 64,
        openpi_commit=OPENPI_COMMIT,
    )


def test_validate_context_round_trips_exact_manifest() -> None:
    context = _context()
    assert validate_provenance_context(
        context.as_manifest(),
        expected=context,
    ) == context


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_git_sha", ""),
        ("project_git_sha", "g" * 40),
        ("project_git_sha", "1" * 39),
        ("openpi_commit", "unknown"),
        ("openpi_commit", "A" * 40),
        ("project_worktree_diff_sha256", "2" * 63),
    ],
)
def test_context_rejects_missing_or_malformed_identity(field: str, value: str) -> None:
    payload: dict[str, Any] = {
        "project_git_sha": "1" * 40,
        "project_worktree_dirty": False,
        "project_worktree_diff_sha256": "2" * 64,
        "openpi_commit": OPENPI_COMMIT,
    }
    payload[field] = value
    with pytest.raises((TypeError, ValueError)):
        ProvenanceContext(**payload)


def test_validate_context_rejects_missing_and_forged_fields() -> None:
    context = _context()
    missing = context.as_manifest()
    missing.pop("openpi_commit")
    with pytest.raises(ValueError, match="incomplete or unknown"):
        validate_provenance_context(missing)

    forged = context.as_manifest()
    forged["project_git_sha"] = "3" * 40
    with pytest.raises(ValueError, match="does not match"):
        validate_provenance_context(forged, expected=context)


def test_resolve_context_records_head_dirty_state_and_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_diff = b"diff --git a/x b/x\n"
    untracked = tmp_path / "scripts" / "pi05" / "new_source.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("VALUE = 1\n", encoding="utf-8")

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        arguments = command[1:]
        if arguments == ["rev-parse", "--show-toplevel"]:
            stdout = str(tmp_path).encode("utf-8") + b"\n"
        elif arguments == ["rev-parse", "HEAD"]:
            stdout = b"1" * 40 + b"\n"
        elif arguments[:2] == ["status", "--porcelain=v1"]:
            stdout = b" M scripts/pi05/example.py\n"
        elif arguments[:2] == ["diff", "--binary"]:
            stdout = tracked_diff
        elif arguments[:2] == ["ls-files", "--others"]:
            stdout = b"scripts/pi05/new_source.py\0"
        else:
            raise AssertionError(f"unexpected Git command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    context = resolve_provenance_context(
        repo_root=tmp_path,
        openpi_commit=OPENPI_COMMIT,
        expected_openpi_commit=OPENPI_COMMIT,
        timeout_s=1.0,
    )
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    digest.update(b"untracked-files\0")
    digest.update(b"scripts/pi05/new_source.py\0")
    digest.update(hashlib.sha256(untracked.read_bytes()).hexdigest().encode("ascii"))
    digest.update(b"\0")
    assert context.project_git_sha == "1" * 40
    assert context.project_worktree_dirty is True
    assert context.project_worktree_diff_sha256 == digest.hexdigest()


def test_resolve_context_rejects_unpinned_openpi_before_git(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        resolve_provenance_context(
            repo_root=tmp_path,
            openpi_commit="3" * 40,
            expected_openpi_commit=OPENPI_COMMIT,
        )


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError("git"), "executable is required"),
        (
            subprocess.TimeoutExpired(["git", "rev-parse"], timeout=1.0),
            "timed out",
        ),
        (
            subprocess.CalledProcessError(
                2,
                ["git", "rev-parse"],
                stderr=b"failure",
            ),
            "query failed",
        ),
    ],
)
def test_resolve_context_fails_closed_for_git_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    def fail_run(*_: Any, **__: Any) -> Any:
        raise error

    monkeypatch.setattr(subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match=message):
        resolve_provenance_context(
            repo_root=tmp_path,
            openpi_commit=OPENPI_COMMIT,
            expected_openpi_commit=OPENPI_COMMIT,
            timeout_s=1.0,
        )


def test_converter_cli_returns_nonzero_when_git_provenance_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = type(
        "Args",
        (),
        {
            "push_to_hub": False,
            "state_mapper": "mapper",
            "split_registry": "registry.json",
            "project_root": str(tmp_path),
            "openpi_commit": OPENPI_COMMIT,
        },
    )()
    monkeypatch.setattr(convert_module, "parse_args", lambda: args)
    monkeypatch.setattr(convert_module, "load_state_mapper", lambda *_a, **_k: object())
    monkeypatch.setattr(convert_module, "load_split_registry", lambda *_a, **_k: object())
    monkeypatch.setattr(
        convert_module,
        "resolve_provenance_context",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Git failed")),
    )
    assert convert_module.main() == 1


def test_norm_stats_cli_returns_nonzero_when_git_provenance_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = type(
        "Args",
        (),
        {
            "quiet": False,
            "state_mapper": "mapper",
            "split_registry": "registry.json",
            "project_root": str(tmp_path),
            "openpi_commit": OPENPI_COMMIT,
        },
    )()
    monkeypatch.setattr(norm_module, "parse_args", lambda: args)
    monkeypatch.setattr(norm_module, "OPENPI_NORMALIZE_AVAILABLE", True)
    monkeypatch.setattr(norm_module, "_normalize", object())
    monkeypatch.setattr(norm_module, "load_state_mapper", lambda *_a, **_k: object())
    monkeypatch.setattr(norm_module, "load_split_registry", lambda *_a, **_k: object())
    monkeypatch.setattr(
        norm_module,
        "resolve_provenance_context",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Git failed")),
    )
    assert norm_module.main() == 1


def test_loader_cli_returns_nonzero_when_git_provenance_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = type(
        "Args",
        (),
        {
            "dataset_root": str(tmp_path / "dataset"),
            "manifest": None,
            "repo_id": "test/pi05",
            "project_root": str(tmp_path),
            "openpi_commit": OPENPI_COMMIT,
        },
    )()
    monkeypatch.setattr(loader_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        loader_module,
        "resolve_provenance_context",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Git failed")),
    )
    assert loader_module.main() == 1
