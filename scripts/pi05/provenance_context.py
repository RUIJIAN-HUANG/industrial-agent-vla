"""Fail-closed source identity for PI05 data artifacts."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


logger = logging.getLogger("pi05.provenance_context")

LEROBOT_PROVENANCE_MANIFEST_TYPE = "pi05_lerobot_provenance_v3"
NORM_STATS_SOURCE_MANIFEST_TYPE = "pi05_norm_stats_source_v2"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PATHS = (
    "scripts/pi05",
    "services/pi05",
    "configs/pi05",
    "tests/pi05",
    "docs",
)


@dataclass(frozen=True)
class ProvenanceContext:
    """Identity of the project source tree and pinned OpenPI checkout."""

    project_git_sha: str
    project_worktree_dirty: bool
    project_worktree_diff_sha256: str
    openpi_commit: str

    def __post_init__(self) -> None:
        _require_git_sha(self.project_git_sha, field="project_git_sha")
        if not isinstance(self.project_worktree_dirty, bool):
            raise TypeError("project_worktree_dirty must be a boolean")
        if _SHA256.fullmatch(self.project_worktree_diff_sha256) is None:
            raise ValueError("project_worktree_diff_sha256 must be 64 lowercase hex")
        _require_git_sha(self.openpi_commit, field="openpi_commit")

    def as_manifest(self) -> dict[str, Any]:
        """Return the stable JSON representation used by all PI05 artifacts."""

        return {
            "project_git_sha": self.project_git_sha,
            "project_worktree_dirty": self.project_worktree_dirty,
            "project_worktree_diff_sha256": self.project_worktree_diff_sha256,
            "openpi_commit": self.openpi_commit,
        }


def _require_git_sha(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise ValueError(f"{field} must be 40 or 64 lowercase hexadecimal characters")
    return value


def validate_provenance_context(
    value: Mapping[str, Any],
    *,
    expected: ProvenanceContext | None = None,
) -> ProvenanceContext:
    """Parse and optionally compare a manifest producer identity."""

    if not isinstance(value, Mapping) or set(value) != {
        "project_git_sha",
        "project_worktree_dirty",
        "project_worktree_diff_sha256",
        "openpi_commit",
    }:
        raise ValueError("PI05 provenance producer fields are incomplete or unknown")
    context = ProvenanceContext(
        project_git_sha=value["project_git_sha"],
        project_worktree_dirty=value["project_worktree_dirty"],
        project_worktree_diff_sha256=value["project_worktree_diff_sha256"],
        openpi_commit=value["openpi_commit"],
    )
    if expected is not None and context != expected:
        raise ValueError(
            "PI05 provenance producer does not match the current source tree"
        )
    return context


def _run_git(
    arguments: Sequence[str],
    *,
    repo_root: Path,
    timeout_s: float,
) -> bytes:
    command = ["git", *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        logger.error("Git executable is unavailable while resolving PI05 provenance")
        raise RuntimeError(
            "Git executable is required for production provenance"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        logger.error("Git provenance query timed out: command=%s", command)
        raise RuntimeError("Git provenance query timed out") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        logger.error(
            "Git provenance query failed: command=%s returncode=%s stderr=%s",
            command,
            exc.returncode,
            stderr,
        )
        raise RuntimeError("Git provenance query failed") from exc
    return completed.stdout


def _worktree_fingerprint(repo_root: Path, *, timeout_s: float) -> tuple[bool, str]:
    status = _run_git(
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *_SOURCE_PATHS,
        ),
        repo_root=repo_root,
        timeout_s=timeout_s,
    )
    tracked_diff = _run_git(
        ("diff", "--binary", "--no-ext-diff", "HEAD", "--", *_SOURCE_PATHS),
        repo_root=repo_root,
        timeout_s=timeout_s,
    )
    untracked_raw = _run_git(
        (
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *_SOURCE_PATHS,
        ),
        repo_root=repo_root,
        timeout_s=timeout_s,
    )
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    digest.update(b"untracked-files\0")
    for relative_bytes in sorted(item for item in untracked_raw.split(b"\0") if item):
        try:
            relative = relative_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("Git returned a non-UTF-8 untracked path") from exc
        candidate = (repo_root / relative).resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError(
                "Git returned an untracked path outside the repository"
            ) from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(
                f"untracked provenance input is not a regular file: {relative}"
            )
        try:
            content_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError as exc:
            logger.error("Cannot read untracked provenance input: path=%s", candidate)
            raise RuntimeError("cannot fingerprint the project worktree") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_sha.encode("ascii"))
        digest.update(b"\0")
    return bool(status.strip()), digest.hexdigest()


def resolve_provenance_context(
    *,
    repo_root: str | Path,
    openpi_commit: str,
    expected_openpi_commit: str,
    timeout_s: float = 5.0,
) -> ProvenanceContext:
    """Resolve a reproducible source identity without mutating Git state."""

    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout_s must be a positive number")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    supplied_commit = _require_git_sha(openpi_commit, field="openpi_commit")
    frozen_commit = _require_git_sha(
        expected_openpi_commit,
        field="expected_openpi_commit",
    )
    if supplied_commit != frozen_commit:
        raise ValueError(
            "supplied OpenPI Commit does not match the frozen project Commit"
        )

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project repository root does not exist: {root}")
    discovered_root = Path(
        _run_git(
            ("rev-parse", "--show-toplevel"),
            repo_root=root,
            timeout_s=float(timeout_s),
        )
        .decode("utf-8")
        .strip()
    ).resolve()
    if discovered_root != root:
        raise ValueError(
            f"project repository root mismatch: supplied={root} actual={discovered_root}"
        )
    project_git_sha = _require_git_sha(
        _run_git(
            ("rev-parse", "HEAD"),
            repo_root=root,
            timeout_s=float(timeout_s),
        )
        .decode("ascii")
        .strip(),
        field="project_git_sha",
    )
    dirty, diff_sha = _worktree_fingerprint(root, timeout_s=float(timeout_s))
    return ProvenanceContext(
        project_git_sha=project_git_sha,
        project_worktree_dirty=dirty,
        project_worktree_diff_sha256=diff_sha,
        openpi_commit=supplied_commit,
    )
