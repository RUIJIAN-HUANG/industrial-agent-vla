from __future__ import annotations

import hashlib
from pathlib import Path

from deploy import preflight


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_env_pins_packaged_yolo_class_map() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment = preflight.load_env_file(
        repo_root / "deploy" / ".env.production.example"
    )
    class_map = (
        repo_root
        / "services"
        / "yolo"
        / "src"
        / "yolo_service"
        / "resources"
        / "class_map.single_bin_v2.json"
    )

    assert environment["YOLO_CLASS_MAP_SHA"] == _sha256_file(class_map)


def _valid_environment(tmp_path: Path) -> dict[str, str]:
    cas = tmp_path / "cas"
    pi05_cache = tmp_path / "pi05-cache"
    yolo_cache = tmp_path / "yolo-cache"
    for directory in (cas, pi05_cache, yolo_cache):
        directory.mkdir()

    pi05_checkpoint = tmp_path / "pi05-checkpoint"
    pi05_checkpoint.mkdir()
    (pi05_checkpoint / "params.bin").write_bytes(b"pi05-params")
    pi05_norm = tmp_path / "pi05-norm.json"
    pi05_norm.write_text('{"mean":[0]}', encoding="utf-8")

    yolo_model = tmp_path / "best.pt"
    yolo_model.write_bytes(b"yolo-weights")
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64
    return {
        "PI05_IMAGE_REPOSITORY": "registry.example.com/pi05:0.1.0",
        "PI05_IMAGE_DIGEST": digest_a,
        "YOLO_IMAGE_REPOSITORY": "registry.example.com/yolo:0.1.0",
        "YOLO_IMAGE_DIGEST": digest_b,
        "PI05_PORT": "8101",
        "YOLO_PORT": "8103",
        "PI05_GPU_ID": "0",
        "YOLO_GPU_ID": "1",
        "SHARED_CAS_DIR": str(cas),
        "PI05_CACHE_DIR_HOST": str(pi05_cache),
        "YOLO_CACHE_DIR_HOST": str(yolo_cache),
        "PI05_CHECKPOINT_DIR_HOST": str(pi05_checkpoint),
        "PI05_NORM_STATS_FILE_HOST": str(pi05_norm),
        "PI05_CHECKPOINT_SHA": preflight._pi05_directory_digest(pi05_checkpoint),
        "PI05_NORM_STATS_SHA": _sha256_file(pi05_norm),
        "YOLO_MODEL_FILE_HOST": str(yolo_model),
        "YOLO_CHECKPOINT_SHA": _sha256_file(yolo_model),
        "YOLO_CLASS_MAP_SHA": digest_a,
        "YOLO_CONFIG_SHA": digest_b,
    }


def test_asset_preflight_accepts_exact_release_assets(tmp_path: Path) -> None:
    report = preflight.validate_environment(_valid_environment(tmp_path))

    assert report["status"] == "PASS"
    assert report["warnings"] == []
    assert all(item["status"] == "PASS" for item in report["checks"])


def test_asset_preflight_rejects_placeholder_identity(tmp_path: Path) -> None:
    environment = _valid_environment(tmp_path)
    environment["PI05_IMAGE_DIGEST"] = "REPLACE_WITH_PI05_IMAGE_SHA256"

    report = preflight.validate_environment(environment)

    assert report["status"] == "FAIL"
    failed = {
        item["name"]: item["detail"]
        for item in report["checks"]
        if item["status"] == "FAIL"
    }
    assert "placeholder" in failed["image:PI05_IMAGE_REPOSITORY"]


def test_service_preflight_rejects_mock_health(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment = _valid_environment(tmp_path)

    def fake_health(url: str, timeout_s: float):
        del timeout_s
        if url.endswith(":8101/health"):
            return {
                "service": "pi05",
                "status": "ready",
                "checkpoint_sha": environment["PI05_CHECKPOINT_SHA"],
                "norm_stats_sha": environment["PI05_NORM_STATS_SHA"],
                "device": {"mode": "mock"},
            }
        return {
            "service": "yolo",
            "status": "ready",
            "checkpoint_sha": environment["YOLO_CHECKPOINT_SHA"],
            "class_map_sha": environment["YOLO_CLASS_MAP_SHA"],
            "config_sha": environment["YOLO_CONFIG_SHA"],
            "device": {"mode": "real"},
        }

    monkeypatch.setattr(preflight, "_health_payload", fake_health)

    report = preflight.validate_services(environment)

    assert report["status"] == "FAIL"
    failed = [item for item in report["checks"] if item["status"] == "FAIL"]
    assert failed == [
        {
            "name": "health:pi05",
            "status": "FAIL",
            "detail": "pi05 health reports non-real mode",
        }
    ]
