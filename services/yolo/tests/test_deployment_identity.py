from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from yolo_service.config import _effective_config_sha, load_config
from yolo_service.model import UltralyticsYoloModel


def _sha256(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def test_manual800_runtime_identity_matches_perception_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    service_config = json.loads(
        (repo_root / "configs" / "yolo.service-manual800.json").read_text(
            encoding="utf-8"
        )
    )
    perception_config = json.loads(
        (repo_root / "configs" / "perception.yolo-manual800.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"manual800-contract-probe")
    monkeypatch.setenv("YOLO_USE_MOCK", "0")
    monkeypatch.setenv("YOLO_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.delenv("YOLO_CHECKPOINT_SHA", raising=False)
    monkeypatch.setenv("YOLO_DEVICE", "cpu")

    runtime_config = load_config()
    expected = perception_config["perception"]["config_sha"]

    assert _effective_config_sha(service_config) == expected
    assert runtime_config["config_sha"] == expected


def test_real_config_hashes_checkpoint_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    raw = b"frozen-yolo-weights"
    checkpoint.write_bytes(raw)
    monkeypatch.setenv("YOLO_USE_MOCK", "0")
    monkeypatch.setenv("YOLO_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.delenv("YOLO_CHECKPOINT_SHA", raising=False)

    config = load_config()

    assert config["checkpoint_sha"] == _sha256(raw)
    assert config["checkpoint_path"] == str(checkpoint.resolve())
    assert config["mock_mode"] is False


def test_real_config_rejects_checkpoint_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"wrong-weights")
    monkeypatch.setenv("YOLO_USE_MOCK", "0")
    monkeypatch.setenv("YOLO_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("YOLO_CHECKPOINT_SHA", "sha256:" + "1" * 64)

    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        load_config()


def test_config_rejects_class_map_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YOLO_USE_MOCK", "1")
    monkeypatch.setenv("YOLO_CLASS_MAP_SHA", "sha256:" + "2" * 64)

    with pytest.raises(ValueError, match="class map SHA256 mismatch"):
        load_config()


class _FakeYolo:
    instances: list["_FakeYolo"] = []
    class_names: tuple[str, ...] = ()

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.names = dict(enumerate(self.class_names))
        self.predict_kwargs: dict[str, Any] | None = None
        self.instances.append(self)

    def predict(self, **kwargs: Any) -> list[Any]:
        self.predict_kwargs = kwargs
        return [
            SimpleNamespace(
                speed={"postprocess": 1.25},
                names=self.names,
                boxes=[],
            )
        ]


def test_ultralytics_boundary_converts_rgb_to_bgr_and_filters_classes(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeYolo.instances.clear()
    monkeypatch.setattr(_FakeYolo, "class_names", tuple(config["class_names"]))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYolo))
    real_config = dict(config)
    real_config["mock_mode"] = False
    real_config["checkpoint_path"] = "best.pt"
    model = UltralyticsYoloModel(real_config)
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)

    detections = model.detect(
        rgb,
        allowed_class_names=("bin_box",),
        confidence=0.3,
        iou=0.4,
    )

    assert detections.detections == ()
    assert detections.nms_ms == 1.25
    kwargs = _FakeYolo.instances[0].predict_kwargs
    assert kwargs is not None
    assert kwargs["source"].tolist() == [[[30, 20, 10]]]
    assert kwargs["source"].flags.c_contiguous
    assert kwargs["classes"] == [tuple(config["class_names"]).index("bin_box")]


def test_ultralytics_boundary_rejects_checkpoint_class_map(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_FakeYolo, "class_names", tuple(config["class_names"]))

    class WrongMapYolo(_FakeYolo):
        def __init__(self, checkpoint_path: str) -> None:
            super().__init__(checkpoint_path)
            self.names[0] = "unexpected"

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=WrongMapYolo),
    )
    real_config = dict(config)
    real_config["mock_mode"] = False
    real_config["checkpoint_path"] = "best.pt"

    with pytest.raises(RuntimeError, match="class map does not match"):
        UltralyticsYoloModel(real_config)


def test_ultralytics_boundary_rejects_missing_postprocess_timing(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_FakeYolo, "class_names", tuple(config["class_names"]))

    class MissingTimingYolo(_FakeYolo):
        def predict(self, **kwargs: Any) -> list[Any]:
            self.predict_kwargs = kwargs
            return [SimpleNamespace(speed={}, names=self.names, boxes=[])]

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=MissingTimingYolo),
    )
    real_config = dict(config)
    real_config["mock_mode"] = False
    real_config["checkpoint_path"] = "best.pt"
    model = UltralyticsYoloModel(real_config)

    with pytest.raises(RuntimeError, match="invalid postprocess timing"):
        model.detect(
            np.zeros((1, 1, 3), dtype=np.uint8),
            allowed_class_names=(),
            confidence=0.25,
            iou=0.45,
        )
