from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from industrial_agent.image_cas import ImageCas

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from yolo_service.config import load_config  # noqa: E402
from yolo_service.routes import YoloService  # noqa: E402


@pytest.fixture
def schema_store() -> dict[str, Any]:
    schema_root = SERVICE_ROOT.parents[1] / "schemas"
    names = (
        "perception-health.schema.json",
        "perception-detect.schema.json",
        "detection-packet.schema.json",
    )
    return {
        name: json.loads((schema_root / name).read_text(encoding="utf-8"))
        for name in names
    }


@pytest.fixture
def config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    monkeypatch.setenv("INDUSTRIAL_AGENT_CAS_ROOT", str(tmp_path / "cas"))
    monkeypatch.setenv("YOLO_USE_MOCK", "1")
    return load_config()


@pytest.fixture
def service(config: dict[str, Any]) -> YoloService:
    return YoloService(config)


@pytest.fixture
def valid_detect_request(config: dict[str, Any]) -> dict[str, Any]:
    cas = ImageCas.from_agent_config(config)
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    rgb[..., 1] = 64
    image = cas.write_rgb(rgb, camera_id="CAM_HANDOFF").to_dict()
    return {
        "schema_version": "1.0",
        "request_id": "req-1",
        "trace_id": "run-1",
        "episode_id": "run-1",
        "task_id": "task-1",
        "subtask_id": "S01_HANDOFF",
        "step_id": 0,
        "observation_id": "obs-1",
        "image_sha256": image["image_sha256"],
        "deadline_ms": 5000,
        "detector": "yolo",
        "checkpoint_sha": config["checkpoint_sha"],
        "class_map_sha": config["class_map_sha"],
        "config_sha": config["config_sha"],
        "expected_detection_contract": "1.0",
        "image": image,
        "allowed_class_names": [],
        "thresholds": {"confidence": 0.25, "iou": 0.45},
    }
