from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np
from industrial_agent.contracts import Observation
from industrial_agent.image_cas import ImageCas
from simulation.yolo_camera_probe import (
    discover_yolo_http_agent,
    probe_yolo_cameras,
)

from yolo_service.app import YoloHTTPServer
from yolo_service.routes import YoloService


def _three_camera_observation(config: dict[str, Any]) -> Observation:
    cas = ImageCas.from_agent_config(config)
    camera: dict[str, Any] = {}
    streams = (
        ("arm_a_rgb", "CAM_A_TOP", (255, 0, 0)),
        ("handoff_rgb", "CAM_HANDOFF", (0, 255, 0)),
        ("arm_b_rgb", "CAM_B_TOP", (0, 0, 255)),
    )
    for stream_name, camera_id, color in streams:
        rgb = np.empty((720, 1280, 3), dtype=np.uint8)
        rgb[:] = color
        camera[stream_name] = cas.write_rgb(rgb, camera_id=camera_id).to_dict()
    return Observation(
        observation_id="http-camera-observation",
        timestamp_ms=123,
        data={"camera": camera},
    )


def test_mock_http_service_accepts_one_synchronized_camera_triplet(
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    service = YoloService(config)
    server = YoloHTTPServer(("127.0.0.1", 0), service)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence_path = tmp_path / "yolo_camera_probes.jsonl"
    try:
        host, port = server.server_address
        agent, health = discover_yolo_http_agent(
            f"http://{host}:{port}",
            allow_mock=True,
        )

        summary = probe_yolo_cameras(
            _three_camera_observation(config),
            agent,
            run_id="http-smoke-run",
            task_id="http-smoke-task",
            subtask_id="S01_ARM_A_PACK_HANDOFF",
            evidence_jsonl_path=evidence_path,
        )
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=5)

    assert health["device"]["mode"] == "mock"
    assert summary["camera_order"] == [
        "CAM_A_TOP",
        "CAM_HANDOFF",
        "CAM_B_TOP",
    ]
    assert [item["detection_count"] for item in summary["results"]] == [0, 0, 0]
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert persisted == summary
