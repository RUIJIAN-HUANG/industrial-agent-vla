from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from industrial_agent.image_cas import ImageCas

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from openvla_oft.config import load_config  # noqa: E402
from openvla_oft.routes import OpenVLAOFTService  # noqa: E402


@pytest.fixture
def cas_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cas"
    monkeypatch.setenv("INDUSTRIAL_AGENT_CAS_ROOT", str(root))
    monkeypatch.setenv("OPENVLA_OFT_USE_MOCK", "1")
    return root


@pytest.fixture
def config(cas_root: Path) -> dict[str, Any]:
    return load_config(SERVICE_ROOT / "configs")


@pytest.fixture
def service(config: dict[str, Any]) -> OpenVLAOFTService:
    return OpenVLAOFTService(config)


@pytest.fixture
def valid_infer_request(config: dict[str, Any]) -> dict[str, Any]:
    cas = ImageCas.from_agent_config(config)
    full_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    full_rgb[..., 1] = 64
    checkpoint_sha = config["artifacts"]["checkpoint_sha"]
    norm_stats_sha = config["artifacts"]["norm_stats_sha"]
    full_ref = cas.write_rgb(full_rgb, camera_id="CAM_B_TOP").to_dict()
    return {
        "schema_version": "1.0",
        "request_id": "req-1",
        "trace_id": "run-1",
        "episode_id": "run-1",
        "task_id": "episode-0001:S02_ARM_B_TRANSPORT",
        "subtask_id": "S02_ARM_B_TRANSPORT",
        "step_id": 0,
        "observation_id": "obs-1",
        "deadline_ms": 5000,
        "executor": "openvla_oft",
        "checkpoint_sha": checkpoint_sha,
        "norm_stats_sha": norm_stats_sha,
        "expected_action_contract": "1.0",
        "model_input": {
            "task_description": config["instruction"],
            "full_image": full_ref,
            "wrist_image": None,
            "state": [0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
        },
    }


@pytest.fixture
def copy_request() -> Any:
    def _copy(value: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(value)

    return _copy
