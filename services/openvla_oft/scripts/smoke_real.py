"""Run one real OpenVLA-OFT service inference against a local RGB frame."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from industrial_agent.image_cas import ImageCas
from PIL import Image

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from openvla_oft.config import load_config  # noqa: E402
from openvla_oft.routes import OpenVLAOFTService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--state",
        default="[0,0,0,0,0,0,0]",
        help="Seven-element Arm_B state JSON.",
    )
    args = parser.parse_args()
    state = json.loads(args.state)
    if not isinstance(state, list) or len(state) != 7:
        parser.error("--state must be a seven-element JSON list")

    config = load_config(SERVICE_ROOT / "configs")
    if config["mock_mode"]:
        parser.error("OPENVLA_OFT_USE_MOCK must be false for this smoke test")
    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    expected_width, expected_height = config["image_size"]
    if image.shape != (expected_height, expected_width, 3):
        parser.error(
            f"--image must be {expected_width}x{expected_height} RGB; got {image.shape}"
        )

    cas = ImageCas.from_agent_config(config)
    image_ref = cas.write_rgb(image, camera_id="CAM_B_TOP").to_dict()
    request_id = f"real-smoke-{uuid.uuid4()}"
    request = {
        "schema_version": "1.0",
        "request_id": request_id,
        "trace_id": request_id,
        "episode_id": request_id,
        "task_id": f"{request_id}:S02_ARM_B_TRANSPORT",
        "subtask_id": "S02_ARM_B_TRANSPORT",
        "step_id": 0,
        "observation_id": f"{request_id}:obs",
        "deadline_ms": int(config["api"]["max_deadline_ms"]),
        "executor": "openvla_oft",
        "checkpoint_sha": config["artifacts"]["checkpoint_sha"],
        "norm_stats_sha": config["artifacts"]["norm_stats_sha"],
        "expected_action_contract": "1.0",
        "model_input": {
            "task_description": config["instruction"],
            "full_image": image_ref,
            "wrist_image": None,
            "state": state,
        },
    }
    status, response = OpenVLAOFTService(config).infer(request)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if status == 200 and response.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
