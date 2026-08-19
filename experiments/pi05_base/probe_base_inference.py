"""Run one isolated π0.5 base compatibility probe.

Mock mode is for Windows plumbing tests only.  Real mode requires official
OpenPI and real exterior/wrist images, and still does not execute Isaac actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
from PIL import Image

from .base_client import BasePolicyClient, OpenPiBaseClient
from .contracts import ExperimentConfig, ExperimentObservation, PolicyOutput
from .mock_client import MockBaseClient
from .validate_actions import inspect_actions


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_ROOT / "configs" / "base_probe.json"
DEFAULT_REPORT = MODULE_ROOT / "artifacts" / "base-inference-report.json"


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"RGB image not found: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _mock_rgb() -> np.ndarray:
    rows, cols = np.indices((224, 224))
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[:, :, 0] = ((cols // 16) % 2 * 255).astype(np.uint8)
    image[:, :, 1] = ((rows // 16) % 2 * 255).astype(np.uint8)
    image[:, :, 2] = 64
    return image


def _parse_joints(raw: str) -> np.ndarray:
    try:
        values = [float(item.strip()) for item in raw.split(",")]
    except ValueError as exc:
        raise ValueError("--joint-state must be seven comma-separated numbers") from exc
    if len(values) != 7:
        raise ValueError("--joint-state must contain exactly seven values")
    return np.asarray(values, dtype=np.float32)


def build_observation(
    *,
    mode: str,
    prompt: str,
    image_path: Path | None,
    wrist_image_path: Path | None,
    joint_state: str,
    gripper_position: float,
) -> ExperimentObservation:
    if mode == "real" and image_path is None:
        raise ValueError("real mode requires --image from the current Isaac scene")
    if mode == "real" and wrist_image_path is None:
        raise ValueError(
            "real droid compatibility mode requires --wrist-image; "
            "the runner will not fabricate a camera frame"
        )
    front = _load_rgb(image_path) if image_path is not None else _mock_rgb()
    wrist = _load_rgb(wrist_image_path) if wrist_image_path is not None else None
    return ExperimentObservation(
        observation_id=f"pi05-base-{uuid4()}",
        timestamp_ns=time.time_ns(),
        front_rgb=front,
        wrist_rgb=wrist,
        joint_position=_parse_joints(joint_state),
        gripper_position=gripper_position,
        prompt=prompt,
    )


def build_report(
    *,
    config: ExperimentConfig,
    observation: ExperimentObservation,
    output: PolicyOutput,
) -> dict[str, Any]:
    inspection = inspect_actions(output.actions)
    real_evidence = output.policy_mode == "real"
    return {
        "schema_version": "1.0",
        "experiment_id": config.experiment_id,
        "record_type": "pi05_base_single_inference_probe",
        "created_at_ns": time.time_ns(),
        "policy": {
            "mode": output.policy_mode,
            "checkpoint_reference": output.checkpoint_reference,
            "openpi_config_name": config.openpi_config_name,
            "input_profile": config.input_profile,
            "latency_ms": output.latency_ms,
            "metadata": dict(output.metadata),
        },
        "input": {
            "observation_id": observation.observation_id,
            "timestamp_ns": observation.timestamp_ns,
            "front_rgb_shape": list(observation.front_rgb.shape),
            "front_rgb_dtype": str(observation.front_rgb.dtype),
            "wrist_rgb_present": observation.wrist_rgb is not None,
            "wrist_rgb_shape": (
                list(observation.wrist_rgb.shape)
                if observation.wrist_rgb is not None
                else None
            ),
            "joint_state_dim": int(observation.joint_position.size),
            "gripper_position": observation.gripper_position,
            "prompt": observation.prompt,
            "ground_truth_included": False,
            "agent_context_included": False,
        },
        "output": inspection.to_dict(),
        "evidence": {
            "valid_model_inference_evidence": real_evidence,
            "valid_closed_loop_evidence": False,
            "weights_modified": False,
            "training_invoked": False,
            "agent_invoked": False,
            "action_semantics_confirmed": False,
        },
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one isolated π0.5 base-checkpoint compatibility probe."
    )
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--wrist-image", type=Path)
    parser.add_argument(
        "--joint-state",
        default="0,0,0,0,0,0,0",
        help="Seven comma-separated Franka joint values.",
    )
    parser.add_argument("--gripper", type=float, default=1.0)
    parser.add_argument("--prompt")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ExperimentConfig.from_path(args.config)
    observation = build_observation(
        mode=args.mode,
        prompt=args.prompt or config.prompt,
        image_path=args.image,
        wrist_image_path=args.wrist_image,
        joint_state=args.joint_state,
        gripper_position=args.gripper,
    )
    client: BasePolicyClient
    if args.mode == "real":
        client = OpenPiBaseClient(config)
    else:
        client = MockBaseClient(config)
    output = client.infer(observation)
    report = build_report(
        config=config,
        observation=observation,
        output=output,
    )
    write_report(args.report, report)
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "policy_mode": output.policy_mode,
                "action_shape": list(output.actions.shape),
                "valid_model_inference_evidence": report["evidence"][
                    "valid_model_inference_evidence"
                ],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
