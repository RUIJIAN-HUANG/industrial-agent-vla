"""Run the YOLO sidecar against one real synchronized three-camera observation.

The input JSON must contain an online Observation with the frozen camera keys:
`arm_a_rgb`, `handoff_rgb`, and `arm_b_rgb`. Each camera entry must already be
registered in the shared CAS and use the standard ImageReference contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from industrial_agent.contracts import Observation
from simulation.yolo_camera_probe import discover_yolo_http_agent, probe_yolo_cameras


def _load_observation(path: Path) -> Observation:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("observation JSON must contain an object")
    required = {"observation_id", "timestamp_ms"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"observation JSON is missing fields: {sorted(missing)}")
    data: dict[str, Any] = {
        key: value
        for key, value in raw.items()
        if key not in {"observation_version", "observation_id", "timestamp_ms"}
    }
    return Observation(
        observation_id=str(raw["observation_id"]),
        timestamp_ms=int(raw["timestamp_ms"]),
        data=data,
        observation_version=str(raw.get("observation_version", "1.0")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run YOLO on a real three-camera Observation."
    )
    parser.add_argument("--observation-json", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8103")
    parser.add_argument("--run-id", default="manual800-three-camera-probe")
    parser.add_argument("--task-id", default="P01_TO_S11")
    parser.add_argument("--subtask-id", default="P01_TO_S11")
    parser.add_argument("--step-id", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--evidence-jsonl", required=True)
    args = parser.parse_args()

    observation = _load_observation(Path(args.observation_json))
    perception, health = discover_yolo_http_agent(
        args.base_url,
        timeout_ms=args.timeout_ms,
        allow_mock=False,
    )
    summary = probe_yolo_cameras(
        observation,
        perception,
        run_id=args.run_id,
        task_id=args.task_id,
        subtask_id=args.subtask_id,
        step_id=args.step_id,
        timeout_ms=args.timeout_ms,
        allowed_class_names=(),
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        evidence_jsonl_path=Path(args.evidence_jsonl),
    )
    output = {
        "health": health,
        "summary": summary,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
