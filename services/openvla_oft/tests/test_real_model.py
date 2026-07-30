from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from openvla_oft.exceptions import ServiceError
from openvla_oft.model import (
    OfficialBindings,
    RealOpenVLAPolicy,
    verify_checkpoint_artifacts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real_config(config, tmp_path: Path) -> dict:
    result = deepcopy(config)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "weights.bin").write_bytes(b"real-test-weights")
    (checkpoint_dir / "dataset_statistics.json").write_text(
        json.dumps({"industrial_arm_b": {"action": {}, "proprio": {}}}),
        encoding="utf-8",
    )
    (checkpoint_dir / "action_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frame": "robot_base",
                "translation_unit": "m",
                "rotation_unit": "rad",
                "rotation_representation": "rotation_vector_axis_angle",
                "gripper_unit": "normalized",
                "gripper_open_threshold": 0.5,
                "action_order": [
                    "dx_m",
                    "dy_m",
                    "dz_m",
                    "dax_rad",
                    "day_rad",
                    "daz_rad",
                    "gripper_norm",
                ],
                "proprio_order": [
                    "x_m",
                    "y_m",
                    "z_m",
                    "roll_rad",
                    "pitch_rad",
                    "yaw_rad",
                    "gripper_norm",
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "files": [
            {
                "path": "weights.bin",
                "sha256": _sha(checkpoint_dir / "weights.bin"),
            },
            {
                "path": "dataset_statistics.json",
                "sha256": _sha(checkpoint_dir / "dataset_statistics.json"),
            },
            {
                "path": "action_contract.json",
                "sha256": _sha(checkpoint_dir / "action_contract.json"),
            },
        ],
    }
    manifest_path = checkpoint_dir / "checkpoint.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    result["mock_mode"] = False
    result["checkpoint_dir"] = str(checkpoint_dir)
    result["artifacts"]["checkpoint_sha"] = f"sha256:{_sha(manifest_path)}"
    result["artifacts"]["norm_stats_sha"] = (
        f"sha256:{_sha(checkpoint_dir / 'dataset_statistics.json')}"
    )
    result["runtime"]["unnorm_key"] = "industrial_arm_b"
    return result


def test_real_policy_verifies_artifacts_and_returns_native_actions(config, tmp_path):
    real_config = _real_config(config, tmp_path)

    class FakeBindings:
        def predict(self, *, image, state, instruction):
            assert image.shape == (720, 1280, 3)
            assert image.dtype == np.uint8
            assert state.shape == (7,)
            assert instruction == real_config["instruction"]
            return np.array(
                [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
                dtype=np.float32,
            )

    policy = RealOpenVLAPolicy(real_config, bindings=FakeBindings())
    actions = policy.predict(
        {
            "model_input": {
                "full_image": np.zeros((720, 1280, 3), dtype=np.uint8),
                "state": [0.0] * 7,
                "task_description": real_config["instruction"],
            }
        }
    )

    assert policy.ready is True
    assert np.asarray(actions).shape == (1, 7)


def test_official_bindings_load_pinned_components(
    config,
    tmp_path,
    monkeypatch,
):
    real_config = _real_config(config, tmp_path)
    upstream_dir = tmp_path / "upstream"
    upstream_dir.mkdir()
    real_config["upstream_dir"] = str(upstream_dir)
    calls = []

    class FakeVLA:
        llm_dim = 64
        norm_stats = {"industrial_arm_b": {"action": {}, "proprio": {}}}

    class FakeUtilities:
        @staticmethod
        def get_vla(cfg):
            calls.append(("vla", cfg.pretrained_checkpoint))
            return FakeVLA()

        @staticmethod
        def get_processor(cfg):
            calls.append(("processor", cfg.pretrained_checkpoint))
            return object()

        @staticmethod
        def get_action_head(cfg, llm_dim):
            calls.append(("action_head", llm_dim))
            return object()

        @staticmethod
        def get_proprio_projector(cfg, llm_dim, proprio_dim):
            calls.append(("proprio", llm_dim, proprio_dim))
            return object()

        @staticmethod
        def get_vla_action(
            cfg,
            vla,
            processor,
            observation,
            instruction,
            action_head,
            proprio_projector,
            use_film,
        ):
            del cfg, vla, processor, action_head, proprio_projector
            assert observation["full_image"].shape == (720, 1280, 3)
            assert instruction == real_config["instruction"]
            assert use_film is False
            return np.zeros((2, 7), dtype=np.float32)

    monkeypatch.setattr(
        "openvla_oft.model._verify_upstream_checkout",
        lambda path, commit: calls.append(("commit", path, commit)),
    )
    monkeypatch.setattr(
        "openvla_oft.model.importlib.import_module",
        lambda name: FakeUtilities,
    )

    bindings = OfficialBindings.load(real_config)
    actions = bindings.predict(
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        state=np.zeros(7, dtype=np.float32),
        instruction=real_config["instruction"],
    )

    assert np.asarray(actions).shape == (2, 7)
    assert ("action_head", 64) in calls
    assert ("proprio", 64, 7) in calls


def test_real_policy_honours_cancellation_before_gpu_call(config, tmp_path):
    real_config = _real_config(config, tmp_path)

    class NeverCalled:
        def predict(self, **kwargs):
            raise AssertionError(kwargs)

    policy = RealOpenVLAPolicy(real_config, bindings=NeverCalled())
    cancel_event = Event()
    cancel_event.set()
    with pytest.raises(ServiceError, match="cancelled before model execution"):
        policy.predict(
            {
                "model_input": {
                    "full_image": np.zeros((720, 1280, 3), dtype=np.uint8),
                    "state": [0.0] * 7,
                    "task_description": real_config["instruction"],
                }
            },
            cancel_event=cancel_event,
        )


def test_artifact_verifier_rejects_tampered_weight(config, tmp_path):
    real_config = _real_config(config, tmp_path)
    Path(real_config["checkpoint_dir"], "weights.bin").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="artifact SHA256 mismatch"):
        verify_checkpoint_artifacts(real_config)


def test_artifact_verifier_rejects_wrong_action_semantics(config, tmp_path):
    real_config = _real_config(config, tmp_path)
    checkpoint_dir = Path(real_config["checkpoint_dir"])
    contract_path = checkpoint_dir / "action_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["gripper_unit"] = "binary"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    manifest_path = checkpoint_dir / "checkpoint.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] == "action_contract.json":
            entry["sha256"] = _sha(contract_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    real_config["artifacts"]["checkpoint_sha"] = f"sha256:{_sha(manifest_path)}"

    with pytest.raises(RuntimeError, match="does not exactly match"):
        verify_checkpoint_artifacts(real_config)


def test_artifact_verifier_rejects_manifest_path_escape(config, tmp_path):
    real_config = _real_config(config, tmp_path)
    manifest_path = Path(
        real_config["checkpoint_dir"],
        real_config["artifacts"]["checkpoint_manifest"],
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "files": [{"path": "../escape.bin", "sha256": "a" * 64}],
            }
        ),
        encoding="utf-8",
    )
    real_config["artifacts"]["checkpoint_sha"] = f"sha256:{_sha(manifest_path)}"

    with pytest.raises(RuntimeError, match="escapes checkpoint_dir"):
        verify_checkpoint_artifacts(real_config)
