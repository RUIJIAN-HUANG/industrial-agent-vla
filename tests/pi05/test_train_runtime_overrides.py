from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ENV_NAMES = (
    "PI05_BATCH_SIZE",
    "PI05_NUM_WORKERS",
    "PI05_NUM_TRAIN_STEPS",
    "PI05_LOG_INTERVAL",
    "PI05_SAVE_INTERVAL",
    "PI05_KEEP_PERIOD",
)


def _load_runtime_config(overrides: dict[str, str]) -> tuple[dict[str, int], str]:
    """Import the config in an isolated process and return its runtime fields."""

    environment = os.environ.copy()
    for name in RUNTIME_ENV_NAMES:
        environment.pop(name, None)
    environment.update(overrides)
    code = """
import json
from configs.pi05.train_config import PI05_INDUSTRIAL_CONFIG

config = PI05_INDUSTRIAL_CONFIG
print(json.dumps({
    "batch_size": config.batch_size,
    "num_workers": config.num_workers,
    "num_train_steps": config.num_train_steps,
    "log_interval": config.log_interval,
    "save_interval": config.save_interval,
    "keep_period": config.keep_period,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout.strip().splitlines()[-1]), result.stderr


def test_runtime_training_defaults_remain_unchanged() -> None:
    values, _ = _load_runtime_config({})

    assert values == {
        "batch_size": 16,
        "num_workers": 2,
        "num_train_steps": 30_000,
        "log_interval": 100,
        "save_interval": 1_000,
        "keep_period": 5_000,
    }


def test_runtime_training_environment_overrides_reach_openpi_config() -> None:
    values, _ = _load_runtime_config(
        {
            "PI05_BATCH_SIZE": "8",
            "PI05_NUM_WORKERS": "4",
            "PI05_NUM_TRAIN_STEPS": "20",
            "PI05_LOG_INTERVAL": "1",
            "PI05_SAVE_INTERVAL": "10",
            "PI05_KEEP_PERIOD": "10",
        }
    )

    assert values == {
        "batch_size": 8,
        "num_workers": 4,
        "num_train_steps": 20,
        "log_interval": 1,
        "save_interval": 10,
        "keep_period": 10,
    }


def test_invalid_runtime_training_environment_uses_safe_defaults() -> None:
    values, stderr = _load_runtime_config(
        {
            "PI05_BATCH_SIZE": "not-an-integer",
            "PI05_NUM_WORKERS": "-1",
            "PI05_NUM_TRAIN_STEPS": "0",
            "PI05_LOG_INTERVAL": "0",
            "PI05_SAVE_INTERVAL": "-10",
            "PI05_KEEP_PERIOD": "invalid",
        }
    )

    assert values == {
        "batch_size": 16,
        "num_workers": 2,
        "num_train_steps": 30_000,
        "log_interval": 100,
        "save_interval": 1_000,
        "keep_period": 5_000,
    }
    for name in RUNTIME_ENV_NAMES:
        assert name in stderr
