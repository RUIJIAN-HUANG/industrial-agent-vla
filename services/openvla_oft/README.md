# OpenVLA-OFT Standalone Service

This directory contains the frozen Arm_B executor for OpenVLA-OFT.

Frozen role:

- only execute `Arm_B`
- only handle `S02_ARM_B_TRANSPORT`
- only run after Supervisor has completed the three-frame handoff check and granted `B_ONLY`
- accept the frozen downstream instruction, `CAM_B_TOP`, optional `CAM_B_WRIST`, and Arm_B state
- emit canonical `N x 7` action chunks for `HANDOFF_CENTER -> FINISHED_01 -> HOME_B`
- never require YOLO detection packets as a prerequisite
- never consume GT, target coordinates, trajectory points, or hidden pose labels

What is implemented here:

- stdlib HTTP entrypoint
- request/response contract validation
- local content-addressed RGB image resolution
- mock policy for smoke tests
- cancel/timeout handling with cooperative request cancellation
- public config and tests

What is still missing:

- real OpenVLA-OFT checkpoint loading
- industrial fine-tuning evidence
- base/tuned comparison results
- end-to-end Isaac Sim control

## Contract

`GET /health`

- returns `schema_version`, `service`, `status`, `checkpoint_sha`, `norm_stats_sha`,
  `supported_task_types`, and `supported_action_contracts`
- mock mode reports `ready`
- real mode currently reports `degraded`

`POST /v1/infer`

- requires the frozen executor envelope
- requires `executor=openvla_oft`
- requires `subtask_id=S02_ARM_B_TRANSPORT`
- requires `model_input.task_description` to match the frozen instruction
- requires `model_input.full_image.camera_id=CAM_B_TOP`
- requires `model_input.full_image` to be `1280x720`
- accepts `model_input.wrist_image=null`
- accepts `model_input.wrist_image` as `CAM_B_WRIST` when present
- resolves all provided image references from CAS before policy execution

`POST /v1/cancel`

- active request -> `cancelled`
- already completed -> `already_completed`
- unknown task -> `not_found`

## Config

Public config lives in:

- `configs/agent.default.json`
- `configs/openvla.default.json`

Frozen values recorded there include:

- camera order
- image size
- language field
- task id field
- canonical action order
- checkpoint and norm stats SHA placeholders
- CAS root layout

The CAS root is configurable through `INDUSTRIAL_AGENT_CAS_ROOT`.

## Run

```powershell
cd services/openvla_oft
python -m pip install -e ".[test]"
$env:OPENVLA_OFT_USE_MOCK = "1"
python scripts/run_service.py --host 127.0.0.1 --port 8102
```

## Test

```powershell
python -m pytest services/openvla_oft/tests -q
```
