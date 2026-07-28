# OpenVLA-OFT Standalone Service

This directory contains the frozen Arm_B executor for OpenVLA-OFT.

Frozen role:

- only execute `Arm_B`
- only handle `S02_ARM_B_TRANSPORT`
- only run after Supervisor has completed the three-frame handoff check and granted `B_ONLY`
- accept the frozen downstream instruction, `CAM_B_TOP`, `wrist_image=null`, and Arm_B state
- emit canonical `N x 7` action chunks for `HANDOFF_CENTER -> FINISHED_01 -> HOME_B`
- never require YOLO detection packets as a prerequisite
- never consume GT, target coordinates, trajectory points, or hidden pose labels

What is implemented here:

- stdlib HTTP entrypoint and service boundary
- request/response contract validation
- repository-wide shared-CAS resolution through
  `industrial_agent.service_images.CasRequestImageResolver`
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
- requires `model_input.wrist_image=null`; the frozen scene has no wrist camera
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

The CAS root is configurable through `INDUSTRIAL_AGENT_CAS_ROOT`. The service
uses the same `industrial_agent.image_cas.ImageCas` implementation and error
taxonomy as π0.5 and YOLO; it does not maintain a private resolver.

## Run

Install the repository package first because it owns the shared contracts and
CAS resolver:

```powershell
python -m pip install -e ".[test]"
python -m pip install -e "services/openvla_oft[test]"
```

Real mode is the fail-closed default and requires non-zero pinned artifact
digests. Mock mode is test-only and must be enabled explicitly:

```powershell
$env:OPENVLA_OFT_USE_MOCK = "1"
python services/openvla_oft/scripts/run_service.py --host 127.0.0.1 --port 8102
```

For a multi-container deployment, bind `--host 0.0.0.0`; the final Compose file
must mount the shared CAS volume read-only into this service.

## Test

```powershell
python -m pytest services/openvla_oft/tests -q
```
