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
- real adapter around the official OpenVLA-OFT `get_vla`, `get_processor`,
  `get_action_head`, `get_proprio_projector`, and `get_vla_action` APIs
- official upstream pinned to commit
  `e4287e94541f459edc4feabc4e181f537cd569a8`
- fail-closed checkpoint manifest, per-file SHA256, and norm-stats verification
- fail-closed `action_contract.json` verification for frame, units, action
  order, proprio order, and gripper convention
- one-image/no-wrist, seven-dimensional proprio and continuous L1 action-head mapping
- mock policy for smoke tests
- cancel/timeout handling with cooperative request cancellation
- GPU Docker/Compose deployment, real inference smoke script, public config and tests
- Canonical HDF5 Arm_B loader and dependency-light RLDS-style export smoke path

Evidence that must come from the team-specific trained artifact:

- industrial fine-tuning evidence
- base/tuned comparison results
- end-to-end Isaac Sim control

## Contract

`GET /health`

- returns `schema_version`, `service`, `status`, `checkpoint_sha`, `norm_stats_sha`,
  `supported_task_types`, and `supported_action_contracts`
- mock mode reports `ready`
- real mode starts only after the pinned checkout, manifest, all checkpoint files,
  norm stats, and `unnorm_key` pass validation; a running instance reports `ready`

`POST /v1/infer`

- requires the frozen executor envelope
- requires `executor=openvla_oft`
- requires `subtask_id=S02_ARM_B_TRANSPORT`
- requires `model_input.task_description` to match the frozen instruction
- requires `model_input.full_image.camera_id=CAM_B_TOP`
- requires `model_input.full_image` to be `1280x720`
- requires `model_input.wrist_image=null`; the frozen scene has no wrist camera
- resolves all provided image references from CAS before policy execution
- emits robot-base rotation-vector deltas; at the Franka boundary,
  `gripper_norm >= 0.5` means open and lower values mean closed

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
- pinned official upstream commit
- checkpoint manifest name and norm-stats file name
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

Real mode is the fail-closed default. Before starting it, generate a manifest
for the team checkpoint. The checkpoint must first contain an
`action_contract.json` identical to
`configs/action_contract.template.json`; this asserts that fine-tuning emitted
the frozen Arm_B semantics rather than a LIBERO/ALOHA-specific convention.

```powershell
Copy-Item services/openvla_oft/configs/action_contract.template.json `
  D:\models\openvla-oft\action_contract.json
python services/openvla_oft/scripts/build_checkpoint_manifest.py `
  D:\models\openvla-oft
```

The script prints the two immutable values that must be exported together with
the checkpoint location and the dataset-specific unnormalization key:

```powershell
$env:OPENVLA_OFT_CHECKPOINT_DIR = "D:\models\openvla-oft"
$env:OPENVLA_OFT_UPSTREAM_DIR = "D:\src\openvla-oft"
$env:OPENVLA_OFT_CHECKPOINT_SHA = "sha256:<manifest-sha256>"
$env:OPENVLA_OFT_NORM_STATS_SHA = "sha256:<dataset_statistics.json-sha256>"
$env:OPENVLA_OFT_UNNORM_KEY = "industrial_arm_b"
$env:INDUSTRIAL_AGENT_CAS_ROOT = "D:\industrial-cas"
python services/openvla_oft/scripts/run_service.py --host 127.0.0.1 --port 8102
```

The checkout at `OPENVLA_OFT_UPSTREAM_DIR` must be exactly
`e4287e94541f459edc4feabc4e181f537cd569a8`. The service verifies it using
`git rev-parse HEAD` and refuses to start on any other revision.

Mock mode is test-only and must be enabled explicitly:

```powershell
$env:OPENVLA_OFT_USE_MOCK = "1"
python services/openvla_oft/scripts/run_service.py --host 127.0.0.1 --port 8102
```

For a multi-container deployment, bind `--host 0.0.0.0`; the final Compose file
must mount the shared CAS volume read-only into this service.

## Docker

Build and start the real service from the repository root:

```bash
export OPENVLA_OFT_MODEL_DIR=/absolute/path/to/checkpoint
export INDUSTRIAL_AGENT_CAS_DIR=/absolute/path/to/cas
export OPENVLA_OFT_CHECKPOINT_SHA=sha256:<manifest-sha256>
export OPENVLA_OFT_NORM_STATS_SHA=sha256:<norm-stats-sha256>
export OPENVLA_OFT_UNNORM_KEY=industrial_arm_b
docker compose -f services/openvla_oft/compose.yaml up --build
```

The model and CAS mounts are read-only. The image contains the exact official
upstream commit and listens on frozen port `8102`.

## Real inference smoke

After the environment variables above are set, run one complete
CAS -> service -> official OpenVLA-OFT -> canonical action-chunk inference:

```powershell
python services/openvla_oft/scripts/smoke_real.py `
  --image artifacts\smoke\CAM_B_TOP.png `
  --state "[0,0,0,0,0,0,0]"
```

A zero exit code and a response with `status=ok` are required evidence. Unit
tests use a fake official binding and therefore do not replace this GPU smoke
or the Isaac Sim closed-loop acceptance run.

## Canonical to RLDS-style export

The OpenVLA-OFT offline data path reads the verified Canonical HDF5 episode
through `industrial_agent.data.CanonicalEpisodeReader`, filters only
`Arm_B/openvla_oft/S02_ARM_B_TRANSPORT`, loads `CAM_B_TOP` RGB pixels, aligns
Arm_B state and action rows by physics tick, and preserves Canonical source
lineage for every exported step.

```powershell
python services/openvla_oft/scripts/convert_canonical_to_rlds.py `
  --episode artifacts\canonical\arm_b_golden_episode `
  --output-dir artifacts\openvla_rlds\arm_b_golden_episode
```

The exporter writes `metadata.json`, `steps.jsonl`, and `arrays.npz`. These are
intermediate training artifacts and must remain outside Git unless a future
approved dataset-card PR explicitly records only checksums and reproduction
commands.

## Test

```powershell
python -m pytest services/openvla_oft/tests -q
```
