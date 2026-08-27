# Manual-800 Real Three-Camera Inference Test

This report records the real three-camera inference procedure for the
Manual-800 YOLO checkpoint.

## Required Inputs

Use one synchronized RGB triplet from the frozen camera order:

| Stream | Camera ID |
|---|---|
| `arm_a_rgb` | `CAM_A_TOP` |
| `handoff_rgb` | `CAM_HANDOFF` |
| `arm_b_rgb` | `CAM_B_TOP` |

Each image must be 1280x720 RGB and should be registered in the shared CAS as
an `ImageReference` before running the contract-level probe.

## Model Identity

```text
checkpoint_sha: sha256:2a8beca3ff52f6cd7a2f81f087df71793889d7017f81156a8286f4ffb106080f
class_map_sha:  sha256:839fdb76e458f9148959e727d289a29495130ce9c868b10b57adcaab4323ba06
config_sha:     sha256:a28227b8296f736280a43e5b2defb559692fe49e14f6876cf6f918321b8f1e56
```

## Procedure

1. Start the YOLO service in real mode with the Manual-800 checkpoint:

```powershell
$env:YOLO_USE_MOCK = "0"
$env:YOLO_CHECKPOINT_PATH = "<external-artifact-path>\\best.pt"
$env:YOLO_CHECKPOINT_SHA = "sha256:2a8beca3ff52f6cd7a2f81f087df71793889d7017f81156a8286f4ffb106080f"
$env:YOLO_CLASS_MAP_SHA = "sha256:839fdb76e458f9148959e727d289a29495130ce9c868b10b57adcaab4323ba06"
$env:YOLO_DEVICE = "cpu"
yolo-service --print-identity
```

2. Capture a synchronized observation containing all three camera references.

3. Run the three-camera probe:

```powershell
python scripts\run_yolo_three_camera_probe.py `
  --observation-json <validated-observation.json> `
  --base-url http://127.0.0.1:8103 `
  --subtask-id P01_TO_S11 `
  --output-json reports\yolo\manual800\three-camera-probe-summary.json `
  --evidence-jsonl artifacts\detection\manual800-three-camera-probe.jsonl
```

4. The probe must report `successful_camera_count = 3`, preserve the camera
   order `CAM_A_TOP`, `CAM_HANDOFF`, `CAM_B_TOP`, and emit per-camera detection
   packets with the model identity above.

## Current Status

The script and configuration are included in this PR. The final real-image
execution artifact is pending availability of the private synchronized
three-camera images.
