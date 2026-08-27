# YOLO Manual-800 Model Card

## Purpose

This entry documents the Manual-800 YOLO11n checkpoint for the V2 single-bin
industrial perception task. The checkpoint detects the seven frozen V2 classes:

1. `shaft_upright`
2. `shaft_inverted`
3. `hex_nut`
4. `open_end_wrench`
5. `bin_box`
6. `bin_slot`
7. `bin_carry_handle`

The checkpoint file itself is not committed to normal Git because repository
policy excludes `.pt` model weights. The deployable artifact is identified by
its SHA-256 digest below and must be fetched from the external model artifact
store or the team delivery bundle.

## Artifact Identity

| Field | Value |
|---|---|
| Artifact ID | `yolo_manual800_yolo11n_e10_cpu` |
| Architecture | YOLO11n |
| Checkpoint filename | `best.pt` |
| Checkpoint SHA-256 | `sha256:2a8beca3ff52f6cd7a2f81f087df71793889d7017f81156a8286f4ffb106080f` |
| Class map | `services/yolo/src/yolo_service/resources/class_map.single_bin_v2.json` |
| Class map SHA-256 | `sha256:839fdb76e458f9148959e727d289a29495130ce9c868b10b57adcaab4323ba06` |
| Service config | `configs/yolo.service-manual800.json` |
| Service config SHA-256 | `sha256:a28227b8296f736280a43e5b2defb559692fe49e14f6876cf6f918321b8f1e56` |
| Perception integration config | `configs/perception.yolo-manual800.json` |
| Perception integration config SHA-256 | `sha256:11753f30a149ad77931d4daaa04d083758b7bbeee8d6d876f102d362972999eb` |
| Training seed | `7` |
| Training date | `2026-08-27` |

## Dataset

Manual-800 is composed of 794 unique images with manually cleaned YOLO labels:

| Split | Images |
|---|---:|
| train | 643 |
| val | 87 |
| test | 64 |

Class instance counts after cleanup:

| Class | Instances |
|---|---:|
| `shaft_upright` | 650 |
| `shaft_inverted` | 1114 |
| `hex_nut` | 514 |
| `open_end_wrench` | 871 |
| `bin_box` | 789 |
| `bin_slot` | 2966 |
| `bin_carry_handle` | 616 |

## Training

Manual-800 was fine-tuned from the previous Manual-500 `best.pt`.

```powershell
C:\yolo312\Scripts\python.exe work\train_manual_800_finetune.py
```

Training parameters:

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Image size | 640 |
| Batch size | 8 |
| Device | CPU |
| Workers | 0 |
| Ultralytics | 8.4.104 |

## Held-Out Test Metrics

| Class | mAP50 | mAP50-95 |
|---|---:|---:|
| all | 0.925 | 0.771 |
| `shaft_upright` | 0.993 | 0.877 |
| `shaft_inverted` | 0.918 | 0.731 |
| `hex_nut` | 0.879 | 0.803 |
| `open_end_wrench` | 0.773 | 0.523 |
| `bin_box` | 0.989 | 0.858 |
| `bin_slot` | 0.983 | 0.933 |
| `bin_carry_handle` | 0.939 | 0.675 |

Overall:

```text
mAP50:    0.925
mAP50-95: 0.771
Precision: 0.902
Recall:    0.880
```

## Known Limitations

- `open_end_wrench` remains the weakest class, especially for tight box quality.
- `bin_carry_handle` improves after targeted cleanup but still needs real-camera
  occlusion coverage.
- Metrics are from held-out same-domain data and must be complemented with the
  real three-camera inference probe before production gating.
