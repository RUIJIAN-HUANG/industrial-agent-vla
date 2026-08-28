# YOLO Manual-994 Model Card

## Purpose

This entry documents the Manual-994 YOLO11n checkpoint for the V2 single-bin
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
| Artifact ID | `yolo_manual994_yolo11n_e10_cpu` |
| Architecture | YOLO11n |
| Checkpoint filename | `best.pt` |
| Checkpoint SHA-256 | `sha256:67a70dd1f575919bde9184a993097771bbdbaa7516cdd251c1f91b2a490f1e5c` |
| Class map | `services/yolo/src/yolo_service/resources/class_map.single_bin_v2.json` |
| Class map SHA-256 | `sha256:839fdb76e458f9148959e727d289a29495130ce9c868b10b57adcaab4323ba06` |
| Service config | `configs/yolo.service-manual994.json` |
| Service config SHA-256 | `sha256:a28227b8296f736280a43e5b2defb559692fe49e14f6876cf6f918321b8f1e56` |
| Runtime effective config SHA-256 | `sha256:f912e17a823bce66092ab730472919c90d024007d1e0cb5a497f54edb24fcff5` |
| Perception integration config | `configs/perception.yolo-manual994.json` |
| Perception integration config SHA-256 | `sha256:1e53b223ad673603f8bcc547e3d5214446f73293aeec365f78456cb88ce61d98` |
| Training seed | `7` |
| Training date | `2026-08-28` |

## Dataset

Manual-994 is composed of 994 unique images with manually cleaned YOLO labels:

| Split | Images |
|---|---:|
| train | 810 |
| val | 105 |
| test | 79 |

Sources:

| Source | Images |
|---|---:|
| Manual-800 | 794 |
| Newly corrected samples | 200 |

Class instance counts after cleanup:

| Class | Instances |
|---|---:|
| `shaft_upright` | 794 |
| `shaft_inverted` | 1335 |
| `hex_nut` | 638 |
| `open_end_wrench` | 1056 |
| `bin_box` | 985 |
| `bin_slot` | 3741 |
| `bin_carry_handle` | 775 |

## Training

Manual-994 was fine-tuned from the Manual-800 `best.pt`.

```powershell
C:\yolo312\Scripts\python.exe work\train_manual_994_finetune.py
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

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.905 | 0.887 | 0.936 | 0.793 |
| `shaft_upright` | 0.952 | 0.951 | 0.982 | 0.868 |
| `shaft_inverted` | 0.877 | 0.850 | 0.915 | 0.739 |
| `hex_nut` | 0.806 | 0.833 | 0.899 | 0.793 |
| `open_end_wrench` | 0.868 | 0.737 | 0.833 | 0.584 |
| `bin_box` | 0.967 | 0.987 | 0.990 | 0.889 |
| `bin_slot` | 0.935 | 0.971 | 0.980 | 0.939 |
| `bin_carry_handle` | 0.932 | 0.878 | 0.956 | 0.738 |

Overall:

```text
mAP50:    0.936
mAP50-95: 0.793
Precision: 0.905
Recall:    0.887
```

Compared with Manual-800, same-domain held-out mAP50-95 improved from `0.771`
to `0.793`. The Manual-994 test split includes newly corrected samples, so this
is practical candidate evidence rather than a fixed-benchmark comparison.

## Known Limitations

- `open_end_wrench` remains the weakest class, especially for tight box quality.
- `bin_carry_handle` improved over Manual-800 but still needs real-camera
  occlusion coverage.
- Metrics are from held-out same-domain data and must be complemented with the
  real three-camera inference probe before production gating.
