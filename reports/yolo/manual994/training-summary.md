# Manual-994 YOLO Training Summary

## Dataset

| Field | Value |
|---|---:|
| Images | 994 |
| Train | 810 |
| Val | 105 |
| Test | 79 |
| Manual-800 source images | 794 |
| Newly corrected images | 200 |

The dataset passed YOLO label structure validation with no missing labels,
out-of-range boxes, or merge errors.

## Training Run

| Field | Value |
|---|---|
| Base checkpoint | Manual-800 `best.pt` |
| Architecture | YOLO11n |
| Epochs | 10 |
| Image size | 640 |
| Batch size | 8 |
| Device | CPU |
| Ultralytics | 8.4.104 |

Final validation metrics from epoch 10:

| Metric | Value |
|---|---:|
| Precision | 0.889 |
| Recall | 0.904 |
| mAP50 | 0.930 |
| mAP50-95 | 0.807 |

## Held-Out Test Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.905 | 0.887 | 0.936 | 0.793 |
| shaft_upright | 0.952 | 0.951 | 0.982 | 0.868 |
| shaft_inverted | 0.877 | 0.850 | 0.915 | 0.739 |
| hex_nut | 0.806 | 0.833 | 0.899 | 0.793 |
| open_end_wrench | 0.868 | 0.737 | 0.833 | 0.584 |
| bin_box | 0.967 | 0.987 | 0.990 | 0.889 |
| bin_slot | 0.935 | 0.971 | 0.980 | 0.939 |
| bin_carry_handle | 0.932 | 0.878 | 0.956 | 0.738 |

## Artifact Identity

```text
checkpoint_sha: sha256:67a70dd1f575919bde9184a993097771bbdbaa7516cdd251c1f91b2a490f1e5c
class_map_sha:  sha256:839fdb76e458f9148959e727d289a29495130ce9c868b10b57adcaab4323ba06
config_sha:     sha256:f912e17a823bce66092ab730472919c90d024007d1e0cb5a497f54edb24fcff5
```

The checkpoint is intentionally excluded from Git and must be supplied through
the external model artifact store or team delivery bundle.
