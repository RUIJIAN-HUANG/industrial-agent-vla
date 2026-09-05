# YOLO Manual-1394 Wrench Model Card

## Purpose

`Manual-1394 Wrench` is a YOLO11n checkpoint for the V2 single-bin industrial
perception service. It extends the reviewed Manual-1194 model with 200
human-reviewed wrench-focused frames and retains the seven frozen V2 classes.

## Artifact Identity

| Field | Value |
|---|---|
| Artifact ID | `yolo_manual1394_wrench_yolo11n_e5_cpu` |
| Checkpoint | `best.pt` |
| Checkpoint SHA-256 | `sha256:6bb9d5006e732426458322e7258d3043e367317dfd46ae54920f9605a90b9536` |
| Class map SHA-256 | `sha256:839fdb76e458f9148959e727d289a29495130ce9c868b10b57adcaab4323ba06` |
| Runtime effective config SHA-256 | `sha256:f912e17a823bce66092ab730472919c90d024007d1e0cb5a497f54edb24fcff5` |
| Service config | `configs/yolo.service-manual1394-wrench.json` |
| Perception config | `configs/perception.yolo-manual1394-wrench.json` |

The weight file is versioned in the dedicated model-artifact repository, not
in this code repository.

## Training

The checkpoint was fine-tuned for five CPU epochs from the deployed
Manual-1194 reviewed checkpoint. The dataset contains 1,394 images: 1,149
training images, 147 validation images, and an unchanged 98-image frozen test
set. The newly reviewed batch contributes 412 open-end-wrench annotations.

## Frozen Test Results

| Metric | Before wrench batch | Manual-1394 Wrench |
|---|---:|---:|
| Overall mAP50-95 | 0.765 | 0.775 |
| Overall mAP50 | 0.934 | 0.934 |
| Wrench precision | 0.895 | 0.766 |
| Wrench recall | 0.682 | 0.839 |
| Wrench mAP50 | 0.834 | 0.852 |
| Wrench mAP50-95 | 0.615 | 0.614 |

The deployment deliberately favors wrench recall. Operators should expect more
wrench candidates and use the existing confidence and downstream verification
controls when false positives are costly.
