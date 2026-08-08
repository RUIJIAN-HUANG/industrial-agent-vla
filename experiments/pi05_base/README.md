# π0.5 Base Compatibility Experiment

This directory is an isolated, zero-shot compatibility probe. It does not
participate in the production Supervisor, does not call an Agent, and does not
modify model weights.

## Safety boundary

- The only accepted checkpoint identity is the original `pi05_base` checkpoint.
- No dataset loader, optimizer, training entrypoint, or norm-stat computation is
  called.
- Mock mode is Windows plumbing only. Its report always sets
  `valid_model_inference_evidence=false`.
- The first real run performs inference only. It never sends actions to Isaac.
- Native action output is preserved. Unknown 32-D output must never be truncated
  to the project's 7-D Cartesian action contract.

The configured `pi05_droid` input profile is a compatibility hypothesis for a
Franka-like embodiment; it does not change the `pi05_base` weights. It requires
both a real exterior image and a real wrist image. Until input and action
semantics are confirmed, the result proves model loading/inference only.

## Windows checks

From the repository root:

```powershell
python -m pytest experiments\pi05_base\tests -q
python -m experiments.pi05_base.probe_base_inference --mode mock
```

The mock command writes
`experiments/pi05_base/artifacts/base-inference-report.json`. That file is a
software wiring artifact, not scientific evidence about π0.5.

## Linux/WSL real probe

Run this only in an official OpenPI environment with the base checkpoint and
real synchronized images:

```bash
python -m experiments.pi05_base.probe_base_inference \
  --mode real \
  --image /path/to/CAM_A_TOP.png \
  --wrist-image /path/to/ARM_A_WRIST.png \
  --joint-state "0,-0.7,0,-2.2,0,1.6,0.8" \
  --gripper 1.0
```

Do not connect the output to Isaac until the report's action dimension and
semantics have been independently confirmed.

