# BIN01_TO_FINISHED01 task contract

- Task ID: `BIN01_TO_FINISHED01`
- Canonical instruction: `把Bin_01搬到FINISHED_01`
- Active arm: `Arm_B`
- Executor identity: `openvla_oft`
- Training camera/state: `CAM_B_TOP` and `Arm_B`

The Episode must start from the unmodified frozen scene configuration. The task
must not relocate, reorient, add, or remove any part at reset. Its only object
goal is to move `Bin_01` from its configured initial pose to `FINISHED_01`.
Only Arm_B actions needed to transport the bin are recordable. Arm_A must not
move, and no handoff phase is part of this instruction.

The operator grasps `BIN_CARRY_TCP`, transports the bin to `FINISHED_01`,
releases it, retreats Arm_B, and presses `C`. Terminal acceptance requires:

1. the complete bin footprint is inside `FINISHED_01`;
2. the bin height and vertical orientation are within the frozen tolerances;
3. three fresh offline-GT votes pass; and
4. ten real 100 ms open-gripper hold actions pass the 1 mm drift gate.

Detailed GT is written only to
`offline_gt/bin01_terminal_success.json`; Canonical observations remain free of
ground truth.

## Formal mother trajectory command

Run this only on the approved Linux Isaac Sim host after the branch commit,
scene SHA, OpenPI commit, HOME, IK, collision, frozen-scene invariance,
bin-only transport and terminal-gate acceptances have been frozen by the group
lead.

```bash
source /home/xyz/miniforge3/etc/profile.d/conda.sh
conda activate mylab_env
source /home/xyz/isaacsim/setup_conda_env.sh
cd /home/xyz/industrial-agent-vla-pr33

FROZEN_SHA="$(git rev-parse HEAD)"
SCENE_SHA="$(sha256sum simulation/configs/single_bin_scene_v2.json | awk '{print $1}')"
RUN_ID="bin01-finished01-train-m01-$(date +%Y%m%d-%H%M%S)"
RUN="/home/xyz/v2-formal-collection/$RUN_ID"
mkdir -p "$RUN"/{episodes,cas,artifacts,scenes}

python simulation/run_v2_keyboard_collection.py \
  --config simulation/configs/single_bin_scene_v2.json \
  --episode-root "$RUN/episodes" \
  --cas-root "$RUN/cas" \
  --artifact-dir "$RUN/artifacts" \
  --output-scene "$RUN/scenes/single_bin_scene_v2.usda" \
  --episode-id "$RUN_ID" \
  --task-id BIN01_TO_FINISHED01 \
  --instruction '把Bin_01搬到FINISHED_01' \
  --scene-seed 0 \
  --split train \
  --frozen-collection-sha "$FROZEN_SHA" \
  --expected-scene-config-sha256 "$SCENE_SHA" \
  --openpi-root /home/xyz/openpi \
  --ik-backend pink \
  --max-actions 500
```

M02 and M03 use the same frozen inputs and fresh Episode IDs with `m02` and
`m03`. A failed or safe-stopped run never counts as a mother trajectory.
