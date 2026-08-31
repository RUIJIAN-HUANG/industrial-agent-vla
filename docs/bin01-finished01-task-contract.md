# BIN01_TO_FINISHED01 task contract

- Task ID: `BIN01_TO_FINISHED01`
- Canonical instruction: `把Bin_01搬到FINISHED_01`
- Active-arm sequence: `Arm_A -> HANDOFF_VERIFY -> Arm_B`
- Executor identities: `Arm_A/pi05`, then `Arm_B/pi05`
- Canonical source: all three frozen cameras and both arm states remain unchanged;
  each action stores its actual `arm_id` and source executor
- Training input: Arm_A windows use `CAM_A_TOP` + Arm_A state; Arm_B windows use
  `CAM_B_TOP` + Arm_B state. Windows never cross the handoff boundary

The Episode must start from the unmodified frozen scene configuration. The task
must not relocate, reorient, add, or remove any part at reset. Its only object
goal is to move `Bin_01` from its configured initial pose to `FINISHED_01`.
The group-lead-approved execution is a controlled dual-arm relay. Arm_A moves
the bin from its frozen initial pose to `HANDOFF_CENTER`, releases it, and
retreats. The operator presses `V` to verify the complete bin footprint,
height/orientation, open Arm_A gripper, and Arm_A clearance. Both arms remain
locked in `HANDOFF_VERIFY` until the operator presses `B`; only then may Arm_B
move the same bin from `HANDOFF_CENTER` to `FINISHED_01`. No scene object is
relocated, reoriented, added, or removed at reset.

The operator uses `BIN_CARRY_TCP` for both transfer legs, releases the bin at
`FINISHED_01`, retreats Arm_B, and presses `C`. Terminal acceptance requires:

1. the complete bin footprint is inside `FINISHED_01`;
2. the bin height and vertical orientation are within the frozen tolerances;
3. three fresh offline-GT votes pass; and
4. ten real 100 ms open-gripper hold actions pass the 1 mm drift gate.

## Visible keyboard workflow

1. The Episode starts as `A_ONLY | arm=Arm_A`.
2. Arm_A places Bin_01 in `HANDOFF_CENTER`, opens its gripper, and retreats.
3. Press `V` once. A failed verification aborts the Episode; a pass displays
   `HANDOFF_VERIFY PASS | both arms locked | press B`.
4. Press `B` once. The display changes to `B_ONLY | arm=Arm_B`.
5. Arm_B places Bin_01 in `FINISHED_01`, opens its gripper, and retreats.
6. Press `C` once to request terminal validation.

`Z` is not used by this task. Motion keys must be tapped, not held. `X` always
requests a safe stop.

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

M02 and M03 use the same frozen inputs, `--split train`, `--scene-seed 0`, and
fresh Episode IDs with `m02` and `m03`. M04 uses a fresh Episode ID containing
`val-m04`, `--split validation`, and `--scene-seed 1`. The distinct validation
seed is mandatory because the Split Registry forbids one scene seed from
crossing Train and Valid. A failed or safe-stopped run never counts as a mother
trajectory.

After each successful mother trajectory, register it before any replay
derivation or LeRobot conversion. The collection CLI uses `validation`, while
the Registry intentionally uses `val`:

```bash
REGISTRY=/home/xyz/v2-formal-collection/split_registry_v1.json

# M01-M03: change only RUN and the m01/m02/m03 scenario-group suffix.
python scripts/pi05/register_v2_split.py \
  --result-json "$RUN/artifacts/result.json" \
  --registry "$REGISTRY" \
  --split train \
  --scenario-group-id bin01-finished01-m01

# M04:
python scripts/pi05/register_v2_split.py \
  --result-json "$RUN/artifacts/result.json" \
  --registry "$REGISTRY" \
  --split val \
  --scenario-group-id bin01-finished01-m04
```

The script strictly validates the Canonical Episode, checks the result's
Episode ID/path/seed and collection split, adds without changing any existing
assignment, and computes `registry_sha256` atomically. Never edit the Registry
JSON or its SHA by hand. A derived Episode must be registered with
`--parent-episode-id <mother-episode-id>` so it inherits the mother's complete
group key and split. Complete all Registry assignments before LeRobot
conversion; changing the Registry afterward requires reconversion.
