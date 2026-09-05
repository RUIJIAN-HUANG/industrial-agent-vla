# W01_TO_S14 task contract

- Task ID: `W01_TO_S14`
- Canonical instruction: `把W01放到S14中`
- Active arm: `Arm_A`
- Target part and slot: `W01` and `S14`

The group-lead acceptance rule is that W01 is inside the S14 cell and remains
stable during the terminal hold. Final wrench orientation is not a success
condition. The scene labels `flat_y` and `wrench_y` describe the initial pose
and slot geometry; orientation and full-bound containment are retained only as
offline diagnostics.

Terminal acceptance therefore requires:

1. the W01 center is inside the configured S14 cell volume;
2. S14 is the nearest configured slot;
3. fresh offline ground-truth votes pass; and
4. the one-second terminal hold passes the drift limit.
