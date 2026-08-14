V2 Pink IK + null-space posture overlay
========================================

Expected repository state before extraction:
  repo:   /home/xyz/Sceneconstruction/industrial-agent-vla
  branch: feat/b-v2-manual-industrial-collection
  HEAD:   214d91a3e0f7471bcfeb9bfe893ab5adf1ff6232

Scope
-----
- Keeps the existing W/A/S/D/Q/E and I/K/J/L/U/O keyboard mapping.
- Keeps the existing scene, gripper, cameras and Canonical Recorder.
- Selects Isaac Lab 2.3.2 native PinkIKController by default.
- Adds the native NullSpacePostureTask and damping task below the TCP task.
- Keeps Lula available with --ik-backend lula for comparison/rollback.

This first integration uses the official null-space posture task. It does not
claim to implement a separate custom directional-manipulability objective.

Install and static validation
-----------------------------
Run the command block supplied with this archive. It verifies the branch/HEAD,
extracts only the six reviewed files, and runs the relevant tests and Ruff.
Do not commit until the GUI practice acceptance below passes.

Runtime environment
-------------------
Pink and Isaac Lab are installed in mylab_env. Run the collection entry from
that environment after sourcing the Isaac Sim conda setup:

  source /home/xyz/miniforge3/etc/profile.d/conda.sh
  conda activate mylab_env
  source /home/xyz/isaacsim/setup_conda_env.sh
  cd /home/xyz/Sceneconstruction/industrial-agent-vla

Practice acceptance gate
------------------------
Start one practice episode with the same arguments used previously, plus:

  --ik-backend pink --split practice --max-actions 1500

The status window must show:

  IK backend: PINK + null-space posture

Complete P01 -> S11 manually. Acceptance requires all of the following:

1. The GUI stays open while moving across the former one-direction dead zone.
2. The grasped part remains stable while TCP orientation is held.
3. The operator can enter S11, release P01, and end the practice episode.
4. result.json reports PASS and the recorded camera/state array checks pass.

If initialization reports that the Pink URDF has no 'right_gripper' frame,
send the complete candidate-frame line. Do not guess a substitute frame:
changing it without a calibrated transform would move the physical TCP.

Rollback
--------
No file rollback is needed for a comparison run. Add:

  --ik-backend lula

Formal-data rule
----------------
Practice output remains training_allowed=false. Do not start formal collection
or commit this overlay until the P01 -> S11 practice acceptance gate passes.
