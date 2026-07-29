# Isaac execution adapter

This document describes the real Isaac Sim execution path used by the
industrial agent. It is distinct from `MockExecutionEnvironment`: commands are
converted to Franka joint targets and sent to Isaac Sim articulations.

## Components

- `src/industrial_agent/isaac_environment.py`
  - Implements the `ExecutionEnvironment` contract.
  - Validates control tokens, observation IDs, state digests, command IDs, and
    safety state before an action can reach a controller.
  - Enforces at-most-once command execution and triggers a safe stop when a
    controller call fails.
- `simulation/isaac_franka_controller.py`
  - Implements the Isaac Sim 5.1 Franka controller backend.
  - Converts the frozen seven-dimensional base-frame action into a world-frame
    end-effector target.
  - Uses Lula inverse kinematics and sends the resulting joint action through
    the Isaac articulation controller.
  - Supports gripper commands and confirmed safe-stop readback.
- `simulation/run_isaac_adapter_smoke.py`
  - Builds the frozen two-Franka scene and sends one small command through the
    complete adapter path.
  - Verifies that the joint state changes and that safe stop is confirmed.
  - This is a controlled simulator smoke test, not a collision-aware production
    motion planner.

## Local contract tests

From the repository root:

```bash
PYTHONPATH=src python3 -m unittest \
  tests.test_isaac_environment \
  tests.test_isaac_franka_controller \
  tests.test_g0_acceptance \
  tests.test_isaac_compat \
  -v
```

## Isaac Sim 5.1 smoke test

Run this only on the Linux workstation with Isaac Sim 5.1. Keep the generated
motion small and supervise the GUI on the first run.

```bash
cd "$HOME/Sceneconstruction/industrial-agent-vla"
export ISAAC_SIM_ROOT="$HOME/isaacsim"

"$ISAAC_SIM_ROOT/python.sh" simulation/run_isaac_adapter_smoke.py \
  --result-file artifacts/isaac-adapter/smoke-result.json
```

The default command moves simulated `Arm_A` by 5 mm along its base-frame Z
axis. A successful result contains:

```json
{
  "status": "PASS",
  "arm_id": "Arm_A",
  "joint_delta_norm": 0.0,
  "safe_stop_confirmed": true
}
```

`joint_delta_norm` must be greater than zero in the real output. The zero above
is only a field-shape example.

After the supervised GUI run passes, repeat headlessly:

```bash
"$ISAAC_SIM_ROOT/python.sh" simulation/run_isaac_adapter_smoke.py \
  --headless \
  --result-file artifacts/isaac-adapter/smoke-result-headless.json
```

Inspect the saved evidence:

```bash
cat artifacts/isaac-adapter/smoke-result.json
cat artifacts/isaac-adapter/smoke-result-headless.json
```

Do not treat the adapter as production-ready if either result is missing,
reports `FAIL`, has no measurable joint change, or lacks confirmed safe-stop
readback.
