"""Run success, same-strategy recovery, and executor-switch demonstrations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_agent.contracts import Postcondition, TaskSchema  # noqa: E402
from industrial_agent.mock import MockExecutor, MockSimulator  # noqa: E402
from industrial_agent.orchestrator import IndustrialAgent  # noqa: E402


def run_scenario(scenario: str) -> dict[str, object]:
    openvla = MockExecutor("openvla_oft", dx_m=0.01)
    pi05 = MockExecutor("pi05", dx_m=0.02)
    agent = IndustrialAgent(
        [openvla, pi05],
        # Keep the demo compact: one failed decision triggers the documented
        # replan path. Production defaults to a larger receding-horizon budget.
        max_decisions_per_strategy_attempt=1,
    )
    task = TaskSchema(
        task_id=f"demo-{scenario}",
        instruction="把红色物体放入料箱",
        task_type="mock_demo",
        preferred_executor="openvla_oft",
        postconditions=(
            Postcondition(
                kind="field_equals",
                path="task.status",
                expected="done",
                required_votes=2,
            ),
        ),
    )
    result = agent.run(task, MockSimulator(scenario=scenario))  # type: ignore[arg-type]
    return {
        "scenario": scenario,
        "state": result.state.value,
        "success": result.success,
        "failure_code": result.failure_code.value,
        "executor_history": list(result.executor_history),
        "replan_counts": result.replan_counts,
        "switch_count": result.switch_count,
        "openvla_plan_calls": openvla.plan_calls,
        "pi05_plan_calls": pi05.plan_calls,
        "event_count": len(result.events),
    }


def main() -> int:
    results = [run_scenario(name) for name in ("success", "recovery", "switch")]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["success"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
