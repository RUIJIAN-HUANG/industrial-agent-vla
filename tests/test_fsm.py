from __future__ import annotations

import unittest

from industrial_agent.fsm import AgentFSM, AgentState


class FSMTests(unittest.TestCase):
    def test_valid_happy_path(self) -> None:
        fsm = AgentFSM()
        for state in (
            AgentState.VALIDATING_TASK,
            AgentState.PLANNING,
            AgentState.OBSERVING,
            AgentState.SELECTING_EXECUTOR,
            AgentState.EXECUTING,
            AgentState.VERIFYING,
            AgentState.SUCCEEDED,
        ):
            fsm.transition(state, "test")
        self.assertEqual(fsm.state, AgentState.SUCCEEDED)

    def test_illegal_transition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "illegal"):
            AgentFSM().transition(AgentState.EXECUTING, "skip validation")


if __name__ == "__main__":
    unittest.main()
