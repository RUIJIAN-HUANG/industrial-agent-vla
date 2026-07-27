from __future__ import annotations

import unittest
from unittest.mock import patch

from simulation import isaac_compat


class IsaacCompatibilityTests(unittest.TestCase):
    def test_isaac_51_boolean_stage_result_uses_current_stage(self) -> None:
        expected_stage = object()

        def stage_function(name: str):
            functions = {
                "create_new_stage": lambda: True,
                "get_current_stage": lambda: expected_stage,
            }
            return functions[name]

        with patch.object(
            isaac_compat,
            "_stage_function",
            side_effect=stage_function,
        ):
            self.assertIs(isaac_compat.create_new_stage(), expected_stage)

    def test_older_stage_object_result_is_returned_directly(self) -> None:
        expected_stage = object()

        def stage_function(name: str):
            if name != "create_new_stage":
                self.fail(f"Unexpected stage utility request: {name}")
            return lambda: expected_stage

        with patch.object(
            isaac_compat,
            "_stage_function",
            side_effect=stage_function,
        ):
            self.assertIs(isaac_compat.create_new_stage(), expected_stage)


if __name__ == "__main__":
    unittest.main()
