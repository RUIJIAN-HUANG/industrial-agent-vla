from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

from simulation.v2_competition_window import V2CompetitionWindow


class V2CompetitionWindowTests(unittest.TestCase):
    def test_window_does_not_override_kit_global_font_atlas(self) -> None:
        source = textwrap.dedent(inspect.getsource(V2CompetitionWindow.__init__))
        tree = ast.parse(source)

        styled_vstacks = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "VStack":
                continue
            if any(keyword.arg == "style" for keyword in node.keywords):
                styled_vstacks.append(node)

        self.assertEqual(styled_vstacks, [])
        self.assertNotIn("_resolve_cjk_font", source)
        self.assertNotIn("_ui_styles", source)


if __name__ == "__main__":
    unittest.main()
