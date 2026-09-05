from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from simulation import isaac_compat


class FakeUsdStage:
    pass


class IsaacCompatibilityTests(unittest.TestCase):
    def test_simplified_chinese_locale_matches_kit_registration(self) -> None:
        self.assertEqual(
            isaac_compat.SIMPLIFIED_CHINESE_LOCALE_ARG,
            "--/persistent/app/locale_id=zh-CN",
        )

    def test_launch_can_preload_simplified_chinese_before_kit_startup(self) -> None:
        simulation_app = object()
        simulation_app_module = types.SimpleNamespace(
            SimulationApp=lambda config: (simulation_app, config)
        )
        with patch.dict(
            sys.modules,
            {"isaacsim": simulation_app_module},
        ):
            result, config = isaac_compat.launch_simulation_app(
                headless=False,
                enable_chinese_ui=True,
            )

        self.assertIs(result, simulation_app)
        self.assertEqual(config["headless"], False)
        self.assertEqual(
            config["extra_args"],
            [
                "--enable",
                isaac_compat.SIMPLIFIED_CHINESE_EXTENSION,
                isaac_compat.SIMPLIFIED_CHINESE_LOCALE_ARG,
            ],
        )

    def test_version_gate_accepts_isaac_sim_51(self) -> None:
        extension_manager = types.SimpleNamespace(
            is_extension_enabled=lambda extension_id: True,
        )
        app_module = types.SimpleNamespace(
            get_app=lambda: types.SimpleNamespace(
                get_extension_manager=lambda: extension_manager
            )
        )

        isaacsim_version = types.SimpleNamespace(
            get_version=lambda: (
                "5.1.0",
                "",
                "5",
                "1",
                "0",
                "",
                "123",
                "release",
            )
        )
        with patch.dict(
            sys.modules,
            {
                "omni": types.SimpleNamespace(
                    kit=types.SimpleNamespace(app=app_module)
                ),
                "omni.kit": types.SimpleNamespace(app=app_module),
                "omni.kit.app": app_module,
                "isaacsim.core.version": isaacsim_version,
            },
        ):
            info = isaac_compat.require_isaac_sim_51()

        self.assertEqual(info["core_version"], "5.1.0")
        self.assertEqual((info["major"], info["minor"]), ("5", "1"))

    def test_version_gate_rejects_other_isaac_sim_release(self) -> None:
        extension_manager = types.SimpleNamespace(
            is_extension_enabled=lambda extension_id: True,
        )
        app_module = types.SimpleNamespace(
            get_app=lambda: types.SimpleNamespace(
                get_extension_manager=lambda: extension_manager
            )
        )

        with patch.dict(
            sys.modules,
            {
                "omni": types.SimpleNamespace(
                    kit=types.SimpleNamespace(app=app_module)
                ),
                "omni.kit": types.SimpleNamespace(app=app_module),
                "omni.kit.app": app_module,
                "isaacsim.core.version": types.SimpleNamespace(
                    get_version=lambda: (
                        "4.5.0",
                        "",
                        "4",
                        "5",
                        "0",
                        "",
                        "123",
                        "release",
                    )
                ),
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "requires Isaac Sim 5.1"):
                isaac_compat.require_isaac_sim_51()

    def test_version_gate_rejects_extension_enable_failure(self) -> None:
        extension_manager = types.SimpleNamespace(
            is_extension_enabled=lambda extension_id: False,
            set_extension_enabled_immediate=lambda extension_id, enabled: False,
        )
        app_module = types.SimpleNamespace(
            get_app=lambda: types.SimpleNamespace(
                get_extension_manager=lambda: extension_manager
            )
        )

        with patch.dict(
            sys.modules,
            {
                "omni": types.SimpleNamespace(
                    kit=types.SimpleNamespace(app=app_module)
                ),
                "omni.kit": types.SimpleNamespace(app=app_module),
                "omni.kit.app": app_module,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "version metadata"):
                isaac_compat.require_isaac_sim_51()

    def test_isaac_51_boolean_stage_result_uses_current_stage(self) -> None:
        expected_stage = FakeUsdStage()

        def stage_function(name: str):
            functions = {
                "create_new_stage": lambda: True,
                "get_current_stage": lambda: expected_stage,
            }
            return functions[name]

        with (
            patch.object(
                isaac_compat,
                "_stage_function",
                side_effect=stage_function,
            ),
            patch.object(isaac_compat, "_usd_stage_type", return_value=FakeUsdStage),
        ):
            self.assertIs(isaac_compat.create_new_stage(), expected_stage)

    def test_isaac_51_false_stage_result_fails_closed(self) -> None:
        with patch.object(
            isaac_compat,
            "_stage_function",
            return_value=lambda: False,
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to create"):
                isaac_compat.create_new_stage()

    def test_older_stage_object_result_is_returned_directly(self) -> None:
        expected_stage = FakeUsdStage()

        def stage_function(name: str):
            if name != "create_new_stage":
                self.fail(f"Unexpected stage utility request: {name}")
            return lambda: expected_stage

        with (
            patch.object(
                isaac_compat,
                "_stage_function",
                side_effect=stage_function,
            ),
            patch.object(isaac_compat, "_usd_stage_type", return_value=FakeUsdStage),
        ):
            self.assertIs(isaac_compat.create_new_stage(), expected_stage)

    def test_get_current_stage_returns_fresh_valid_handle(self) -> None:
        expected_stage = FakeUsdStage()

        with (
            patch.object(
                isaac_compat,
                "_stage_function",
                return_value=lambda: expected_stage,
            ),
            patch.object(isaac_compat, "_usd_stage_type", return_value=FakeUsdStage),
        ):
            self.assertIs(isaac_compat.get_current_stage(), expected_stage)

    def test_get_current_stage_rejects_boolean_result(self) -> None:
        with patch.object(
            isaac_compat,
            "_stage_function",
            return_value=lambda: True,
        ):
            with self.assertRaisesRegex(RuntimeError, "valid USD Stage"):
                isaac_compat.get_current_stage()

    def test_get_current_stage_rejects_non_stage_object(self) -> None:
        with (
            patch.object(
                isaac_compat,
                "_stage_function",
                return_value=lambda: object(),
            ),
            patch.object(isaac_compat, "_usd_stage_type", return_value=FakeUsdStage),
        ):
            with self.assertRaisesRegex(TypeError, "expected pxr.Usd.Stage"):
                isaac_compat.get_current_stage()

    def test_stage_contract_accepts_z_up_meters_and_kilograms(self) -> None:
        stage = FakeUsdStage()
        pxr_module = types.SimpleNamespace(
            Usd=types.SimpleNamespace(Stage=FakeUsdStage),
            UsdGeom=types.SimpleNamespace(
                GetStageUpAxis=lambda value: "Z",
                GetStageMetersPerUnit=lambda value: 1.0,
            ),
            UsdPhysics=types.SimpleNamespace(
                GetStageKilogramsPerUnit=lambda value: 1.0
            ),
        )

        with patch.dict(sys.modules, {"pxr": pxr_module}):
            isaac_compat.validate_stage_contract(stage)

    def test_stage_contract_writes_and_reads_back_frozen_metadata(self) -> None:
        stage = FakeUsdStage()
        metadata: dict[str, object] = {}
        pxr_module = types.SimpleNamespace(
            Usd=types.SimpleNamespace(Stage=FakeUsdStage),
            UsdGeom=types.SimpleNamespace(
                Tokens=types.SimpleNamespace(z="Z"),
                SetStageUpAxis=lambda value, axis: metadata.update(up_axis=axis),
                SetStageMetersPerUnit=lambda value, unit: metadata.update(
                    meters_per_unit=unit
                ),
                GetStageUpAxis=lambda value: metadata["up_axis"],
                GetStageMetersPerUnit=lambda value: metadata["meters_per_unit"],
            ),
            UsdPhysics=types.SimpleNamespace(
                SetStageKilogramsPerUnit=lambda value, unit: metadata.update(
                    kilograms_per_unit=unit
                ),
                GetStageKilogramsPerUnit=lambda value: metadata["kilograms_per_unit"],
            ),
        )

        with patch.dict(sys.modules, {"pxr": pxr_module}):
            isaac_compat.configure_and_validate_stage_contract(stage)

        self.assertEqual(
            metadata,
            {
                "up_axis": "Z",
                "meters_per_unit": 1.0,
                "kilograms_per_unit": 1.0,
            },
        )

    def test_stage_contract_rejects_wrong_axis_or_units(self) -> None:
        stage = FakeUsdStage()
        pxr_module = types.SimpleNamespace(
            Usd=types.SimpleNamespace(Stage=FakeUsdStage),
            UsdGeom=types.SimpleNamespace(
                GetStageUpAxis=lambda value: "Y",
                GetStageMetersPerUnit=lambda value: 0.01,
            ),
            UsdPhysics=types.SimpleNamespace(
                GetStageKilogramsPerUnit=lambda value: 0.001
            ),
        )

        with patch.dict(sys.modules, {"pxr": pxr_module}):
            with self.assertRaisesRegex(
                RuntimeError,
                "up axis.*metersPerUnit.*kilogramsPerUnit",
            ):
                isaac_compat.validate_stage_contract(stage)


if __name__ == "__main__":
    unittest.main()
