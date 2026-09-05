from simulation import run_v2_keyboard_collection as collection
from simulation import run_v2_ik_reachability_acceptance as reachability


def test_pink_preloads_bindings_in_required_order(monkeypatch) -> None:
    imported: list[str] = []

    def fake_import(module_name: str):
        imported.append(module_name)
        return object()

    monkeypatch.setattr(collection.importlib, "import_module", fake_import)

    collection._preload_pink_runtime("pink")

    assert imported == ["eigenpy", "pinocchio"]


def test_lula_does_not_preload_pink_dependencies(monkeypatch) -> None:
    imported: list[str] = []

    def fake_import(module_name: str):
        imported.append(module_name)
        return object()

    monkeypatch.setattr(collection.importlib, "import_module", fake_import)

    collection._preload_pink_runtime("lula")

    assert imported == []


def test_read_only_reachability_preloads_pink_in_same_order(monkeypatch) -> None:
    imported: list[str] = []

    def fake_import(module_name: str):
        imported.append(module_name)
        return object()

    monkeypatch.setattr(reachability.importlib, "import_module", fake_import)

    reachability._preload_pink_runtime("pink")

    assert imported == ["eigenpy", "pinocchio"]
