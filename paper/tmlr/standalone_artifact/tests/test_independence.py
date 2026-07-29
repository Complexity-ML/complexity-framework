from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_has_no_private_framework_dependency() -> None:
    package_name = "complex" + "ity"
    forbidden = (f"from {package_name}", f"import {package_name}", package_name + "-framework")
    checked = []
    for path in ROOT.rglob("*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text()
        checked.append(path)
        assert not any(term in text for term in forbidden), path
    assert checked


def test_public_package_is_small_and_focused() -> None:
    source_files = sorted((ROOT / "mini_wrv").glob("*.py"))
    assert {path.name for path in source_files} == {
        "__init__.py",
        "attention.py",
        "data.py",
        "model.py",
        "train.py",
    }
