"""Validation CPU des notebooks autonomes ; aucun dataset n'est entraîné."""
import ast
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qualification():
    notebook = json.loads((ROOT / "TR_Hash_Qualification.ipynb").read_text())
    namespace = {"__name__": "notebook_unit_checks"}
    for cell in notebook["cells"]:
        if cell.get("id") in {"q-config", "q-model", "q-data", "q-training", "q-plots"}:
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    return namespace


@pytest.mark.parametrize("filename", ["TR_Hash_Qualification.ipynb", "TR_Hash_Research.ipynb"])
def test_notebooks_contain_all_code_without_framework_or_checkpoint_import(filename):
    notebook = json.loads((ROOT / filename).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Import):
                assert all(n.name.split(".")[0] not in {"complexity", "qualification"} for n in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"complexity", "qualification"}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (node.func.attr == "load" and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch")


@pytest.mark.parametrize("name,count", [("tr_hash", 110976), ("dense", 110976), ("dense_upgraded", 110976), ("learned_moe", 111488)])
def test_four_models_have_declared_budget_and_causal_predictions(qualification, name, count):
    model = qualification["build_model"](42, name).eval()
    assert model.num_parameters() == count
    ids = torch.tensor([[17, 8, 11, 42, 5, 33]])
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([[91, 127]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids)[:, :4], model(changed)[:, :4], rtol=1e-6, atol=1e-6)


def test_learned_router_receives_supervised_gradients(qualification):
    model = qualification["build_model"](42, "learned_moe")
    ids = torch.tensor([[17, 8, 11, 42, 5], [25, 9, 3, 21, 7]])
    labels = ids.clone()
    labels[:, :2] = -100
    loss = qualification["calculer_loss"](model, ids, labels)
    auxiliary = qualification["auxiliary_loss"](model, labels)
    (loss + .001 * auxiliary).backward()
    for layer in model.layers:
        gradient = layer.mlp.router.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.any(gradient != 0)
