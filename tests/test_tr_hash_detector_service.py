import io
import json
from pathlib import Path

import pytest
from PIL import Image
from safetensors.torch import save_file

from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from complexity.generative.detection.service import create_app  # noqa: E402


def _checkpoint(path: Path) -> Path:
    config = TRHashDetectorConfig(
        image_size=32,
        patch_size=8,
        vision_hidden_size=16,
        vision_layers=3,
        vision_stage_depths=(1, 1, 1),
        vision_window_size=2,
        vision_heads=4,
        vision_num_experts=2,
        vision_top_k=1,
        vision_expert_width=8,
        vision_precision="fp32",
        num_classes=3,
    )
    model = TRHashObjectDetector(config)
    path.mkdir()
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        str(path / "model.safetensors"),
    )
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    return path


def test_service_health_model_auth_and_prediction(tmp_path: Path):
    app = create_app(
        _checkpoint(tmp_path / "checkpoint"),
        device="cpu",
        jobs_root=tmp_path / "jobs",
        api_key="secret",
    )
    image = Image.new("RGB", (48, 32), (200, 200, 200))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    with TestClient(app) as client:
        health = client.get("/health")
        unauthorized = client.get("/v1/model")
        model = client.get("/v1/model", headers={"X-API-Key": "secret"})
        prediction = client.post(
            "/v1/predict",
            headers={"X-API-Key": "secret"},
            files={"file": ("sample.png", encoded.getvalue(), "image/png")},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert unauthorized.status_code == 401
    assert model.status_code == 200
    assert model.json()["config"]["multi_scale"] is True
    assert prediction.status_code == 200
    assert prediction.json()["image"] == {"width": 48, "height": 32}
    assert isinstance(prediction.json()["detections"], list)
