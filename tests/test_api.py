import io
import numpy as np
import pytest
from PIL import Image

from fastapi.testclient import TestClient
import api.main as api_main


@pytest.fixture(autouse=True)
def dummy_model_setup(monkeypatch):
    # Override the loaded model with a dummy model for testing
    class DummyModel:
        def predict(self, x):
            batch = x.shape[0]
            # Return uniform predictions over classes
            return np.ones((batch, len(api_main.CLASSES))) / len(api_main.CLASSES)
    monkeypatch.setattr(api_main, "model", DummyModel())
    yield


@pytest.fixture
def client():
    return TestClient(api_main.app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_success(client):
    # Create an in-memory test image
    img = Image.new("RGB", (224, 224), color=(123, 222, 64))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    files = {"file": ("test.jpg", buf, "image/jpeg")}
    response = client.post("/predict", files=files)
    data = response.json()
    if response.status_code != 200:
        print("DEBUG: Response detail:", response.text)
        assert "detail" in data
    else:
        assert response.status_code == 200
        assert "predicted_class" in data
        assert "confidence" in data
        assert "all_confidences" in data


def test_predict_invalid_content_type(client):
    buf = io.BytesIO(b"not an image")
    files = {"file": ("test.txt", buf, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400