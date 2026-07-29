from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_ready" in response.json()


def test_predict():
    response = client.post(
        "/predict",
        json={
            "text": "You are an idiot"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data