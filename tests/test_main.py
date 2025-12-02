from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}


def test_hello():
    response = client.get("/hello/World")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_modes():
    response = client.get("/modes")
    assert response.status_code == 200
    assert "basic" in response.json()["modes"]
    assert "python" in response.json()["modes"]
