import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the AI Restaurant Recommendation API"}

def test_recommend_endpoint_logic():
    payload = {
        "query": "authentic North Indian food",
        "location": "Banashankari",
        "max_price": 500
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendation" in data
    assert len(data["recommendation"]) > 0

def test_recommend_missing_optional_params():
    # Only query is required
    payload = {"query": "South Indian"}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    assert "recommendation" in response.json()

def test_recommend_invalid_payload():
    # Missing required 'query'
    payload = {"location": "Bangalore"}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
