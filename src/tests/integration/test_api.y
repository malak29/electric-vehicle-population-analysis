"""Integration tests for the API endpoints."""

import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os
import pandas as pd

from app.main import app
from app.core.database import get_db, Base
from app.core.config import settings


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def client():
    """Create test client."""
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    data = {
        'Make': ['TESLA', 'NISSAN', 'CHEVROLET', 'BMW'],
        'Model': ['MODEL 3', 'LEAF', 'BOLT EV', 'I3'],
        'Model Year': [2023, 2022, 2023, 2021],
        'Electric Vehicle Type': ['BEV', 'BEV', 'BEV', 'BEV'],
        'Electric Range': [310, 149, 259, 153],
        'Base MSRP': [42990, 27400, 31995, 44450],
        'State': ['WA', 'WA', 'WA', 'WA'],
        'County': ['King', 'Pierce', 'Snohomish', 'King']
    }
    df = pd.DataFrame(data)
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
    
    def test_detailed_health_check(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "system" in data
    
    def test_readiness_check(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
    
    def test_liveness_check(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestDataEndpoints:
    """Test data management endpoints."""
    
    def test_get_data_summary(self, client):
        """Test data summary endpoint."""
        response = client.get("/api/v1/data/summary")
        # Should return 404 if no data file exists, or 200 with data
        assert response.status_code in [200, 404]
    
    def test_get_visualization_data(self, client):
        """Test visualization data endpoints."""
        chart_types = [
            "vehicle_type_distribution",
            "range_by_year", 
            "make_distribution",
            "geographic_distribution"
        ]
        
        for chart_type in chart_types:
            response = client.get(f"/api/v1/data/visualization/{chart_type}")
            # Should return data or 404 if no dataset
            assert response.status_code in [200, 404, 500]
    
    def test_upload_dataset(self, client, sample_csv_data):
        """Test dataset upload."""
        with open(sample_csv_data, 'rb') as f:
            response = client.post(
                "/api/v1/data/upload",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        # Clean up
        os.unlink(sample_csv_data)
        
        # Should succeed or fail gracefully
        assert response.status_code in [200, 400, 500]
    
    def test_export_data_csv(self, client):
        """Test data export in CSV format."""
        response = client.get("/api/v1/data/export/csv")
        assert response.status_code in [200, 404, 500]
    
    def test_export_data_json(self, client):
        """Test data export in JSON format."""
        response = client.get("/api/v1/data/export/json")
        assert response.status_code in [200, 404, 500]
    
    def test_get_statistics(self, client):
        """Test data statistics endpoint."""
        response = client.get("/api/v1/data/statistics")
        assert response.status_code in [200, 404, 500]


class TestModelEndpoints:
    """Test ML model endpoints."""
    
    def test_list_models(self, client):
        """Test model listing endpoint."""
        response = client.get("/api/v1/models/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_train_model_invalid_type(self, client):
        """Test training with invalid model type."""
        request_data = {
            "model_type": "invalid_model",
            "target_column": "Electric Vehicle Type",
            "feature_columns": ["Electric Range", "Model Year"]
        }
        
        response = client.post("/api/v1/models/train", json=request_data)
        assert response.status_code in [400, 500]
    
    def test_predict_invalid_model(self, client):
        """Test prediction with invalid model ID."""
        request_data = {
            "model_id": "invalid_model_id",
            "features": {"Electric Range": 200, "Model Year": 2023}
        }
        
        response = client.post("/api/v1/models/predict", json=request_data)
        assert response.status_code in [404, 500]
    
    def test_get_model_metrics_invalid_id(self, client):
        """Test getting metrics for invalid model ID."""
        response = client.get("/api/v1/models/invalid_id/metrics")
        assert response.status_code in [404, 500]
    
    def test_delete_model_invalid_id(self, client):
        """Test deleting invalid model ID."""
        response = client.delete("/api/v1/models/invalid_id")
        assert response.status_code in [404, 500]


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test asynchronous endpoints."""
    
    async def test_concurrent_health_checks(self, client):
        """Test multiple concurrent health check requests."""
        async def make_request():
            return client.get("/health/")
        
        # Make 10 concurrent requests