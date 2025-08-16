# Electric Vehicle Population Analysis Application

This full-stack application offers a comprehensive platform for analyzing electric vehicle (EV) populations. It leverages a powerful backend built with Flask to handle data processing, cleaning, exploratory data analysis (EDA), and machine learning model executions. The frontend, crafted with React, provides an intuitive user interface for interacting with the backend services, enabling users to load data, clean it, perform EDA, and run various predictive models.

**Author:** Malak Parmar  
**Email:** malakparmar.29@gmail.com  
## Overview

Production-ready ML platform for analyzing electric vehicle adoption patterns using Washington State EV population data. Built with modern MLOps practices, featuring real-time predictions, experiment tracking, and scalable architecture.

## Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[React Web App]
        API_CLIENT[API Clients]
    end
    
    subgraph "API Gateway"
        NGINX[Nginx Reverse Proxy]
    end
    
    subgraph "Application Layer"
        FASTAPI[FastAPI Server]
        CELERY[Celery Workers]
        MLFLOW[MLflow Server]
    end
    
    subgraph "Data Layer"
        POSTGRES[(PostgreSQL)]
        REDIS[(Redis Cache)]
        S3[S3 Storage]
    end
    
    subgraph "Monitoring"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
        LOGS[ELK Stack]
    end
    
    WEB --> NGINX
    API_CLIENT --> NGINX
    NGINX --> FASTAPI
    FASTAPI --> POSTGRES
    FASTAPI --> REDIS
    FASTAPI --> CELERY
    CELERY --> POSTGRES
    CELERY --> S3
    FASTAPI --> MLFLOW
    MLFLOW --> S3
    FASTAPI --> PROMETHEUS
    PROMETHEUS --> GRAFANA
    FASTAPI --> LOGS
```

## Tech Stack

- **Backend:** FastAPI, SQLAlchemy, Celery
- **Frontend:** React 18, TypeScript, Ant Design
- **ML:** Scikit-learn, XGBoost, MLflow
- **Database:** PostgreSQL, Redis
- **Infrastructure:** Docker, Kubernetes, GitHub Actions
- **Monitoring:** Prometheus, Grafana

## Key Features

- **Data Pipeline:** Automated EV dataset processing and feature engineering
- **ML Models:** Random Forest (94.3% accuracy), Gradient Boosting, XGBoost
- **API:** RESTful endpoints with JWT authentication and rate limiting
- **Real-time Predictions:** Single and batch prediction capabilities
- **Experiment Tracking:** MLflow integration for model versioning
- **Monitoring:** Comprehensive metrics and dashboards


## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+

### Local Development

```bash
# Clone repository
git clone https://github.com/malak29/electric-vehicle-analysis.git
cd electric-vehicle-analysis

# Start services
docker-compose up -d

# Access applications
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
# MLflow UI: http://localhost:5000
```

### API Example

```python
import requests

# Get predictions
response = requests.post(
    "http://localhost:8000/api/v1/predictions/single",
    json={
        "model_id": "random_forest_v1",
        "features": {
            "make": "TESLA",
            "model": "Model 3",
            "year": 2024,
            "ev_type": "BEV",
            "range": 300
        }
    }
)
print(response.json())
```

## Model Performance

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Random Forest | 94.3% | 0.942 | 2.3s |
| XGBoost | 92.8% | 0.925 | 3.1s |
| Gradient Boosting | 91.2% | 0.908 | 4.5s |

## API Endpoints

- `POST /api/v1/auth/login` - Authentication
- `GET /api/v1/data/datasets` - List datasets
- `POST /api/v1/models/train` - Train model
- `POST /api/v1/predictions/single` - Single prediction
- `POST /api/v1/predictions/batch` - Batch predictions
- `GET /api/v1/analytics/summary` - Platform metrics

Full API documentation available at `/docs` when running locally.

## Deployment

### Production (Kubernetes)

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Scale deployment
kubectl scale deployment ev-analysis --replicas=5
```

### Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost/ev_analysis
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000
SECRET_KEY=your-secret-key
```

## Testing

```bash
# Run tests
pytest tests/ --cov=app

# Run with coverage
pytest --cov=app --cov-report=html
```

## CI/CD Pipeline

Automated pipeline using GitHub Actions:
- **Build:** Docker image creation
- **Test:** Unit and integration tests
- **Deploy:** Kubernetes deployment
- **Monitor:** Health checks and alerts

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contact

**Malak Parmar**  
Email: malakparmar.29@gmail.com  
LinkedIn: [linkedin.com/in/malak29](https://linkedin.com/in/malak29)  
GitHub: [github.com/malak29](https://github.com/malak29)

