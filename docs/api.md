# Electric Vehicle Analysis Platform - API Documentation

## Base URL
```
Production: https://api.ev-analysis.com/api/v1
Development: http://localhost:8000/api/v1
```

## Authentication

### JWT Authentication
The API uses JWT (JSON Web Token) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

### API Key Authentication
For programmatic access, use API keys in the header:

```http
X-API-Key: <your_api_key>
```

## Endpoints

### Authentication

#### POST /auth/login
Login with credentials and receive JWT tokens.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### POST /auth/refresh
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

#### POST /auth/register
Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "user@example.com",
  "password": "string",
  "full_name": "string"
}
```

### Data Management

#### GET /data/datasets
List all available datasets.

**Query Parameters:**
- `skip` (int): Number of items to skip
- `limit` (int): Number of items to return (max: 100)
- `status` (string): Filter by status (uploaded, processed, failed)

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "name": "string",
      "description": "string",
      "record_count": 1000,
      "upload_date": "2024-01-01T00:00:00Z",
      "status": "processed"
    }
  ],
  "total": 100,
  "skip": 0,
  "limit": 20
}
```

#### POST /data/upload
Upload a new dataset.

**Request Body:** (multipart/form-data)
- `file`: CSV or Excel file
- `name`: Dataset name
- `description`: Dataset description

**Response:**
```json
{
  "dataset_id": 1,
  "status": "processing",
  "message": "Dataset uploaded successfully"
}
```

#### GET /data/{dataset_id}
Get detailed information about a specific dataset.

**Response:**
```json
{
  "id": 1,
  "name": "string",
  "description": "string",
  "file_path": "string",
  "record_count": 1000,
  "column_count": 20,
  "columns_metadata": {
    "column_name": {
      "type": "string",
      "nullable": false,
      "unique_values": 100
    }
  },
  "statistics": {
    "numeric_columns": {},
    "categorical_columns": {}
  }
}
```

#### GET /data/{dataset_id}/preview
Preview first 100 rows of dataset.

**Response:**
```json
{
  "columns": ["col1", "col2"],
  "data": [
    {"col1": "value1", "col2": "value2"}
  ]
}
```

### Model Training

#### POST /models/train
Train a new machine learning model.

**Request Body:**
```json
{
  "dataset_id": 1,
  "model_type": "random_forest",
  "target_column": "target",
  "features": ["feature1", "feature2"],
  "parameters": {
    "test_size": 0.2,
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "enable_grid_search": false
}
```

**Response:**
```json
{
  "job_id": "job_123",
  "status": "started",
  "estimated_time": 300
}
```

#### GET /models
List all trained models.

**Query Parameters:**
- `skip` (int): Number of items to skip
- `limit` (int): Number of items to return
- `model_type` (string): Filter by model type
- `is_active` (bool): Filter by active status

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "model_id": "model_123",
      "name": "Random Forest Classifier",
      "model_type": "random_forest",
      "accuracy": 0.943,
      "training_date": "2024-01-01T00:00:00Z",
      "is_active": true
    }
  ],
  "total": 50
}
```

#### GET /models/{model_id}
Get detailed information about a specific model.

**Response:**
```json
{
  "id": 1,
  "model_id": "model_123",
  "name": "Random Forest Classifier",
  "model_type": "random_forest",
  "version": "1.0",
  "parameters": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "metrics": {
    "accuracy": 0.943,
    "precision": 0.941,
    "recall": 0.943,
    "f1_score": 0.942
  },
  "feature_importance": {
    "feature1": 0.25,
    "feature2": 0.15
  },
  "confusion_matrix": [[100, 10], [5, 85]]
}
```

#### DELETE /models/{model_id}
Delete a model (soft delete).

### Predictions

#### POST /predictions/single
Make a single prediction.

**Request Body:**
```json
{
  "model_id": "model_123",
  "features": {
    "make": "TESLA",
    "model": "Model 3",
    "model_year": 2024,
    "ev_type": "BEV",
    "electric_range": 300,
    "city": "Seattle",
    "postal_code": "98101"
  }
}
```

**Response:**
```json
{
  "prediction_id": "pred_456",
  "prediction": "High Adoption",
  "confidence": 0.92,
  "processing_time": 0.125,
  "model_used": "model_123"
}
```

#### POST /predictions/batch
Make batch predictions.

**Request Body:** (multipart/form-data)
- `file`: CSV file with input data
- `model_id`: Model to use for predictions

**Response:**
```json
{
  "job_id": "batch_789",
  "status": "processing",
  "total_records": 1000,
  "estimated_time": 60
}
```

#### GET /predictions/batch/{job_id}
Get batch prediction job status.

**Response:**
```json
{
  "job_id": "batch_789",
  "status": "completed",
  "total_records": 1000,
  "processed_records": 1000,
  "predictions_url": "/predictions/batch/batch_789/download"
}
```

#### GET /predictions/history
Get prediction history.

**Query Parameters:**
- `skip` (int): Number of items to skip
- `limit` (int): Number of items to return
- `model_id` (string): Filter by model
- `date_from` (datetime): Filter from date
- `date_to` (datetime): Filter to date

### Experiments

#### GET /experiments
List all experiments.

**Response:**
```json
{
  "items": [
    {
      "experiment_id": "exp_001",
      "name": "Hyperparameter Tuning",
      "model_type": "gradient_boosting",
      "best_score": 0.956,
      "status": "completed",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### POST /experiments/compare
Compare multiple models.

**Request Body:**
```json
{
  "model_ids": ["model_123", "model_456"],
  "metrics": ["accuracy", "f1_score", "training_time"]
}
```

### Jobs

#### GET /jobs/{job_id}
Get job status and details.

**Response:**
```json
{
  "job_id": "job_123",
  "job_type": "model_training",
  "status": "running",
  "progress": 65.5,
  "created_at": "2024-01-01T00:00:00Z",
  "started_at": "2024-01-01T00:01:00Z",
  "eta": "2024-01-01T00:10:00Z"
}
```

#### POST /jobs/{job_id}/cancel
Cancel a running job.

### Analytics

#### GET /analytics/summary
Get platform usage summary.

**Response:**
```json
{
  "total_models": 50,
  "total_predictions": 10000,
  "total_datasets": 25,
  "active_users": 100,
  "avg_model_accuracy": 0.923,
  "popular_models": [
    {
      "model_type": "random_forest",
      "usage_count": 500
    }
  ]
}
```

#### GET /analytics/model-performance
Get model performance analytics.

**Query Parameters:**
- `model_id` (string): Specific model ID
- `date_from` (datetime): Start date
- `date_to` (datetime): End date

## Error Responses

### Standard Error Format
```json
{
  "detail": "Error message",
  "status_code": 400,
  "error_code": "INVALID_REQUEST",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes
| Code | Status | Description |
|------|--------|-------------|
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |

## Rate Limiting

Default rate limits:
- **Authenticated users**: 1000 requests/hour
- **Unauthenticated users**: 100 requests/hour
- **Training endpoints**: 10 requests/hour
- **Batch predictions**: 50 requests/hour

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641024000
```

## Pagination

All list endpoints support pagination:

```http
GET /api/v1/models?skip=20&limit=10
```

Response includes pagination metadata:
```json
{
  "items": [...],
  "total": 100,
  "skip": 20,
  "limit": 10,
  "has_more": true
}
```

## Webhooks

Configure webhooks for async events:

### POST /webhooks
Register a webhook endpoint.

**Request Body:**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["model.trained", "prediction.completed"],
  "secret": "webhook_secret"
}
```

### Webhook Events
- `model.trained`: Model training completed
- `model.failed`: Model training failed
- `prediction.completed`: Batch prediction completed
- `dataset.processed`: Dataset processing completed

### Webhook Payload
```json
{
  "event": "model.trained",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "model_id": "model_123",
    "accuracy": 0.943
  }
}
```

## Code Examples

### Python
```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    json={"username": "user", "password": "pass"}
)
token = response.json()["access_token"]

# Make prediction
headers = {"Authorization": f"Bearer {token}"}
prediction = requests.post(
    "http://localhost:8000/api/v1/predictions/single",
    headers=headers,
    json={
        "model_id": "model_123",
        "features": {...}
    }
)
print(prediction.json())
```

### JavaScript
```javascript
// Using Axios
const axios = require('axios');

// Login
const { data } = await axios.post('/api/v1/auth/login', {
  username: 'user',
  password: 'pass'
});

// Make authenticated request
const prediction = await axios.post(
  '/api/v1/predictions/single',
  {
    model_id: 'model_123',
    features: {...}
  },
  {
    headers: {
      'Authorization': `Bearer ${data.access_token}`
    }
  }
);
```

### cURL
```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user","password":"pass"}'

# Make prediction
curl -X POST http://localhost:8000/api/v1/predictions/single \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"model_123","features":{...}}'
```

## SDKs

Official SDKs available:
- Python: `pip install ev-analysis-sdk`
- JavaScript: `npm install @ev-analysis/sdk`
- Go: `go get github.com/ev-analysis/go-sdk`

## API Versioning

The API uses URL versioning:
- Current version: `v1`
- Legacy support: 6 months after new version release
- Deprecation notices: Via `X-API-Deprecated` header

## Support

- Documentation: https://docs.ev-analysis.com
- API Status: https://status.ev-analysis.com
- Support Email: api-support@ev-analysis.com
- Developer Forum: https://forum.ev-analysis.com