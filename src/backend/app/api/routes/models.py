"""Machine learning model endpoints."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
import mlflow
from datetime import datetime

from app.ml.models.classification import EVClassificationPipeline
from app.ml.models.clustering import EVClusteringPipeline
from app.ml.models.regression import EVRegressionPipeline
from app.ml.pipelines.training import ModelTrainer
from app.ml.pipelines.inference import EVModelRegistry
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class TrainingRequest(BaseModel):
    """Model training request."""
    model_type: str = Field(..., description="Type of model to train")
    target_column: str = Field(..., description="Target column for prediction")
    feature_columns: List[str] = Field(..., description="Feature columns")
    hyperparameters: Optional[Dict[str, Any]] = Field(default={}, description="Model hyperparameters")
    test_size: Optional[float] = Field(default=0.2, description="Test set size")


class PredictionRequest(BaseModel):
    """Prediction request."""
    model_id: str = Field(..., description="Model ID from MLflow")
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")


class ModelResponse(BaseModel):
    """Model response."""
    model_id: str
    model_type: str
    metrics: Dict[str, Any]
    status: str
    created_at: str


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: Any
    probability: Optional[List[float]] = None
    model_id: str
    timestamp: str


@router.post("/train", response_model=ModelResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train a machine learning model."""
    try:
        # Set MLflow experiment
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        
        # Load data
        df = pd.read_csv(f"{settings.DATA_PATH}/processed/electric_vehicles.csv")
        
        # Validate columns
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
        
        missing_features = [col for col in request.feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Feature columns not found: {missing_features}")
        
        # Prepare data
        X = df[request.feature_columns]
        y = df[request.target_column]
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Start training in background
        with mlflow.start_run() as run:
            mlflow.log_params({
                "model_type": request.model_type,
                "target_column": request.target_column,
                "feature_columns": request.feature_columns,
                "test_size": request.test_size
            })
            
            # Train model based on type
            if request.model_type in ["random_forest", "decision_tree", "gradient_boosting", "logistic_regression"]:
                pipeline = EVClassificationPipeline()
                result = pipeline.train_and_evaluate(X, y, request.model_type, request.test_size)
            elif request.model_type in ["linear_regression", "random_forest_regressor"]:
                pipeline = EVRegressionPipeline()
                result = pipeline.train_and_evaluate(X, y, request.model_type, request.test_size)
            elif request.model_type in ["kmeans", "dbscan"]:
                pipeline = EVClusteringPipeline()
                result = pipeline.train_and_evaluate(X, request.model_type)
            else:
                raise HTTPException(status_code=400, detail="Unsupported model type")
            
            # Log metrics
            mlflow.log_metrics(result.get("metrics", {}))
            
            return ModelResponse(
                model_id=run.info.run_id,
                model_type=request.model_type,
                metrics=result.get("metrics", {}),
                status="completed",
                created_at=datetime.utcnow().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/list")
async def list_models():
    """List all trained models."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
        
        if not experiment:
            return {"models": []}
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=50
        )
        
        models = []
        for run in runs:
            models.append({
                "model_id": run.info.run_id,
                "model_type": run.data.params.get("model_type", "unknown"),
                "status": run.info.status,
                "created_at": run.info.start_time,
                "metrics": run.data.metrics,
                "parameters": run.data.params
            })
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using a trained model."""
    try:
        # Load model from MLflow
        model_uri = f"runs:/{request.model_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Prepare input data
        input_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_df)[0].tolist()
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_id=request.model_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get detailed metrics for a specific model."""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(model_id)
        
        return {
            "model_id": model_id,
            "metrics": run.data.metrics,
            "parameters": run.data.params,
            "tags": run.data.tags,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time
        }
        
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=404, detail="Model not found")


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        client.delete_run(model_id)
        
        return {"message": f"Model {model_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")