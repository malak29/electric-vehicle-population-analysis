"""
MLflow client for experiment tracking and model registry.
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from typing import Dict, Any, Optional, List
import logging
import json
import pandas as pd
from datetime import datetime
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configure MLflow
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

# Initialize MLflow client
client = MlflowClient()


class MLflowManager:
    """
    Manager class for MLflow operations.
    """
    
    def __init__(self):
        """Initialize MLflow manager."""
        self.client = MlflowClient()
        self.tracking_uri = settings.MLFLOW_TRACKING_URI
        self.default_experiment = settings.MLFLOW_EXPERIMENT_NAME
    
    def create_experiment(self, name: str, description: str = None) -> str:
        """
        Create new MLflow experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
        
        Returns:
            str: Experiment ID
        """
        try:
            experiment_id = self.client.create_experiment(
                name=name,
                artifact_location=f"{self.tracking_uri}/artifacts/{name}",
                tags={"description": description} if description else {}
            )
            logger.info(f"Created experiment: {name} with ID: {experiment_id}")
            return experiment_id
        except Exception as e:
            # Experiment might already exist
            experiment = self.client.get_experiment_by_name(name)
            if experiment:
                return experiment.experiment_id
            logger.error(f"Failed to create experiment: {str(e)}")
            raise
    
    def start_run(
        self,
        experiment_name: str = None,
        run_name: str = None,
        tags: Dict[str, str] = None
    ) -> mlflow.ActiveRun:
        """
        Start MLflow run.
        
        Args:
            experiment_name: Experiment name
            run_name: Run name
            tags: Run tags
        
        Returns:
            ActiveRun: MLflow run object
        """
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        run = mlflow.start_run(run_name=run_name)
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Set default tags
        mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
        mlflow.set_tag("environment", settings.ENVIRONMENT)
        
        return run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Parameters to log
        """
        for key, value in params.items():
            # MLflow has a limit on param value length
            if isinstance(value, (dict, list)):
                value = json.dumps(value)[:250]
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Metrics to log
            step: Optional step number
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model,
        artifact_path: str,
        model_type: str = "sklearn",
        signature=None,
        input_example=None,
        registered_model_name: str = None
    ):
        """
        Log model to MLflow.
        
        Args:
            model: Model object
            artifact_path: Path to save model
            model_type: Type of model (sklearn, tensorflow, pytorch)
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry
        """
        # Infer signature if not provided
        if signature is None and input_example is not None:
            if isinstance(input_example, pd.DataFrame):
                signature = infer_signature(input_example)
        
        # Log model based on type
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif model_type == "tensorflow":
            mlflow.tensorflow.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        else:
            # Generic model logging
            mlflow.pyfunc.log_model(
                artifact_path,
                python_model=model,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Log artifact to MLflow.
        
        Args:
            local_path: Local file path
            artifact_path: Artifact path in MLflow
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """
        Log directory of artifacts to MLflow.
        
        Args:
            local_dir: Local directory path
            artifact_path: Artifact path in MLflow
        """
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def end_run(self, status: str = "FINISHED"):
        """
        End MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
    
    def get_run(self, run_id: str) -> mlflow.entities.Run:
        """
        Get MLflow run by ID.
        
        Args:
            run_id: Run ID
        
        Returns:
            Run: MLflow run object
        """
        return self.client.get_run(run_id)
    
    def search_runs(
        self,
        experiment_ids: List[str] = None,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[mlflow.entities.Run]:
        """
        Search MLflow runs.
        
        Args:
            experiment_ids: List of experiment IDs
            filter_string: Filter query string
            max_results: Maximum results to return
        
        Returns:
            List[Run]: List of runs
        """
        return self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results
        )
    
    def load_model(self, model_uri: str):
        """
        Load model from MLflow.
        
        Args:
            model_uri: Model URI (runs:/run_id/model or models:/model_name/version)
        
        Returns:
            Loaded model
        """
        return mlflow.pyfunc.load_model(model_uri)
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Dict[str, str] = None
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_uri: Model URI
            name: Registered model name
            tags: Model tags
        
        Returns:
            ModelVersion: Registered model version
        """
        # Register model
        model_version = mlflow.register_model(model_uri, name)
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name,
                    model_version.version,
                    key,
                    value
                )
        
        return model_version
    
    def transition_model_stage(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing_versions: bool = True
    ):
        """
        Transition model version to new stage.
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Archive existing versions in target stage
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
    
    def get_model_version(
        self,
        name: str,
        version: int
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Get specific model version.
        
        Args:
            name: Model name
            version: Model version
        
        Returns:
            ModelVersion: Model version object
        """
        return self.client.get_model_version(name, version)
    
    def get_latest_model_version(
        self,
        name: str,
        stages: List[str] = None
    ) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """
        Get latest model version.
        
        Args:
            name: Model name
            stages: Filter by stages
        
        Returns:
            ModelVersion: Latest model version or None
        """
        versions = self.client.get_latest_versions(name, stages=stages)
        return versions[0] if versions else None
    
    def delete_run(self, run_id: str):
        """
        Delete MLflow run.
        
        Args:
            run_id: Run ID to delete
        """
        self.client.delete_run(run_id)
    
    def clean_old_runs(self, experiment_id: str, keep_last: int = 10):
        """
        Clean old runs from experiment.
        
        Args:
            experiment_id: Experiment ID
            keep_last: Number of recent runs to keep
        """
        runs = self.search_runs(
            experiment_ids=[experiment_id],
            max_results=1000
        )
        
        # Sort by start time
        runs.sort(key=lambda x: x.info.start_time, reverse=True)
        
        # Delete old runs
        for run in runs[keep_last:]:
            self.delete_run(run.info.run_id)
            logger.info(f"Deleted old run: {run.info.run_id}")


# Global MLflow manager instance
mlflow_manager = MLflowManager()


def track_model_training(
    model_name: str,
    model_type: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model_object=None,
    dataset_info: Dict[str, Any] = None
) -> str:
    """
    Track model training with MLflow.
    
    Args:
        model_name: Model name
        model_type: Model type
        params: Training parameters
        metrics: Model metrics
        model_object: Trained model object
        dataset_info: Dataset information
    
    Returns:
        str: Run ID
    """
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Log parameters
        mlflow_manager.log_params(params)
        
        # Log metrics
        mlflow_manager.log_metrics(metrics)
        
        # Log dataset info
        if dataset_info:
            for key, value in dataset_info.items():
                mlflow.set_tag(f"dataset.{key}", str(value))
        
        # Log model
        if model_object:
            mlflow_manager.log_model(
                model_object,
                artifact_path="model",
                model_type=model_type,
                registered_model_name=model_name
            )
        
        # Log additional metadata
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("model_name", model_name)
        
        return run.info.run_id