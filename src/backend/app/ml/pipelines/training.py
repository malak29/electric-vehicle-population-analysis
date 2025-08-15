"""ML training pipeline for electric vehicle analysis."""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer

from app.ml.models.classification import EVClassificationPipeline
from app.ml.models.regression import EVRegressionPipeline
from app.ml.models.clustering import EVClusteringPipeline
from app.ml.preprocessing.data_cleaning import EVDataProcessor
from app.ml.evaluation.metrics import ModelEvaluator
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Main training pipeline for EV analysis models."""
    
    def __init__(self):
        self.data_processor = EVDataProcessor()
        self.evaluator = ModelEvaluator()
        self.classification_pipeline = EVClassificationPipeline()
        self.regression_pipeline = EVRegressionPipeline()
        self.clustering_pipeline = EVClusteringPipeline()
        
    async def train_classification_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        use_hyperparameter_tuning: bool = True
    ) -> Dict[str, Any]:
        """Train a classification model with optional hyperparameter tuning."""
        logger.info(f"Training classification model: {model_type}")
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.data_processor.prepare_for_modeling(
                data, target_column, feature_columns
            )
            
            # Hyperparameter tuning if requested
            if use_hyperparameter_tuning and hyperparameters:
                best_params = await self._tune_hyperparameters(
                    X_train, y_train, model_type, hyperparameters
                )
                hyperparameters = best_params
            
            # Train and evaluate
            with mlflow.start_run():
                mlflow.set_tag("model_category", "classification")
                mlflow.set_tag("dataset_size", len(data))
                
                result = self.classification_pipeline.train_and_evaluate(
                    X_train, X_test, y_train, y_test, model_type, hyperparameters
                )
                
                # Additional evaluation
                evaluation_results = self.evaluator.evaluate_classification_model(
                    result["model"], X_test, y_test
                )
                result.update(evaluation_results)
                
                logger.info(f"Classification model trained successfully. Accuracy: {result['metrics']['accuracy']:.4f}")
                return result
                
        except Exception as e:
            logger.error(f"Error training classification model: {e}")
            raise
    
    async def train_regression_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        use_hyperparameter_tuning: bool = True
    ) -> Dict[str, Any]:
        """Train a regression model with optional hyperparameter tuning."""
        logger.info(f"Training regression model: {model_type}")
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.data_processor.prepare_for_modeling(
                data, target_column, feature_columns
            )
            
            # Hyperparameter tuning if requested
            if use_hyperparameter_tuning and hyperparameters:
                best_params = await self._tune_hyperparameters(
                    X_train, y_train, model_type, hyperparameters
                )
                hyperparameters = best_params
            
            # Train and evaluate
            with mlflow.start_run():
                mlflow.set_tag("model_category", "regression")
                mlflow.set_tag("dataset_size", len(data))
                
                result = self.regression_pipeline.train_and_evaluate(
                    X_train, X_test, y_train, y_test, model_type, hyperparameters
                )
                
                # Additional evaluation
                evaluation_results = self.evaluator.evaluate_regression_model(
                    result["model"], X_test, y_test
                )
                result.update(evaluation_results)
                
                logger.info(f"Regression model trained successfully. R²: {result['metrics']['test_r2']:.4f}")
                return result
                
        except Exception as e:
            logger.error(f"Error training regression model: {e}")
            raise
    
    async def train_clustering_model(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train a clustering model."""
        logger.info(f"Training clustering model: {model_type}")
        
        try:
            # Prepare features
            X = data[feature_columns]
            X_encoded = self.data_processor.encode_categorical_features(X)
            
            with mlflow.start_run():
                mlflow.set_tag("model_category", "clustering")
                mlflow.set_tag("dataset_size", len(data))
                
                result = self.clustering_pipeline.train_and_evaluate(
                    X_encoded, model_type, hyperparameters
                )
                
                logger.info(f"Clustering model trained successfully. Clusters found: {result['metrics']['n_clusters_found']}")
                return result
                
        except Exception as e:
            logger.error(f"Error training clustering model: {e}")
            raise
    
    async def _tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str,
        param_grid: Dict[str, Any],
        cv_folds: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV."""
        logger.info(f"Tuning hyperparameters for {model_type}")
        
        try:
            # Get base model
            if model_type in self.classification_pipeline.models:
                base_model = self.classification_pipeline.models[model_type]
            elif model_type in self.regression_pipeline.models:
                base_model = self.regression_pipeline.models[model_type]
                scoring = 'r2'
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', base_model)
            ])
            
            # Adjust parameter names for pipeline
            pipeline_param_grid = {f'model__{k}': v for k, v in param_grid.items()}
            
            # Use RandomizedSearchCV for large parameter spaces
            if len(param_grid) > 3:
                search = RandomizedSearchCV(
                    pipeline,
                    pipeline_param_grid,
                    n_iter=20,
                    cv=cv_folds,
                    scoring=scoring,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                search = GridSearchCV(
                    pipeline,
                    pipeline_param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1
                )
            
            # Perform search
            search.fit(X_train, y_train)
            
            # Extract best parameters for the model (remove 'model__' prefix)
            best_params = {
                k.replace('model__', ''): v 
                for k, v in search.best_params_.items()
                if k.startswith('model__')
            }
            
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best score: {search.best_score_:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise
    
    async def train_ensemble_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        model_types: List[str]
    ) -> Dict[str, Any]:
        """Train an ensemble of models and combine predictions."""
        logger.info(f"Training ensemble with models: {model_types}")
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.data_processor.prepare_for_modeling(
                data, target_column, feature_columns
            )
            
            ensemble_results = {}
            models = {}
            
            with mlflow.start_run():
                mlflow.set_tag("model_category", "ensemble")
                mlflow.set_tag("ensemble_models", ",".join(model_types))
                
                # Train individual models
                for model_type in model_types:
                    logger.info(f"Training ensemble component: {model_type}")
                    
                    model_result = self.classification_pipeline.train_and_evaluate(
                        X_train, X_test, y_train, y_test, model_type
                    )
                    
                    ensemble_results[model_type] = model_result
                    models[model_type] = model_result["model"]
                
                # Combine predictions (simple voting)
                ensemble_predictions = self._combine_predictions(models, X_test)
                
                # Evaluate ensemble
                ensemble_metrics = self.evaluator.evaluate_ensemble(
                    y_test, ensemble_predictions, ensemble_results
                )
                
                mlflow.log_metrics(ensemble_metrics)
                
                return {
                    "ensemble_results": ensemble_results,
                    "ensemble_metrics": ensemble_metrics,
                    "ensemble_predictions": ensemble_predictions
                }
                
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            raise
    
    def _combine_predictions(
        self, 
        models: Dict[str, Any], 
        X_test: pd.DataFrame
    ) -> List[Any]:
        """Combine predictions from multiple models using majority voting."""
        all_predictions = []
        
        for model_name, model in models.items():
            predictions = model.predict(X_test)
            all_predictions.append(predictions)
        
        # Simple majority voting
        all_predictions = np.array(all_predictions)
        ensemble_predictions = []
        
        for i in range(len(X_test)):
            # Get most common prediction
            unique, counts = np.unique(all_predictions[:, i], return_counts=True)
            most_common = unique[np.argmax(counts)]
            ensemble_predictions.append(most_common)
        
        return ensemble_predictions
    
    async def auto_ml_pipeline(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_type: str = "auto"
    ) -> Dict[str, Any]:
        """Automated ML pipeline that selects best model automatically."""
        logger.info("Starting AutoML pipeline")
        
        try:
            # Determine problem type if auto
            if problem_type == "auto":
                problem_type = self._determine_problem_type(data[target_column])
            
            # Select appropriate models based on problem type
            if problem_type == "classification":
                model_types = ["random_forest", "gradient_boosting", "logistic_regression"]
            elif problem_type == "regression":
                model_types = ["random_forest", "linear_regression", "gradient_boosting"]
            else:
                raise ValueError(f"Unsupported problem type: {problem_type}")
            
            # Train multiple models
            results = {}
            best_model = None
            best_score = -float('inf')
            
            feature_columns = [col for col in data.columns if col != target_column]
            
            for model_type in model_types:
                if problem_type == "classification":
                    result = await self.train_classification_model(
                        data, target_column, feature_columns, model_type
                    )
                    score = result["metrics"]["accuracy"]
                else:
                    result = await self.train_regression_model(
                        data, target_column, feature_columns, model_type
                    )
                    score = result["metrics"]["test_r2"]
                
                results[model_type] = result
                
                if score > best_score:
                    best_score = score
                    best_model = model_type
            
            logger.info(f"AutoML completed. Best model: {best_model} with score: {best_score:.4f}")
            
            return {
                "best_model": best_model,
                "best_score": best_score,
                "all_results": results,
                "problem_type": problem_type
            }
            
        except Exception as e:
            logger.error(f"Error in AutoML pipeline: {e}")
            raise
    
    def _determine_problem_type(self, target_series: pd.Series) -> str:
        """Automatically determine if it's a classification or regression problem."""
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        # If less than 10 unique values or less than 5% unique ratio, treat as classification
        if unique_values < 10 or (unique_values / total_values) < 0.05:
            return "classification"
        else:
            return "regression"