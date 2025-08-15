"""Regression models for electric vehicle analysis."""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)
import mlflow
import mlflow.sklearn
from datetime import datetime

from app.utils.logger import get_logger

logger = get_logger(__name__)


class EVRegressionPipeline:
    """Electric Vehicle regression pipeline."""
    
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'lasso': Lasso(
                alpha=1.0,
                random_state=42,
                max_iter=1000
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                random_state=42
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
        }
        
    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        hyperparameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Train and evaluate a regression model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            model_name: Name of the model to train
            hyperparameters: Optional hyperparameters to override defaults
            
        Returns:
            Dictionary containing evaluation metrics and model artifacts
        """
        logger.info(f"Training {model_name} regression model")
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get model and apply hyperparameters if provided
        model = self.models[model_name]
        if hyperparameters:
            model.set_params(**hyperparameters)
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(
            y_train, y_pred_train, y_test, y_pred_test
        )
        metrics['training_time'] = training_time
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='r2'
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models, use coefficients as feature importance
            feature_importance = dict(zip(X_train.columns, np.abs(model.coef_)))
        
        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        if feature_importance:
            mlflow.log_dict(feature_importance, "feature_importance.json")
        
        logger.info(f"Model {model_name} trained successfully with R²: {metrics['test_r2']:.4f}")
        
        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "train_predictions": y_pred_train.tolist(),
            "test_predictions": y_pred_test.tolist(),
            "residuals": {
                "train": (y_train - y_pred_train).tolist(),
                "test": (y_test - y_pred_test).tolist()
            }
        }
    
    def _calculate_regression_metrics(
        self,
        y_train_true: pd.Series,
        y_train_pred: np.ndarray,
        y_test_true: pd.Series,
        y_test_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        metrics = {
            # Training metrics
            'train_mse': mean_squared_error(y_train_true, y_train_pred),
            'train_mae': mean_absolute_error(y_train_true, y_train_pred),
            'train_r2': r2_score(y_train_true, y_train_pred),
            'train_explained_variance': explained_variance_score(y_train_true, y_train_pred),
            
            # Test metrics
            'test_mse': mean_squared_error(y_test_true, y_test_pred),
            'test_mae': mean_absolute_error(y_test_true, y_test_pred),
            'test_r2': r2_score(y_test_true, y_test_pred),
            'test_explained_variance': explained_variance_score(y_test_true, y_test_pred),
            'test_max_error': max_error(y_test_true, y_test_pred),
            
            # RMSE
            'train_rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_true, y_test_pred))
        }
        
        # Calculate MAPE (Mean Absolute Percentage Error) if no zero values
        if not (y_test_true == 0).any():
            mape = np.mean(np.abs((y_test_true - y_test_pred) / y_test_true)) * 100
            metrics['test_mape'] = mape
        
        return metrics
    
    def predict(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with a trained model."""
        predictions = model.predict(X)
        
        result = {"predictions": predictions.tolist()}
        
        # Add prediction intervals for some models
        if hasattr(model, 'predict') and model.__class__.__name__ in ['RandomForestRegressor']:
            # Calculate prediction intervals using quantile regression
            # This is a simplified approach - in production, you might want more sophisticated methods
            try:
                residuals = getattr(model, '_residuals', None)
                if residuals is not None:
                    std_residual = np.std(residuals)
                    result["prediction_intervals"] = {
                        "lower": (predictions - 1.96 * std_residual).tolist(),
                        "upper": (predictions + 1.96 * std_residual).tolist()
                    }
            except Exception as e:
                logger.warning(f"Could not calculate prediction intervals: {e}")
        
        return result
    
    def analyze_residuals(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze model residuals for diagnostics."""
        residuals = y_true - y_pred
        
        analysis = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
            "q25": float(residuals.quantile(0.25)),
            "q50": float(residuals.quantile(0.50)),
            "q75": float(residuals.quantile(0.75)),
            "skewness": float(residuals.skew()),
            "kurtosis": float(residuals.kurtosis())
        }
        
        # Test for normality (simplified)
        analysis["appears_normal"] = abs(analysis["skewness"]) < 1 and abs(analysis["kurtosis"]) < 3
        
        return analysis