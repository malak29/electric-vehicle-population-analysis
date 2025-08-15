"""Classification models for electric vehicle analysis."""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import mlflow
import mlflow.sklearn
from datetime import datetime

from app.utils.logger import get_logger

logger = get_logger(__name__)


class EVClassificationPipeline:
    """Electric Vehicle classification pipeline."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
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
        Train and evaluate a classification model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            model_name: Name of the model to train
            hyperparameters: Optional hyperparameters to override defaults
            
        Returns:
            Dictionary containing evaluation metrics and model artifacts
        """
        logger.info(f"Training {model_name} classification model")
        
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
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        )
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        if feature_importance:
            mlflow.log_dict(feature_importance, "feature_importance.json")
        
        logger.info(f"Model {model_name} trained successfully with accuracy: {metrics['accuracy']:.4f}")
        
        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "predictions": y_pred.tolist(),
            "probabilities": y_pred_proba.tolist() if y_pred_proba is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro')
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        return metrics
    
    def predict(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with a trained model."""
        predictions = model.predict(X)
        
        result = {"predictions": predictions.tolist()}
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            result["probabilities"] = probabilities.tolist()
        
        return result
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from a trained model."""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        else:
            return {}