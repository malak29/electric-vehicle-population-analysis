"""Unit tests for ML models."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from app.ml.models.classification import EVClassificationPipeline
from app.ml.models.regression import EVRegressionPipeline
from app.ml.models.clustering import EVClusteringPipeline


class TestEVClassificationPipeline:
    """Test cases for EV classification pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        X_train = pd.DataFrame({
            'Electric Range': np.random.normal(100, 30, n_samples),
            'Model Year': np.random.randint(2015, 2024, n_samples),
            'Base MSRP': np.random.normal(40000, 15000, n_samples)
        })
        
        X_test = pd.DataFrame({
            'Electric Range': np.random.normal(100, 30, 20),
            'Model Year': np.random.randint(2015, 2024, 20),
            'Base MSRP': np.random.normal(40000, 15000, 20)
        })
        
        y_train = pd.Series(np.random.choice(['BEV', 'PHEV'], n_samples))
        y_test = pd.Series(np.random.choice(['BEV', 'PHEV'], 20))
        
        return X_train, X_test, y_train, y_test
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = EVClassificationPipeline()
        assert 'random_forest' in pipeline.models
        assert 'logistic_regression' in pipeline.models
        assert 'decision_tree' in pipeline.models
        assert 'gradient_boosting' in pipeline.models
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.sklearn.log_model')
    def test_train_and_evaluate(self, mock_log_model, mock_log_metrics, 
                               mock_log_params, mock_start_run, sample_data):
        """Test model training and evaluation."""
        X_train, X_test, y_train, y_test = sample_data
        pipeline = EVClassificationPipeline()
        
        # Mock MLflow context manager
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        result = pipeline.train_and_evaluate(
            X_train, X_test, y_train, y_test, 'random_forest'
        )
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'predictions' in result
        assert result['metrics']['accuracy'] >= 0
        assert result['metrics']['accuracy'] <= 1
    
    def test_invalid_model_name(self, sample_data):
        """Test handling of invalid model names."""
        X_train, X_test, y_train, y_test = sample_data
        pipeline = EVClassificationPipeline()
        
        with pytest.raises(ValueError, match="Unknown model"):
            pipeline.train_and_evaluate(
                X_train, X_test, y_train, y_test, 'invalid_model'
            )


class TestEVRegressionPipeline:
    """Test cases for EV regression pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        X_train = pd.DataFrame({
            'Model Year': np.random.randint(2015, 2024, n_samples),
            'Base MSRP': np.random.normal(40000, 15000, n_samples),
            'Battery Capacity': np.random.normal(50, 20, n_samples)
        })
        
        X_test = pd.DataFrame({
            'Model Year': np.random.randint(2015, 2024, 20),
            'Base MSRP': np.random.normal(40000, 15000, 20),
            'Battery Capacity': np.random.normal(50, 20, 20)
        })
        
        # Generate target variable with some correlation
        y_train = pd.Series(
            X_train['Battery Capacity'] * 2 + np.random.normal(0, 10, n_samples)
        )
        y_test = pd.Series(
            X_test['Battery Capacity'] * 2 + np.random.normal(0, 10, 20)
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = EVRegressionPipeline()
        assert 'linear_regression' in pipeline.models
        assert 'random_forest' in pipeline.models
        assert 'ridge' in pipeline.models
        assert 'lasso' in pipeline.models
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.sklearn.log_model')
    def test_train_and_evaluate(self, mock_log_model, mock_log_metrics,
                               mock_log_params, mock_start_run, sample_data):
        """Test regression model training and evaluation."""
        X_train, X_test, y_train, y_test = sample_data
        pipeline = EVRegressionPipeline()
        
        # Mock MLflow context manager
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        result = pipeline.train_and_evaluate(
            X_train, X_test, y_train, y_test, 'linear_regression'
        )
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'test_predictions' in result
        assert 'residuals' in result
        assert isinstance(result['metrics']['test_r2'], float)


class TestEVClusteringPipeline:
    """Test cases for EV clustering pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'Electric Range': np.random.normal(100, 30, n_samples),
            'Base MSRP': np.random.normal(40000, 15000, n_samples),
            'Model Year': np.random.randint(2015, 2024, n_samples)
        })
        
        return X
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = EVClusteringPipeline()
        assert 'kmeans' in pipeline.models
        assert 'dbscan' in pipeline.models
        assert 'agglomerative' in pipeline.models
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.sklearn.log_model')
    def test_train_and_evaluate(self, mock_log_model, mock_log_metrics,
                               mock_log_params, mock_start_run, sample_data):
        """Test clustering model training and evaluation."""
        pipeline = EVClusteringPipeline()
        
        # Mock MLflow context manager
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        result = pipeline.train_and_evaluate(sample_data, 'kmeans')
        
        assert 'model' in result
        assert 'labels' in result
        assert 'metrics' in result
        assert 'cluster_analysis' in result
        assert len(result['labels']) == len(sample_data)
    
    def test_find_optimal_clusters(self, sample_data):
        """Test optimal cluster finding."""
        pipeline = EVClusteringPipeline()
        result = pipeline.find_optimal_clusters(sample_data, max_clusters=5)
        
        assert 'optimal_k_elbow' in result
        assert 'optimal_k_silhouette' in result
        assert 'silhouette_scores' in result
        assert len(result['n_clusters']) == 4  # 2 to 5 clusters


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('mlflow.start_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.sklearn.log_model'):
        yield


class TestModelMetrics:
    """Test model evaluation metrics."""
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        pipeline = EVClassificationPipeline()
        
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], 
                                [0.1, 0.9], [0.9, 0.1], [0.6, 0.4]])
        
        metrics = pipeline._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        pipeline = EVRegressionPipeline()
        
        y_train_true = pd.Series([1, 2, 3, 4, 5])
        y_train_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_test_true = pd.Series([6, 7, 8])
        y_test_pred = np.array([5.9, 7.1, 7.8])
        
        metrics = pipeline._calculate_regression_metrics(
            y_train_true, y_train_pred, y_test_true, y_test_pred
        )
        
        assert 'test_r2' in metrics
        assert 'test_mse' in metrics
        assert 'test_mae' in metrics
        assert 'test_rmse' in metrics
        assert metrics['test_r2'] <= 1