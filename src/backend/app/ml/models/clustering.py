"""Clustering models for electric vehicle analysis."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import mlflow
import mlflow.sklearn
from datetime import datetime

from app.utils.logger import get_logger

logger = get_logger(__name__)


class EVClusteringPipeline:
    """Electric Vehicle clustering pipeline."""
    
    def __init__(self):
        self.models = {
            'kmeans': KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=5
            ),
            'agglomerative': AgglomerativeClustering(
                n_clusters=3,
                linkage='ward'
            )
        }
        self.scaler = StandardScaler()
        
    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        model_name: str,
        hyperparameters: Dict[str, Any] = None,
        n_clusters: int = None
    ) -> Dict[str, Any]:
        """
        Train and evaluate a clustering model.
        
        Args:
            X: Feature matrix
            model_name: Name of the clustering algorithm
            hyperparameters: Optional hyperparameters
            n_clusters: Number of clusters (for applicable algorithms)
            
        Returns:
            Dictionary containing clustering results and metrics
        """
        logger.info(f"Training {model_name} clustering model")
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        
        # Get model and apply hyperparameters
        model = self.models[model_name]
        if hyperparameters:
            model.set_params(**hyperparameters)
        
        if n_clusters and hasattr(model, 'n_clusters'):
            model.set_params(n_clusters=n_clusters)
        
        # Train model
        start_time = datetime.now()
        labels = model.fit_predict(X_scaled)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        metrics = self._calculate_clustering_metrics(X_scaled, labels)
        metrics['training_time'] = training_time
        metrics['n_clusters_found'] = len(np.unique(labels))
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        # Create cluster analysis
        cluster_analysis = self._analyze_clusters(X, labels)
        
        logger.info(f"Clustering model {model_name} trained successfully")
        
        return {
            "model": model,
            "labels": labels.tolist(),
            "metrics": metrics,
            "pca_coordinates": X_pca.tolist(),
            "cluster_analysis": cluster_analysis,
            "scaler": self.scaler
        }
    
    def _calculate_clustering_metrics(
        self, 
        X: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering evaluation metrics."""
        metrics = {}
        
        # Only calculate metrics if we have more than one cluster
        n_clusters = len(np.unique(labels))
        if n_clusters > 1 and -1 not in labels:  # -1 indicates noise in DBSCAN
            metrics['silhouette_score'] = silhouette_score(X, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        
        # Count noise points (for DBSCAN)
        if -1 in labels:
            metrics['noise_points'] = np.sum(labels == -1)
            metrics['noise_percentage'] = (np.sum(labels == -1) / len(labels)) * 100
        
        return metrics
    
    def _analyze_clusters(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        analysis = {}
        
        # Add cluster labels to dataframe
        X_analysis = X.copy()
        X_analysis['cluster'] = labels
        
        # Cluster sizes
        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        analysis['cluster_sizes'] = cluster_sizes
        
        # Cluster centroids (mean values)
        centroids = {}
        for cluster_id in np.unique(labels):
            if cluster_id != -1:  # Exclude noise points
                cluster_data = X_analysis[X_analysis['cluster'] == cluster_id]
                centroids[int(cluster_id)] = cluster_data.select_dtypes(include=[np.number]).mean().to_dict()
        
        analysis['centroids'] = centroids
        
        # Feature statistics per cluster
        feature_stats = {}
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for cluster_id in np.unique(labels):
            if cluster_id != -1:
                cluster_data = X_analysis[X_analysis['cluster'] == cluster_id]
                stats = {}
                for col in numeric_columns:
                    stats[col] = {
                        'mean': float(cluster_data[col].mean()),
                        'std': float(cluster_data[col].std()),
                        'min': float(cluster_data[col].min()),
                        'max': float(cluster_data[col].max())
                    }
                feature_stats[int(cluster_id)] = stats
        
        analysis['feature_statistics'] = feature_stats
        
        return analysis
    
    def find_optimal_clusters(
        self, 
        X: pd.DataFrame, 
        max_clusters: int = 10
    ) -> Dict[str, Any]:
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        logger.info("Finding optimal number of clusters")
        
        X_scaled = self.scaler.fit_transform(X)
        
        results = {
            'n_clusters': [],
            'inertia': [],
            'silhouette_scores': [],
            'calinski_harabasz_scores': []
        }
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            results['n_clusters'].append(k)
            results['inertia'].append(kmeans.inertia_)
            results['silhouette_scores'].append(silhouette_score(X_scaled, labels))
            results['calinski_harabasz_scores'].append(calinski_harabasz_score(X_scaled, labels))
        
        # Find optimal k using elbow method
        inertias = results['inertia']
        diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        second_diffs = [diffs[i] - diffs[i+1] for i in range(len(diffs)-1)]
        optimal_k = np.argmax(second_diffs) + 3  # +3 because we start from k=2
        
        results['optimal_k_elbow'] = optimal_k
        results['optimal_k_silhouette'] = results['n_clusters'][np.argmax(results['silhouette_scores'])]
        
        logger.info(f"Optimal k found: elbow method={optimal_k}, silhouette={results['optimal_k_silhouette']}")
        
        return results