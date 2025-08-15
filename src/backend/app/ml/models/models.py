"""
SQLAlchemy ORM models for the Electric Vehicle Analysis platform.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Dataset(Base):
    """Model for storing uploaded datasets."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)  # in bytes
    record_count = Column(Integer)
    column_count = Column(Integer)
    columns_metadata = Column(JSON)  # Store column names and types
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    processed_date = Column(DateTime(timezone=True))
    status = Column(String(50), default="uploaded")  # uploaded, processing, processed, failed
    uploaded_by = Column(String(255))
    processing_errors = Column(Text)
    statistics = Column(JSON)  # Store basic statistics about the dataset
    
    # Relationships
    models = relationship("Model", back_populates="dataset", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="dataset")
    
    __table_args__ = (
        Index('idx_dataset_status', 'status'),
        Index('idx_dataset_upload_date', 'upload_date'),
    )


class Model(Base):
    """Model for storing trained ML models."""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=False)  # random_forest, gradient_boosting, etc.
    algorithm_family = Column(String(50))  # classification, regression, clustering
    version = Column(String(50), default="1.0")
    
    # Model file information
    model_path = Column(String(500))  # Path to serialized model
    model_size = Column(Integer)  # Size in bytes
    
    # Training information
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    training_date = Column(DateTime(timezone=True), server_default=func.now())
    training_duration = Column(Float)  # in seconds
    training_params = Column(JSON)  # Hyperparameters used
    feature_names = Column(JSON)  # List of feature names used
    target_variable = Column(String(255))
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    mse = Column(Float)
    mae = Column(Float)
    confusion_matrix = Column(JSON)
    feature_importance = Column(JSON)
    cross_validation_scores = Column(JSON)
    
    # Model metadata
    is_active = Column(Boolean, default=True)
    is_deployed = Column(Boolean, default=False)
    deployment_date = Column(DateTime(timezone=True))
    mlflow_run_id = Column(String(255))
    mlflow_experiment_id = Column(String(255))
    created_by = Column(String(255))
    description = Column(Text)
    tags = Column(JSON)  # Additional tags for categorization
    
    # Relationships
    dataset = relationship("Dataset", back_populates="models")
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="model")
    
    __table_args__ = (
        Index('idx_model_type', 'model_type'),
        Index('idx_model_active', 'is_active'),
        Index('idx_model_training_date', 'training_date'),
    )


class Prediction(Base):
    """Model for storing predictions made by models."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Model information
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    model_version = Column(String(50))
    
    # Input data
    input_data = Column(JSON, nullable=False)  # Features used for prediction
    dataset_id = Column(Integer, ForeignKey("datasets.id"))  # Optional reference to dataset
    
    # Prediction results
    prediction_value = Column(JSON)  # Can be single value or array
    prediction_class = Column(String(255))  # For classification
    prediction_probability = Column(Float)  # Confidence score
    prediction_probabilities = Column(JSON)  # All class probabilities
    
    # Metadata
    prediction_date = Column(DateTime(timezone=True), server_default=func.now())
    processing_time = Column(Float)  # in milliseconds
    batch_id = Column(String(255))  # For batch predictions
    user_id = Column(String(255))
    feedback = Column(String(50))  # correct, incorrect, null
    feedback_date = Column(DateTime(timezone=True))
    feedback_notes = Column(Text)
    
    # Relationships
    model = relationship("Model", back_populates="predictions")
    dataset = relationship("Dataset", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_date', 'prediction_date'),
        Index('idx_prediction_batch', 'batch_id'),
        Index('idx_prediction_user', 'user_id'),
    )


class Experiment(Base):
    """Model for tracking ML experiments."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Experiment configuration
    model_id = Column(Integer, ForeignKey("models.id"))
    hypothesis = Column(Text)
    experiment_type = Column(String(100))  # hyperparameter_tuning, feature_engineering, etc.
    parameters = Column(JSON)  # Experiment parameters
    
    # Results
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    duration = Column(Float)  # in seconds
    status = Column(String(50), default="running")  # running, completed, failed, cancelled
    results = Column(JSON)  # Experiment results and metrics
    best_params = Column(JSON)  # Best parameters found
    
    # Tracking
    mlflow_experiment_id = Column(String(255))
    mlflow_runs = Column(JSON)  # List of MLflow run IDs
    created_by = Column(String(255))
    notes = Column(Text)
    
    # Relationships
    model = relationship("Model", back_populates="experiments")
    
    __table_args__ = (
        Index('idx_experiment_status', 'status'),
        Index('idx_experiment_start', 'start_time'),
    )


class User(Base):
    """Model for user management."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    
    # User status
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # User preferences
    preferences = Column(JSON)  # UI preferences, default models, etc.
    api_key = Column(String(255), unique=True, index=True)
    api_key_created = Column(DateTime(timezone=True))
    
    # Usage tracking
    total_predictions = Column(Integer, default=0)
    total_models_trained = Column(Integer, default=0)
    usage_quota = Column(Integer, default=1000)  # Monthly quota
    current_usage = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_user_active', 'is_active'),
        Index('idx_user_created', 'created_at'),
    )


class Job(Base):
    """Model for tracking background jobs."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, nullable=False, index=True)
    job_type = Column(String(100), nullable=False)  # training, batch_prediction, data_processing
    
    # Job details
    status = Column(String(50), default="pending")  # pending, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)  # 0-100
    total_items = Column(Integer)
    processed_items = Column(Integer, default=0)
    
    # Configuration
    configuration = Column(JSON)  # Job-specific configuration
    input_data = Column(JSON)  # Input parameters
    output_data = Column(JSON)  # Results or output location
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration = Column(Float)  # in seconds
    
    # Error handling
    error_message = Column(Text)
    error_traceback = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # User tracking
    user_id = Column(String(255))
    priority = Column(Integer, default=5)  # 1-10, higher is more important
    
    __table_args__ = (
        Index('idx_job_status', 'status'),
        Index('idx_job_type', 'job_type'),
        Index('idx_job_created', 'created_at'),
        Index('idx_job_user', 'user_id'),
    )


class DataQuality(Base):
    """Model for tracking data quality metrics."""
    __tablename__ = "data_quality"

    id = Column(Integer, primary_key=True, index=True)
    check_id = Column(String(255), unique=True, nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Quality checks
    check_date = Column(DateTime(timezone=True), server_default=func.now())
    check_type = Column(String(100))  # completeness, consistency, accuracy, validity
    
    # Results
    total_records = Column(Integer)
    valid_records = Column(Integer)
    invalid_records = Column(Integer)
    missing_values = Column(JSON)  # Per column missing value counts
    outliers = Column(JSON)  # Detected outliers
    duplicates = Column(Integer)
    
    # Metrics
    completeness_score = Column(Float)  # 0-1
    consistency_score = Column(Float)  # 0-1
    validity_score = Column(Float)  # 0-1
    overall_score = Column(Float)  # 0-1
    
    # Issues found
    issues = Column(JSON)  # List of issues with severity
    recommendations = Column(JSON)  # Suggested fixes
    
    __table_args__ = (
        Index('idx_quality_check_date', 'check_date'),
        Index('idx_quality_dataset', 'dataset_id'),
    )


class APILog(Base):
    """Model for API request logging."""
    __tablename__ = "api_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Request information
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    request_body = Column(JSON)
    request_headers = Column(JSON)
    
    # Response information
    status_code = Column(Integer)
    response_body = Column(JSON)
    response_time = Column(Float)  # in milliseconds
    
    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(String(255))
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    
    # Error tracking
    error_message = Column(Text)
    error_type = Column(String(100))
    
    __table_args__ = (
        Index('idx_api_timestamp', 'timestamp'),
        Index('idx_api_endpoint', 'endpoint'),
        Index('idx_api_user', 'user_id'),
        Index('idx_api_status', 'status_code'),
    )