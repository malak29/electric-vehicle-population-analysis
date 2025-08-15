"""Configuration settings for the application."""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Electric Vehicle Analysis API"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    API_V1_STR: str = "/api/v1"
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost:5432/ev_analysis",
        env="DATABASE_URL"
    )
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="ev_analysis",
        env="MLFLOW_EXPERIMENT_NAME"
    )
    
    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="BACKEND_CORS_ORIGINS"
    )
    
    # Data paths
    DATA_PATH: str = Field(default="./data", env="DATA_PATH")
    MODEL_PATH: str = Field(default="./models", env="MODEL_PATH")
    UPLOAD_PATH: str = Field(default="./uploads", env="UPLOAD_PATH")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="app.log", env="LOG_FILE")
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()