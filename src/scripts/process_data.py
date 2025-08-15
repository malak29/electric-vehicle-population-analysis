#!/usr/bin/env python3
"""
Data processing script for Electric Vehicle Analysis.

This script downloads, cleans, and preprocesses the electric vehicle dataset
for use in the ML models.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import settings
from app.ml.preprocessing.data_cleaning import EVDataProcessor
from app.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


class DataProcessor:
    """Main data processing class."""
    
    def __init__(self):
        self.processor = EVDataProcessor()
        self.data_path = Path(settings.DATA_PATH)
        
    def download_dataset(self) -> None:
        """Download the electric vehicle dataset."""
        logger.info("Downloading electric vehicle dataset...")
        
        # Washington State Electric Vehicle Population Data
        url = "https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD"
        
        try:
            df = pd.read_csv(url)
            logger.info(f"Downloaded dataset with {len(df)} records")
            
            # Save raw data
            raw_path = self.data_path / "raw" / "electric_vehicles_raw.csv"
            df.to_csv(raw_path, index=False)
            logger.info(f"Raw data saved to: {raw_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def clean_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the dataset."""
        logger.info("Starting data cleaning and processing...")
        
        # Basic cleaning
        df_clean = self.processor.clean_data(df)
        
        # Feature engineering
        df_features = self.processor.feature_engineering(df_clean)
        
        # Data quality report
        quality_report = self._generate_quality_report(df_features)
        logger.info(f"Data quality report: {quality_report}")
        
        return df_features
    
    def _generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a data quality report."""
        report = {
            "total_records": len(df),
            "total_features": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.value_counts().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Feature-specific quality checks
        if 'Electric Range' in df.columns:
            report['range_statistics'] = {
                'min': df['Electric Range'].min(),
                'max': df['Electric Range'].max(),
                'mean': df['Electric Range'].mean(),
                'zero_range_count': (df['Electric Range'] == 0).sum()
            }
        
        if 'Base MSRP' in df.columns:
            report['price_statistics'] = {
                'min': df['Base MSRP'].min(),
                'max': df['Base MSRP'].max(),
                'mean': df['Base MSRP'].mean(),
                'zero_price_count': (df['Base MSRP'] == 0).sum()
            }
        
        return report
    
    def save_processed_data(self, df: pd.DataFrame) -> None:
        """Save processed data to multiple formats."""
        processed_path = self.data_path / "processed"
        
        # Save as CSV
        csv_path = processed_path / "electric_vehicles.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Processed data saved to: {csv_path}")
        
        # Save as Parquet for better performance
        parquet_path = processed_path / "electric_vehicles.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Processed data saved to: {parquet_path}")
        
        # Save summary statistics
        summary_path = processed_path / "data_summary.json"
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": df.nunique().to_dict(),
        }
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Data summary saved to: {summary_path}")
    
    def create_sample_datasets(self, df: pd.DataFrame) -> None:
        """Create sample datasets for testing and development."""
        logger.info("Creating sample datasets...")
        
        sample_path = self.data_path / "processed"
        
        # Small sample for quick testing (1000 records)
        sample_small = df.sample(n=min(1000, len(df)), random_state=42)
        sample_small.to_csv(sample_path / "electric_vehicles_sample_small.csv", index=False)
        
        # Medium sample for development (10000 records)
        sample_medium = df.sample(n=min(10000, len(df)), random_state=42)
        sample_medium.to_csv(sample_path / "electric_vehicles_sample_medium.csv", index=False)
        
        logger.info("Sample datasets created")
    
    def validate_processed_data(self, df: pd.DataFrame) -> bool:
        """Validate the processed dataset."""
        logger.info("Validating processed data...")
        
        # Check for required columns
        required_columns = [
            'Make', 'Model', 'Model Year', 'Electric Vehicle Type', 'Electric Range'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data ranges
        if 'Model Year' in df.columns:
            if df['Model Year'].min() < 2000 or df['Model Year'].max() > 2030:
                logger.warning("Model Year values outside expected range")
        
        if 'Electric Range' in df.columns:
            if df['Electric Range'].min() < 0 or df['Electric Range'].max() > 1000:
                logger.warning("Electric Range values outside expected range")
        
        logger.info("Data validation completed")
        return True


def main():
    """Main processing function."""
    try:
        processor = DataProcessor()
        
        # Check if raw data exists, if not download it
        raw_file = processor.data_path / "raw" / "electric_vehicles_raw.csv"
        
        if raw_file.exists():
            logger.info("Loading existing raw data...")
            df = pd.read_csv(raw_file)
        else:
            logger.info("Raw data not found, downloading...")
            df = processor.download_dataset()
        
        # Process data
        df_processed = processor.clean_and_process(df)
        
        # Validate processed data
        if not processor.validate_processed_data(df_processed):
            logger.error("Data validation failed")
            sys.exit(1)
        
        # Save processed data
        processor.save_processed_data(df_processed)
        
        # Create sample datasets
        processor.create_sample_datasets(df_processed)
        
        logger.info("Data processing completed successfully! ✅")
        logger.info(f"Final dataset shape: {df_processed.shape}")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()