"""Data cleaning and preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import logging

from app.utils.logger import get_logger

logger = get_logger(__name__)


class EVDataProcessor:
    """Electric Vehicle data processor for cleaning and feature engineering."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.one_hot_encoders: Dict[str, OneHotEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        
    def validate_ev_dataset(self, df: pd.DataFrame) -> bool:
        """Validate that the dataset has required columns for EV analysis."""
        required_columns = [
            'Make', 'Model', 'Model Year', 'Electric Vehicle Type'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the electric vehicle dataset."""
        logger.info("Starting data cleaning process")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df_clean)} duplicate rows")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Clean specific EV columns
        df_clean = self._clean_ev_specific_columns(df_clean)
        
        # Validate data ranges
        df_clean = self._validate_data_ranges(df_clean)
        
        logger.info(f"Data cleaning completed. Final dataset size: {len(df_clean)} rows")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Numeric columns - fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_value}")
        
        # Categorical columns - fill with mode or 'Unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                df[col].fillna(fill_value, inplace=True)
                logger.info(f"Filled {col} missing values with: {fill_value}")
        
        return df
    
    def _clean_ev_specific_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean EV-specific columns."""
        # Clean Electric Range column
        if 'Electric Range' in df.columns:
            # Remove negative values and outliers
            df = df[df['Electric Range'] >= 0]
            # Cap unrealistic values (over 1000 miles)
            df.loc[df['Electric Range'] > 1000, 'Electric Range'] = 1000
        
        # Clean Base MSRP column
        if 'Base MSRP' in df.columns:
            # Remove entries with price 0 or negative
            df = df[df['Base MSRP'] > 0]
            # Cap extremely high values (over $500k)
            df.loc[df['Base MSRP'] > 500000, 'Base MSRP'] = 500000
        
        # Clean Model Year column
        if 'Model Year' in df.columns:
            current_year = pd.Timestamp.now().year
            # Remove future years and very old years
            df = df[(df['Model Year'] >= 2010) & (df['Model Year'] <= current_year + 1)]
        
        return df
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data ranges."""
        # Remove rows with invalid coordinates if present
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df = df[(df['Latitude'].between(-90, 90)) & 
                   (df['Longitude'].between(-180, 180))]
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features for better model performance."""
        logger.info("Starting feature engineering")
        
        df_features = df.copy()
        
        # Electric range categories
        if 'Electric Range' in df_features.columns:
            df_features['Range_Category'] = pd.cut(
                df_features['Electric Range'],
                bins=[0, 50, 100, 200, 500],
                labels=['Low', 'Medium', 'High', 'Ultra'],
                include_lowest=True
            )
        
        # Price per mile of range
        if 'Base MSRP' in df_features.columns and 'Electric Range' in df_features.columns:
            df_features['Price_per_Range'] = (
                df_features['Base MSRP'] / (df_features['Electric Range'] + 1)
            )
        
        # Model year categories
        if 'Model Year' in df_features.columns:
            current_year = pd.Timestamp.now().year
            df_features['Vehicle_Age'] = current_year - df_features['Model Year']
            df_features['Model_Year_Category'] = pd.cut(
                df_features['Model Year'],
                bins=[2009, 2015, 2018, current_year + 1],
                labels=['Early', 'Mid', 'Recent'],
                include_lowest=True
            )
        
        # Brand categorization
        if 'Make' in df_features.columns:
            luxury_brands = ['TESLA', 'BMW', 'MERCEDES-BENZ', 'AUDI', 'JAGUAR', 'PORSCHE']
            df_features['Is_Luxury_Brand'] = df_features['Make'].str.upper().isin(luxury_brands)
        
        logger.info("Feature engineering completed")
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            if fit:
                df_encoded[col] = self.label_encoders[col].fit_transform(
                    df_encoded[col].astype(str)
                )
            else:
                # Handle unseen categories
                try:
                    df_encoded[col] = self.label_encoders[col].transform(
                        df_encoded[col].astype(str)
                    )
                except ValueError:
                    # Handle unseen labels
                    unique_labels = df_encoded[col].unique()
                    for label in unique_labels:
                        if label not in self.label_encoders[col].classes_:
                            # Add new class to encoder
                            self.label_encoders[col].classes_ = np.append(
                                self.label_encoders[col].classes_, label
                            )
                    df_encoded[col] = self.label_encoders[col].transform(
                        df_encoded[col].astype(str)
                    )
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        
        for col in columns:
            if col in df_scaled.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                
                if fit:
                    df_scaled[col] = self.scalers[col].fit_transform(
                        df_scaled[col].values.reshape(-1, 1)
                    ).flatten()
                else:
                    df_scaled[col] = self.scalers[col].transform(
                        df_scaled[col].values.reshape(-1, 1)
                    ).flatten()
        
        return df_scaled
    
    def prepare_for_modeling(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for machine learning modeling."""
        logger.info("Preparing data for modeling")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_features = self.feature_engineering(df_clean)
        
        # Select features
        if feature_columns is None:
            # Use all numeric columns except target as features
            feature_columns = df_features.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)
        
        # Prepare X and y
        X = df_encoded[feature_columns]
        y = df_encoded[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test