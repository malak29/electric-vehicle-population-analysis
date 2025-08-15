"""Data management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import pandas as pd
import io
from datetime import datetime

from app.core.database import get_db
from app.core.config import settings
from app.utils.logger import get_logger
from app.ml.preprocessing.data_cleaning import EVDataProcessor
from pydantic import BaseModel

logger = get_logger(__name__)

router = APIRouter()


class DataSummaryResponse(BaseModel):
    """Data summary response model."""
    total_records: int
    columns: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    unique_values: Dict[str, int]
    date_range: Dict[str, str]
    vehicle_types: Dict[str, int]


class VisualizationDataResponse(BaseModel):
    """Visualization data response model."""
    chart_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]


@router.get("/summary", response_model=DataSummaryResponse)
async def get_data_summary() -> DataSummaryResponse:
    """Get summary statistics of the EV dataset."""
    try:
        # Load the dataset
        df = pd.read_csv(f"{settings.DATA_PATH}/processed/electric_vehicles.csv")
        
        # Calculate summary statistics
        summary = {
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": df.nunique().to_dict(),
        }
        
        # Date range
        if 'Model Year' in df.columns:
            summary["date_range"] = {
                "min_year": str(int(df['Model Year'].min())),
                "max_year": str(int(df['Model Year'].max()))
            }
        
        # Vehicle type distribution
        if 'Electric Vehicle Type' in df.columns:
            summary["vehicle_types"] = df['Electric Vehicle Type'].value_counts().to_dict()
        
        return DataSummaryResponse(**summary)
        
    except FileNotFoundError:
        logger.error("Dataset file not found")
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/visualization/{chart_type}")
async def get_visualization_data(
    chart_type: str,
    limit: Optional[int] = Query(100, description="Limit number of data points")
) -> VisualizationDataResponse:
    """Get data for various chart types."""
    try:
        df = pd.read_csv(f"{settings.DATA_PATH}/processed/electric_vehicles.csv")
        
        if chart_type == "vehicle_type_distribution":
            data = df['Electric Vehicle Type'].value_counts().head(limit).to_dict()
            return VisualizationDataResponse(
                chart_type=chart_type,
                data={"labels": list(data.keys()), "values": list(data.values())},
                metadata={"total_count": len(df)}
            )
            
        elif chart_type == "range_by_year":
            yearly_range = df.groupby('Model Year')['Electric Range'].mean().to_dict()
            return VisualizationDataResponse(
                chart_type=chart_type,
                data={"years": list(yearly_range.keys()), "ranges": list(yearly_range.values())},
                metadata={"unit": "miles"}
            )
            
        elif chart_type == "make_distribution":
            make_counts = df['Make'].value_counts().head(limit).to_dict()
            return VisualizationDataResponse(
                chart_type=chart_type,
                data={"makes": list(make_counts.keys()), "counts": list(make_counts.values())},
                metadata={"total_makes": df['Make'].nunique()}
            )
            
        elif chart_type == "geographic_distribution":
            geo_data = df.groupby(['State', 'County']).size().reset_index(name='count')
            geo_data = geo_data.head(limit)
            return VisualizationDataResponse(
                chart_type=chart_type,
                data=geo_data.to_dict('records'),
                metadata={"total_locations": len(geo_data)}
            )
            
        else:
            raise HTTPException(status_code=400, detail="Invalid chart type")
            
    except Exception as e:
        logger.error(f"Error generating visualization data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset."""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Basic validation
        processor = EVDataProcessor()
        if not processor.validate_ev_dataset(df):
            raise HTTPException(status_code=400, detail="Invalid dataset format")
        
        # Save the processed dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{settings.DATA_PATH}/processed/electric_vehicles_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        return {
            "message": "Dataset uploaded successfully",
            "filename": file.filename,
            "records": len(df),
            "output_path": output_path
        }
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload dataset")


@router.get("/export/{format}")
async def export_data(
    format: str,
    filters: Optional[str] = Query(None, description="JSON string of filters")
):
    """Export dataset in various formats."""
    try:
        df = pd.read_csv(f"{settings.DATA_PATH}/processed/electric_vehicles.csv")
        
        # Apply filters if provided
        if filters:
            import json
            filter_dict = json.loads(filters)
            for column, value in filter_dict.items():
                if column in df.columns:
                    df = df[df[column] == value]
        
        if format.lower() == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            return {"data": output.getvalue(), "content_type": "text/csv"}
            
        elif format.lower() == "json":
            return {"data": df.to_json(orient='records'), "content_type": "application/json"}
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail="Export failed")


@router.get("/statistics")
async def get_data_statistics():
    """Get detailed statistics about the dataset."""
    try:
        df = pd.read_csv(f"{settings.DATA_PATH}/processed/electric_vehicles.csv")
        
        stats = {
            "overview": {
                "total_vehicles": len(df),
                "total_makes": df['Make'].nunique(),
                "total_models": df['Model'].nunique(),
                "year_range": {
                    "min": int(df['Model Year'].min()),
                    "max": int(df['Model Year'].max())
                }
            },
            "vehicle_types": df['Electric Vehicle Type'].value_counts().to_dict(),
            "top_makes": df['Make'].value_counts().head(10).to_dict(),
            "range_statistics": {
                "mean": float(df['Electric Range'].mean()),
                "median": float(df['Electric Range'].median()),
                "std": float(df['Electric Range'].std()),
                "min": float(df['Electric Range'].min()),
                "max": float(df['Electric Range'].max())
            },
            "msrp_statistics": {
                "mean": float(df['Base MSRP'].mean()),
                "median": float(df['Base MSRP'].median()),
                "std": float(df['Base MSRP'].std()),
                "min": float(df['Base MSRP'].min()),
                "max": float(df['Base MSRP'].max())
            } if 'Base MSRP' in df.columns else None
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate statistics")