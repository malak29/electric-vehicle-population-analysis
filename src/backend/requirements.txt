"""Health check endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import psutil
import time
from datetime import datetime

from app.core.config import settings
from app.core.database import get_db
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    uptime: float
    database: str
    memory_usage: float
    cpu_usage: float


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model."""
    status: str
    timestamp: str
    version: str
    uptime: float
    services: Dict[str, Any]
    system: Dict[str, Any]


# Store start time for uptime calculation
start_time = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    try:
        uptime = time.time() - start_time
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version=settings.VERSION,
            uptime=uptime,
            database="connected",
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(db = Depends(get_db)):
    """Detailed health check with system information."""
    try:
        uptime = time.time() - start_time
        
        # Check database connection
        try:
            db.execute("SELECT 1")
            db_status = "connected"
        except Exception:
            db_status = "disconnected"
        
        # System information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
        
        services = {
            "database": {
                "status": db_status,
                "url": settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "N/A"
            },
            "mlflow": {
                "status": "configured",
                "tracking_uri": settings.MLFLOW_TRACKING_URI
            }
        }
        
        return DetailedHealthResponse(
            status="healthy" if db_status == "connected" else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            version=settings.VERSION,
            uptime=uptime,
            services=services,
            system=system_info
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}