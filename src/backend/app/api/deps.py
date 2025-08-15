"""
API dependencies for authentication, database sessions, and common utilities.
"""

from typing import Generator, Optional, Annotated
from fastapi import Depends, HTTPException, status, Header, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
import redis
import logging

from app.database.session import get_db
from app.core.config import settings
from app.database.models import User
from app.core.security import verify_token

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Redis client for caching
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


class RateLimiter:
    """
    Rate limiting dependency.
    """
    
    def __init__(self, max_requests: int = 60, window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed
            window: Time window in seconds
        """
        self.max_requests = max_requests
        self.window = window
    
    def __call__(self, 
                 authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
                 x_forwarded_for: Optional[str] = Header(None)):
        """
        Check rate limit for request.
        
        Args:
            authorization: Auth credentials
            x_forwarded_for: Client IP from proxy
        
        Raises:
            HTTPException: If rate limit exceeded
        """
        # Get identifier (user token or IP)
        if authorization:
            identifier = f"user:{authorization.credentials}"
        else:
            identifier = f"ip:{x_forwarded_for or 'unknown'}"
        
        key = f"rate_limit:{identifier}"
        
        try:
            current = redis_client.incr(key)
            if current == 1:
                redis_client.expire(key, self.window)
            
            if current > self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window} seconds."
                )
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {str(e)}")
            # Don't block requests if Redis is down
            pass


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user.
    
    Args:
        credentials: Bearer token credentials
        db: Database session
    
    Returns:
        User: Current user object
    
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    
    try:
        # Verify and decode token
        payload = verify_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current user from get_current_user
    
    Returns:
        User: Active user object
    
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current superuser.
    
    Args:
        current_user: Current user from get_current_user
    
    Returns:
        User: Superuser object
    
    Raises:
        HTTPException: If user is not superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def get_optional_current_user(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Args:
        authorization: Optional auth credentials
        db: Database session
    
    Returns:
        Optional[User]: User object or None
    """
    if not authorization:
        return None
    
    try:
        return get_current_user(authorization, db)
    except HTTPException:
        return None


class PaginationParams:
    """
    Common pagination parameters.
    """
    
    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Number of items to skip"),
        limit: int = Query(100, ge=1, le=1000, description="Number of items to return")
    ):
        self.skip = skip
        self.limit = limit


class SortParams:
    """
    Common sorting parameters.
    """
    
    def __init__(
        self,
        sort_by: str = Query("created_at", description="Field to sort by"),
        sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
    ):
        self.sort_by = sort_by
        self.sort_order = sort_order


class FilterParams:
    """
    Common filter parameters.
    """
    
    def __init__(
        self,
        search: Optional[str] = Query(None, description="Search term"),
        date_from: Optional[datetime] = Query(None, description="Filter from date"),
        date_to: Optional[datetime] = Query(None, description="Filter to date"),
        status: Optional[str] = Query(None, description="Filter by status")
    ):
        self.search = search
        self.date_from = date_from
        self.date_to = date_to
        self.status = status


def verify_api_key(
    x_api_key: str = Header(..., description="API Key"),
    db: Session = Depends(get_db)
) -> User:
    """
    Verify API key for authentication.
    
    Args:
        x_api_key: API key from header
        db: Database session
    
    Returns:
        User: User associated with API key
    
    Raises:
        HTTPException: If API key is invalid
    """
    user = db.query(User).filter(User.api_key == x_api_key).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Check usage quota
    if user.current_usage >= user.usage_quota:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Usage quota exceeded"
        )
    
    # Increment usage counter
    user.current_usage += 1
    db.commit()
    
    return user


def get_cache_key(prefix: str, **kwargs) -> str:
    """
    Generate cache key from prefix and parameters.
    
    Args:
        prefix: Cache key prefix
        **kwargs: Additional parameters
    
    Returns:
        str: Cache key
    """
    params = "_".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return f"{prefix}:{params}" if params else prefix


def cache_dependency(key_prefix: str, ttl: int = 3600):
    """
    Cache dependency decorator.
    
    Args:
        key_prefix: Cache key prefix
        ttl: Time to live in seconds
    
    Returns:
        Cached value or None
    """
    def dependency(**kwargs):
        key = get_cache_key(key_prefix, **kwargs)
        try:
            cached = redis_client.get(key)
            if cached:
                import json
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache error: {str(e)}")
        return None
    
    return dependency


def invalidate_cache(pattern: str):
    """
    Invalidate cache entries matching pattern.
    
    Args:
        pattern: Cache key pattern
    """
    try:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries")
    except Exception as e:
        logger.error(f"Cache invalidation error: {str(e)}")


# Common dependency annotations
DatabaseDep = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_active_user)]
OptionalUser = Annotated[Optional[User], Depends(get_optional_current_user)]
SuperUser = Annotated[User, Depends(get_current_superuser)]
Pagination = Annotated[PaginationParams, Depends()]
Sorting = Annotated[SortParams, Depends()]
Filtering = Annotated[FilterParams, Depends()]
RateLimit = Annotated[None, Depends(RateLimiter())]