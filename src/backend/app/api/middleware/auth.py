"""Authentication middleware and utilities."""

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"
security = HTTPBearer()


class AuthManager:
    """Authentication manager for handling JWT tokens and user auth."""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
        """Get current user from JWT token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = self.verify_token(credentials.credentials)
            if payload is None:
                raise credentials_exception
            
            username = payload.get("sub")
            if username is None:
                raise credentials_exception
                
            return {"username": username, "payload": payload}
            
        except JWTError:
            raise credentials_exception


# Global auth manager instance
auth_manager = AuthManager()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency to get current authenticated user."""
    return auth_manager.get_current_user(credentials)


def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    """Dependency to get current user (optional authentication)."""
    if credentials is None:
        return None
    
    try:
        return auth_manager.get_current_user(credentials)
    except HTTPException:
        return None


class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed based on rate limits."""
        now = datetime.utcnow()
        
        # Clean old entries
        cutoff = now - timedelta(seconds=self.window_seconds)
        self.requests = {
            cid: requests for cid, requests in self.requests.items()
            if any(req > cutoff for req in requests)
        }
        
        # Check current client
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests for this client
        self.requests[client_id] = [
            req for req in self.requests[client_id] if req > cutoff
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        
        return False


# Global rate limiter instance
rate_limiter = RateLimiter(
    max_requests=60,  # 60 requests per minute
    window_seconds=60
)


def check_rate_limit(client_id: str = None):
    """Rate limiting dependency."""
    if client_id is None:
        client_id = "anonymous"
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return True