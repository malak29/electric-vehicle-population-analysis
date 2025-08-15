"""
Security utilities for authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Any, Union, Optional
from passlib.context import CryptContext
from jose import jwt, JWTError
import secrets
import hashlib
import hmac
from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    
    Args:
        subject: Token subject (usually user ID)
        expires_delta: Optional expiration time delta
    
    Returns:
        str: Encoded JWT token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token to verify
    
    Returns:
        dict: Decoded token payload
    
    Raises:
        JWTError: If token is invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        return payload
    except JWTError:
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
    
    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash password.
    
    Args:
        password: Plain text password
    
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def generate_api_key() -> str:
    """
    Generate secure API key.
    
    Returns:
        str: API key
    """
    return secrets.token_urlsafe(32)


def generate_reset_token() -> str:
    """
    Generate password reset token.
    
    Returns:
        str: Reset token
    """
    return secrets.token_urlsafe(32)


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT refresh token.
    
    Args:
        subject: Token subject (usually user ID)
        expires_delta: Optional expiration time delta
    
    Returns:
        str: Encoded JWT refresh token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)  # 7 days for refresh token
    
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return encoded_jwt


def verify_signature(payload: str, signature: str, secret: str) -> bool:
    """
    Verify HMAC signature.
    
    Args:
        payload: Payload to verify
        signature: Signature to check
        secret: Secret key
    
    Returns:
        bool: True if signature is valid
    """
    expected_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_signature, signature)


def hash_data(data: str) -> str:
    """
    Create SHA256 hash of data.
    
    Args:
        data: Data to hash
    
    Returns:
        str: Hexadecimal hash
    """
    return hashlib.sha256(data.encode()).hexdigest()


def is_strong_password(password: str) -> tuple[bool, str]:
    """
    Check if password meets strength requirements.
    
    Args:
        password: Password to check
    
    Returns:
        tuple: (is_strong, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"


def sanitize_input(input_string: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_string: Input to sanitize
    
    Returns:
        str: Sanitized input
    """
    # Remove potentially dangerous characters
    dangerous_chars = ["<", ">", "&", '"', "'", "/", "\\", ";", "(", ")", "{", "}"]
    
    sanitized = input_string
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")
    
    # Limit length
    max_length = 1000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()


class PermissionChecker:
    """
    Permission checking utility.
    """
    
    @staticmethod
    def has_permission(user, resource: str, action: str) -> bool:
        """
        Check if user has permission for action on resource.
        
        Args:
            user: User object
            resource: Resource name
            action: Action name (read, write, delete)
        
        Returns:
            bool: True if user has permission
        """
        # Superusers have all permissions
        if user.is_superuser:
            return True
        
        # Check specific permissions based on resource and action
        permissions_map = {
            "model": {
                "read": True,  # All authenticated users can read
                "write": user.is_active,  # Active users can write
                "delete": user.is_superuser  # Only superusers can delete
            },
            "dataset": {
                "read": True,
                "write": user.is_active,
                "delete": user.is_superuser
            },
            "prediction": {
                "read": True,
                "write": user.is_active,
                "delete": False  # Predictions cannot be deleted
            },
            "user": {
                "read": user.is_superuser,  # Only superusers can read other users
                "write": user.is_superuser,
                "delete": user.is_superuser
            }
        }
        
        if resource in permissions_map:
            return permissions_map[resource].get(action, False)
        
        return False
    
    @staticmethod
    def check_rate_limit(user, action: str) -> bool:
        """
        Check if user has exceeded rate limit for action.
        
        Args:
            user: User object
            action: Action being performed
        
        Returns:
            bool: True if within rate limit
        """
        # Check usage quota
        if user.current_usage >= user.usage_quota:
            return False
        
        # Action-specific limits
        action_limits = {
            "train_model": 10,  # Max 10 model training per day
            "batch_prediction": 50,  # Max 50 batch predictions per day
            "single_prediction": 1000  # Max 1000 single predictions per day
        }
        
        # This would typically check against a database or cache
        # For now, we'll just return True
        return True