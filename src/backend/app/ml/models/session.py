"""
Database session management and connection configuration.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool
from contextlib import contextmanager
from typing import Generator
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create database engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=settings.DEBUG,  # Log SQL queries in debug mode
    connect_args={
        "connect_timeout": 10,
        "application_name": "ev_analysis_api",
        "options": "-c statement_timeout=30000"  # 30 second statement timeout
    } if "postgresql" in settings.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Usage:
        with get_db_session() as db:
            # Use db session
            pass
    
    Yields:
        Session: Database session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        session.close()


class DatabaseManager:
    """
    Manager class for database operations and health checks.
    """
    
    @staticmethod
    def init_db():
        """
        Initialize database tables and perform migrations.
        """
        from app.database.models import Base
        
        try:
            # Create all tables
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
            
            # Run any pending migrations
            DatabaseManager._run_migrations()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    @staticmethod
    def _run_migrations():
        """
        Run database migrations using Alembic.
        """
        try:
            from alembic.config import Config
            from alembic import command
            
            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")
            logger.info("Database migrations completed")
        except ImportError:
            logger.warning("Alembic not installed, skipping migrations")
        except Exception as e:
            logger.error(f"Migration error: {str(e)}")
    
    @staticmethod
    def check_connection() -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False
    
    @staticmethod
    def get_pool_status() -> dict:
        """
        Get connection pool status.
        
        Returns:
            dict: Pool status information
        """
        pool = engine.pool
        return {
            "size": pool.size() if hasattr(pool, 'size') else None,
            "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else None,
            "overflow": pool.overflow() if hasattr(pool, 'overflow') else None,
            "total": pool.size() + pool.overflow() if hasattr(pool, 'size') else None
        }
    
    @staticmethod
    def close_all_connections():
        """
        Close all database connections.
        """
        engine.dispose()
        logger.info("All database connections closed")


# Event listeners for connection management
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """
    Set SQLite pragmas for better performance (if using SQLite).
    """
    if "sqlite" in