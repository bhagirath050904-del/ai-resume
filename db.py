"""Async SQLAlchemy database engine and session management."""

# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import DeclarativeBase

from resume_ranker.config import settings


engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def get_session() -> AsyncSession:
    """Dependency for FastAPI — yields an async DB session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables (dev convenience — use Alembic in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
