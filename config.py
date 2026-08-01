"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Centralized configuration loaded from environment or .env file."""

    # Application
    app_name: str = "Resume Ranker"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://ranker:ranker@localhost:5432/resume_ranker",
        description="Async PostgreSQL connection string",
    )
    database_url_sync: str = Field(
        default="postgresql://ranker:ranker@localhost:5432/resume_ranker",
        description="Sync PostgreSQL connection string (for Alembic)",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string for Celery broker + cache",
    )

    # File storage
    upload_dir: str = Field(
        default="./uploads",
        description="Directory for uploaded resume files",
    )
    max_file_size_mb: int = Field(
        default=10,
        description="Maximum file size in MB for uploads",
    )

    # NLP Models
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model to load (use en_core_web_trf for production)",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model name",
    )
    embedding_dim: int = Field(
        default=384,
        description="Embedding vector dimension (must match model)",
    )

    # Scoring weights
    weight_skills: float = 0.35
    weight_experience: float = 0.20
    weight_education: float = 0.10
    weight_semantic: float = 0.25
    weight_keyword: float = 0.10

    # Data retention
    retention_days: int = Field(
        default=90,
        description="Days to retain raw resume files before auto-deletion",
    )

    # API
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    @property
    def scoring_weights(self) -> dict[str, float]:
        """Return scoring weights as a dictionary."""
        return {
            "skills_match": self.weight_skills,
            "experience_match": self.weight_experience,
            "education_match": self.weight_education,
            "semantic_similarity": self.weight_semantic,
            "keyword_overlap": self.weight_keyword,
        }


# Singleton instance
settings = Settings()
