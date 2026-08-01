"""SQLAlchemy ORM models for the resume ranking system."""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship

from resume_ranker.db import Base
from resume_ranker.config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobDescription(Base):
    """A job description uploaded by a recruiter."""

    __tablename__ = "job_descriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    raw_text = Column(Text, nullable=False)
    structured = Column(JSONB, nullable=False, default=dict)
    embedding = Column(Vector(settings.embedding_dim), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    created_by = Column(String(255), nullable=True)

    # Relationships
    match_results = relationship(
        "MatchResult", back_populates="job", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<JobDescription id={self.id} title='{self.title[:40]}'>"


class Candidate(Base):
    """A parsed candidate resume."""

    __tablename__ = "candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name_hash = Column(String(64), nullable=True, index=True)
    raw_text = Column(Text, nullable=True)
    structured = Column(JSONB, nullable=False, default=dict)
    embedding = Column(Vector(settings.embedding_dim), nullable=True)
    file_path = Column(String(1024), nullable=True)
    uploaded_at = Column(DateTime(timezone=True), default=utcnow)
    retention_until = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    match_results = relationship(
        "MatchResult", back_populates="candidate", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Candidate id={self.id}>"


class MatchResult(Base):
    """Scoring result for a candidate–job pair."""

    __tablename__ = "match_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("job_descriptions.id"), nullable=False
    )
    candidate_id = Column(
        UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False
    )
    overall_score = Column(Float, nullable=False)
    breakdown = Column(JSONB, nullable=False, default=dict)
    explanations = Column(ARRAY(Text), nullable=False, default=list)
    flags = Column(ARRAY(Text), nullable=True, default=list)
    model_version = Column(String(50), nullable=False, default="v1.0.0")
    scored_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    job = relationship("JobDescription", back_populates="match_results")
    candidate = relationship("Candidate", back_populates="match_results")

    __table_args__ = (
        UniqueConstraint("job_id", "candidate_id", "model_version", name="uq_match"),
        Index("idx_match_job", "job_id"),
        Index("idx_match_score", "job_id", "overall_score"),
    )

    def __repr__(self) -> str:
        return (
            f"<MatchResult job={self.job_id} candidate={self.candidate_id} "
            f"score={self.overall_score:.3f}>"
        )


class AuditLog(Base):
    """Append-only audit trail for compliance."""

    __tablename__ = "audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action = Column(String(100), nullable=False, index=True)
    actor_id = Column(String(255), nullable=True)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(255), nullable=True)
    detail = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    def __repr__(self) -> str:
        return f"<AuditLog action='{self.action}' at={self.created_at}>"
