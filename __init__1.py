"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Job Description schemas
# ──────────────────────────────────────────────

class SkillRequirement(BaseModel):
    skill: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class JobDescriptionCreate(BaseModel):
    """Request body for creating a job description."""
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=10)
    required_skills: list[SkillRequirement] | None = None
    preferred_skills: list[SkillRequirement] | None = None
    min_experience_years: int | None = None
    preferred_experience_years: int | None = None
    education_requirements: list[str] | None = None


class JobDescriptionResponse(BaseModel):
    """Response after creating/retrieving a job description."""
    job_id: UUID
    title: str
    parsed_profile: dict
    created_at: datetime

    model_config = {"from_attributes": True}


# ──────────────────────────────────────────────
# Resume upload schemas
# ──────────────────────────────────────────────

class ResumeUploadResponse(BaseModel):
    """Response after uploading resumes for ranking."""
    batch_id: str
    job_id: UUID
    status: str = "processing"
    total_files: int


# ──────────────────────────────────────────────
# Scoring / ranking schemas
# ──────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    skills_match: float
    experience_match: float
    education_match: float
    semantic_similarity: float
    keyword_overlap: float


class CandidateRanking(BaseModel):
    """A single candidate's ranking result."""
    rank: int
    candidate_id: UUID
    overall_score: float
    breakdown: ScoreBreakdown
    explanations: list[str]
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    flags: list[str] = []


class RankingListResponse(BaseModel):
    """Paginated ranked results for a job."""
    job_id: UUID
    ranked_candidates: list[CandidateRanking]
    metadata: RankingMetadata


class RankingMetadata(BaseModel):
    total_candidates: int
    processing_time_ms: float | None = None
    model_version: str = "v1.0.0"


# Fix forward reference
RankingListResponse.model_rebuild()


class CandidateDetailResponse(BaseModel):
    """Detailed match report for a single candidate."""
    candidate_id: UUID
    overall_score: float
    breakdown: ScoreBreakdown
    explanations: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    experience_analysis: dict = {}
    education_analysis: dict = {}
    flags: list[str] = []

    model_config = {"from_attributes": True}


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    database: str = "connected"
