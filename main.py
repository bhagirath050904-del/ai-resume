"""
FastAPI application — resume ranking REST API.

Endpoints:
- POST   /api/v1/jobs                        Create a job description
- GET    /api/v1/jobs/{job_id}               Get parsed JD profile
- POST   /api/v1/jobs/{job_id}/resumes       Upload resumes for ranking
- GET    /api/v1/jobs/{job_id}/rankings       Get ranked results
- GET    /api/v1/jobs/{job_id}/rankings/{cid} Detailed match report
- DELETE /api/v1/candidates/{candidate_id}   GDPR right-to-delete
- GET    /health                              Health check
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from resume_ranker.config import settings
from resume_ranker.parsers.jd_parser import JobDescriptionParser
from resume_ranker.parsers.resume_parser import ResumeParser
from resume_ranker.scoring.scorer import ResumeScorer
from resume_ranker.schemas import (
    CandidateDetailResponse,
    CandidateRanking,
    HealthResponse,
    JobDescriptionCreate,
    JobDescriptionResponse,
    RankingListResponse,
    RankingMetadata,
    ResumeUploadResponse,
    ScoreBreakdown,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# In-memory stores (MVP — replaced with DB in production)
# ──────────────────────────────────────────────

_jobs: dict[str, dict] = {}           # job_id → parsed JD profile dict
_candidates: dict[str, dict] = {}     # candidate_id → parsed resume dict
_results: dict[str, list[dict]] = {}  # job_id → list of match result dicts


# ──────────────────────────────────────────────
# Singleton services (initialized at startup)
# ──────────────────────────────────────────────

_resume_parser: ResumeParser | None = None
_jd_parser: JobDescriptionParser | None = None
_scorer: ResumeScorer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy services on startup, clean up on shutdown."""
    global _resume_parser, _jd_parser, _scorer

    logger.info("Starting Resume Ranker API v%s", settings.app_version)

    # Initialize parsers (lightweight)
    _resume_parser = ResumeParser()
    _jd_parser = JobDescriptionParser()

    # Initialize scorer (loads sentence-transformer model — takes a few seconds)
    logger.info("Loading scoring model: %s", settings.embedding_model)
    _scorer = ResumeScorer()
    logger.info("Model loaded, API ready.")

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    yield

    # Cleanup
    logger.info("Shutting down Resume Ranker API.")


# ──────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────

app = FastAPI(
    title="Resume Ranker API",
    description=(
        "NLP-based resume ranking system for HR screening. "
        "Upload a job description and candidate resumes to get "
        "a ranked shortlist with explainable scores."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        database="in-memory (MVP)",
    )


# ──────────────────────────────────────────────
# Job description endpoints
# ──────────────────────────────────────────────

@app.post(
    f"{settings.api_prefix}/jobs",
    status_code=201,
    response_model=JobDescriptionResponse,
    tags=["Jobs"],
)
async def create_job(request: JobDescriptionCreate):
    """Upload and parse a job description."""
    overrides = {}
    if request.required_skills:
        overrides["required_skills"] = [
            {"skill": s.skill, "weight": s.weight}
            for s in request.required_skills
        ]
    if request.preferred_skills:
        overrides["preferred_skills"] = [
            {"skill": s.skill, "weight": s.weight}
            for s in request.preferred_skills
        ]
    if request.min_experience_years is not None:
        overrides["min_experience_years"] = request.min_experience_years
    if request.preferred_experience_years is not None:
        overrides["preferred_experience_years"] = request.preferred_experience_years
    if request.education_requirements:
        overrides["education_requirements"] = request.education_requirements

    parsed = _jd_parser.parse(
        title=request.title,
        description=request.description,
        overrides=overrides or None,
    )

    job_id = str(uuid.uuid4())
    profile = parsed.to_dict()
    profile["job_id"] = job_id
    _jobs[job_id] = profile

    logger.info("Created job %s: %s", job_id, request.title)

    return JobDescriptionResponse(
        job_id=job_id,
        title=request.title,
        parsed_profile=profile,
        created_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
    )


@app.get(
    f"{settings.api_prefix}/jobs/{{job_id}}",
    response_model=JobDescriptionResponse,
    tags=["Jobs"],
)
async def get_job(job_id: str):
    """Retrieve a parsed job description."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    profile = _jobs[job_id]
    return JobDescriptionResponse(
        job_id=job_id,
        title=profile["title"],
        parsed_profile=profile,
        created_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
    )


# ──────────────────────────────────────────────
# Resume upload + ranking
# ──────────────────────────────────────────────

@app.post(
    f"{settings.api_prefix}/jobs/{{job_id}}/resumes",
    status_code=202,
    response_model=ResumeUploadResponse,
    tags=["Resumes"],
)
async def upload_resumes(
    job_id: str,
    files: list[UploadFile] = File(..., description="Resume files (PDF, DOCX, TXT)"),
):
    """
    Upload resumes and score them against a job description.

    In MVP mode, scoring is synchronous and results are stored in memory.
    In production, this dispatches Celery tasks for async processing.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_profile = _jobs[job_id]
    batch_id = str(uuid.uuid4())
    results: list[dict] = []

    for upload_file in files:
        # Validate file type
        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix not in (".pdf", ".docx", ".doc", ".txt"):
            logger.warning("Skipping unsupported file: %s", upload_file.filename)
            continue

        # Save uploaded file
        file_id = str(uuid.uuid4())
        save_path = Path(settings.upload_dir) / f"{file_id}{suffix}"
        with open(save_path, "wb") as f:
            content = await upload_file.read()
            f.write(content)

        try:
            # Parse resume
            parsed = _resume_parser.parse(save_path)
            resume_dict = parsed.to_dict()
            _candidates[parsed.candidate_id] = resume_dict

            # Score against job
            match_result = _scorer.score(resume_dict, job_profile)
            result_dict = match_result.to_dict()
            result_dict["candidate_name"] = parsed.name
            results.append(result_dict)

        except Exception as e:
            logger.error(
                "Error processing %s: %s", upload_file.filename, str(e)
            )
            results.append({
                "candidate_id": file_id,
                "candidate_name": upload_file.filename,
                "overall_score": 0.0,
                "error": str(e),
                "flags": ["PARSE_ERROR"],
            })

    # Sort by score and store
    results.sort(key=lambda r: r.get("overall_score", 0), reverse=True)

    if job_id not in _results:
        _results[job_id] = []
    _results[job_id].extend(results)
    # Re-sort all results for this job
    _results[job_id].sort(
        key=lambda r: r.get("overall_score", 0), reverse=True
    )

    return ResumeUploadResponse(
        batch_id=batch_id,
        job_id=job_id,
        status="completed",
        total_files=len(files),
    )


@app.get(
    f"{settings.api_prefix}/jobs/{{job_id}}/rankings",
    tags=["Rankings"],
)
async def get_rankings(
    job_id: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    min_score: Annotated[float | None, Query(ge=0, le=1)] = None,
):
    """Retrieve ranked candidate results for a job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    all_results = _results.get(job_id, [])

    # Filter by min score
    if min_score is not None:
        all_results = [
            r for r in all_results
            if r.get("overall_score", 0) >= min_score
        ]

    # Paginate
    page = all_results[offset : offset + limit]

    # Add rank numbers
    ranked = []
    for i, r in enumerate(page, start=offset + 1):
        ranked.append({
            "rank": i,
            **r,
        })

    return {
        "job_id": job_id,
        "ranked_candidates": ranked,
        "metadata": {
            "total_candidates": len(all_results),
            "model_version": settings.app_version,
        },
    }


@app.get(
    f"{settings.api_prefix}/jobs/{{job_id}}/rankings/{{candidate_id}}",
    tags=["Rankings"],
)
async def get_candidate_detail(job_id: str, candidate_id: str):
    """Get detailed match report for a specific candidate."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    results = _results.get(job_id, [])
    for r in results:
        if r.get("candidate_id") == candidate_id:
            return r

    raise HTTPException(status_code=404, detail="Candidate not found for this job")


# ──────────────────────────────────────────────
# GDPR compliance
# ──────────────────────────────────────────────

@app.delete(
    f"{settings.api_prefix}/candidates/{{candidate_id}}",
    status_code=204,
    tags=["Compliance"],
)
async def delete_candidate(candidate_id: str):
    """
    GDPR right-to-delete: remove all data for a candidate.

    Deletes from in-memory store (MVP) and uploaded files.
    In production, cascades through DB tables + file storage.
    """
    # Remove from candidates store
    if candidate_id in _candidates:
        del _candidates[candidate_id]

    # Remove from all job results
    for job_id in _results:
        _results[job_id] = [
            r for r in _results[job_id]
            if r.get("candidate_id") != candidate_id
        ]

    logger.info("Deleted candidate data: %s", candidate_id)
    return None


# ──────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
