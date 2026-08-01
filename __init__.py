"""Celery task definitions for async resume processing."""

from __future__ import annotations

import logging
from pathlib import Path

from celery import Celery

from resume_ranker.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Celery app
# ──────────────────────────────────────────────

celery_app = Celery(
    "resume_ranker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,        # 5 minute hard limit per task
    task_soft_time_limit=240,   # 4 minute soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)


# ──────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────

@celery_app.task(bind=True, name="resume_ranker.tasks.parse_and_score")
def parse_and_score_task(
    self,
    file_path: str,
    job_profile: dict,
    job_id: str,
) -> dict:
    """
    Parse a single resume and score it against a job description.

    This task is designed to be dispatched per-resume in a batch upload.
    Results are stored in the database (production) or returned directly (MVP).

    Args:
        file_path: Path to the uploaded resume file.
        job_profile: Parsed JD profile dict.
        job_id: ID of the job description.

    Returns:
        Match result dict with score, breakdown, and explanations.
    """
    from resume_ranker.parsers.resume_parser import ResumeParser
    from resume_ranker.scoring.scorer import ResumeScorer

    logger.info("Task %s: parsing %s", self.request.id, file_path)

    try:
        # Parse resume
        parser = ResumeParser()
        parsed = parser.parse(file_path)
        resume_dict = parsed.to_dict()

        # Score against job
        scorer = ResumeScorer()
        result = scorer.score(resume_dict, job_profile)

        logger.info(
            "Task %s: scored %s → %.3f",
            self.request.id, parsed.candidate_id, result.overall_score,
        )

        return {
            "status": "success",
            "job_id": job_id,
            "result": result.to_dict(),
            "candidate_name": parsed.name,
        }

    except Exception as exc:
        logger.error("Task %s failed: %s", self.request.id, str(exc))
        return {
            "status": "error",
            "job_id": job_id,
            "error": str(exc),
            "file_path": file_path,
        }


@celery_app.task(bind=True, name="resume_ranker.tasks.batch_score")
def batch_score_task(
    self,
    file_paths: list[str],
    job_profile: dict,
    job_id: str,
) -> dict:
    """
    Parse and score a batch of resumes.

    Dispatches individual parse_and_score tasks as a Celery group
    for parallel processing across workers.

    Args:
        file_paths: List of paths to uploaded resume files.
        job_profile: Parsed JD profile dict.
        job_id: ID of the job description.

    Returns:
        Aggregated results with ranked candidates.
    """
    from celery import group

    tasks = group(
        parse_and_score_task.s(fp, job_profile, job_id)
        for fp in file_paths
    )

    result = tasks.apply_async()
    results = result.get(timeout=300)  # Wait up to 5 minutes

    # Filter successes, sort by score
    successes = [
        r for r in results
        if r.get("status") == "success"
    ]
    errors = [
        r for r in results
        if r.get("status") == "error"
    ]

    successes.sort(
        key=lambda r: r["result"]["overall_score"],
        reverse=True,
    )

    return {
        "job_id": job_id,
        "total": len(file_paths),
        "scored": len(successes),
        "errors": len(errors),
        "ranked_results": [r["result"] for r in successes],
        "error_details": errors,
    }
