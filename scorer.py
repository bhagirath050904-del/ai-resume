"""
Hybrid resume scoring engine.

Combines rule-based dimension scores (skills, experience, education)
with ML-based signals (sentence embeddings, TF-IDF) into a weighted
ensemble with per-dimension explanations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_ranker.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    """Per-dimension scores (all in [0, 1])."""
    skills_match: float = 0.0
    experience_match: float = 0.0
    education_match: float = 0.0
    semantic_similarity: float = 0.0
    keyword_overlap: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "skills_match": round(self.skills_match, 4),
            "experience_match": round(self.experience_match, 4),
            "education_match": round(self.education_match, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "keyword_overlap": round(self.keyword_overlap, 4),
        }


@dataclass
class MatchResult:
    """Complete scoring result for one candidate–job pair."""
    candidate_id: str
    overall_score: float
    breakdown: ScoreBreakdown
    explanations: list[str]
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "overall_score": round(self.overall_score, 4),
            "breakdown": self.breakdown.to_dict(),
            "explanations": self.explanations,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "flags": self.flags,
        }


# ──────────────────────────────────────────────
# Degree hierarchy for education scoring
# ──────────────────────────────────────────────

DEGREE_LEVELS: dict[str, int] = {
    "phd": 5, "ph.d": 5, "ph.d.": 5, "doctorate": 5,
    "ms": 4, "m.s.": 4, "m.s": 4, "m.a.": 4, "m.a": 4,
    "master": 4, "master's": 4, "mba": 4, "m.eng": 4, "m.tech": 4,
    "bs": 3, "b.s.": 3, "b.s": 3, "b.a.": 3, "b.a": 3,
    "bachelor": 3, "bachelor's": 3, "b.eng": 3, "b.tech": 3,
    "associate": 2, "associate's": 2,
}


# ──────────────────────────────────────────────
# Scorer class
# ──────────────────────────────────────────────

class ResumeScorer:
    """
    Hybrid scoring engine combining rule-based and ML-based signals.

    Dimensions:
    1. Skills match — weighted Jaccard with synonym-normalized taxonomy
    2. Experience match — years vs. required/preferred range
    3. Education match — degree level comparison
    4. Semantic similarity — sentence-transformer cosine similarity
    5. Keyword overlap — TF-IDF cosine similarity

    All dimensions output a score in [0, 1] and are combined via
    a configurable weighted average.
    """

    def __init__(
        self,
        model_name: str | None = None,
        weights: dict[str, float] | None = None,
    ):
        model_name = model_name or settings.embedding_model
        logger.info("Loading sentence-transformer: %s", model_name)
        self._embed_model = SentenceTransformer(model_name)
        self._tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._weights = weights or settings.scoring_weights

    # ── Public API ──────────────────────────

    def score(
        self,
        resume: dict,
        job: dict,
        weights: dict[str, float] | None = None,
    ) -> MatchResult:
        """
        Score a single resume against a job description.

        Args:
            resume: Parsed resume dict (from ResumeParser.to_dict()).
            job: Parsed JD dict (from JobDescriptionParser.to_dict()).
            weights: Optional per-request weight overrides.

        Returns:
            MatchResult with overall score, breakdown, and explanations.
        """
        w = weights or self._weights

        # 1. Skills match
        skills_score, matched, missing = self._score_skills(resume, job)

        # 2. Experience match
        exp_score = self._score_experience(resume, job)

        # 3. Education match
        edu_score = self._score_education(resume, job)

        # 4. Semantic similarity
        semantic_score = self._score_semantic(resume, job)

        # 5. Keyword overlap
        keyword_score = self._score_keywords(resume, job)

        # Weighted ensemble
        breakdown = ScoreBreakdown(
            skills_match=skills_score,
            experience_match=exp_score,
            education_match=edu_score,
            semantic_similarity=semantic_score,
            keyword_overlap=keyword_score,
        )

        overall = (
            w.get("skills_match", 0.35) * skills_score
            + w.get("experience_match", 0.20) * exp_score
            + w.get("education_match", 0.10) * edu_score
            + w.get("semantic_similarity", 0.25) * semantic_score
            + w.get("keyword_overlap", 0.10) * keyword_score
        )

        # Generate explanations
        explanations = self._generate_explanations(
            resume, job, matched, missing,
            skills_score, exp_score, edu_score,
            semantic_score, keyword_score,
        )

        # Flags
        flags = self._generate_flags(resume, job, exp_score, skills_score)

        return MatchResult(
            candidate_id=resume.get("candidate_id", "unknown"),
            overall_score=overall,
            breakdown=breakdown,
            explanations=explanations,
            matched_skills=sorted(matched),
            missing_skills=sorted(missing),
            flags=flags,
        )

    def rank_batch(
        self,
        resumes: list[dict],
        job: dict,
        weights: dict[str, float] | None = None,
    ) -> list[MatchResult]:
        """Score and rank a batch of resumes, sorted by score descending."""
        results = [self.score(r, job, weights) for r in resumes]
        results.sort(key=lambda r: r.overall_score, reverse=True)
        return results

    # ── Dimension scorers ───────────────────

    def _score_skills(
        self, resume: dict, job: dict
    ) -> tuple[float, set[str], set[str]]:
        """
        Compute skill match using weighted Jaccard.

        Required skills have weight 1.0, preferred have weight 0.5.
        The score accounts for these weights so missing a nice-to-have
        is penalized less than missing a requirement.
        """
        resume_skills = set(s.lower() for s in resume.get("skills", []))

        required = {
            s["skill"].lower(): s.get("weight", 1.0)
            for s in job.get("required_skills", [])
        }
        preferred = {
            s["skill"].lower(): s.get("weight", 0.5)
            for s in job.get("preferred_skills", [])
        }

        all_jd_skills = {**required, **preferred}

        if not all_jd_skills:
            return 1.0, resume_skills, set()

        matched = resume_skills & set(all_jd_skills.keys())
        missing = set(all_jd_skills.keys()) - resume_skills

        # Weighted score: sum of matched weights / sum of all weights
        total_weight = sum(all_jd_skills.values())
        matched_weight = sum(all_jd_skills[s] for s in matched)
        score = matched_weight / total_weight if total_weight > 0 else 0.0

        return score, matched, missing

    def _score_experience(self, resume: dict, job: dict) -> float:
        """Score experience: years vs. required/preferred range."""
        candidate_years = resume.get("total_experience_years", 0)
        min_years = job.get("min_experience_years", 0)
        pref_years = job.get("preferred_experience_years", min_years)

        if min_years == 0 and pref_years == 0:
            return 1.0  # No experience requirement

        if candidate_years >= pref_years:
            return 1.0
        elif candidate_years >= min_years:
            # Linear interpolation between min and preferred
            range_size = max(pref_years - min_years, 1)
            return 0.6 + 0.4 * ((candidate_years - min_years) / range_size)
        else:
            # Below minimum — partial credit
            return max(0.0, 0.6 * (candidate_years / max(min_years, 1)))

    def _score_education(self, resume: dict, job: dict) -> float:
        """Score education: candidate degree level vs. required."""
        requirements = job.get("education_requirements", [])
        if not requirements:
            return 1.0  # No education requirement

        # Find candidate's highest degree level
        candidate_level = 0
        for edu in resume.get("education", []):
            deg = edu.get("degree", "").lower().strip(".")
            candidate_level = max(
                candidate_level,
                DEGREE_LEVELS.get(deg, 0),
            )

        # Find required degree level
        required_level = 0
        for req in requirements:
            for key, level in DEGREE_LEVELS.items():
                if key in req.lower():
                    required_level = max(required_level, level)
                    break

        if required_level == 0:
            return 1.0  # Couldn't parse requirement → no penalty
        if candidate_level >= required_level:
            return 1.0
        elif candidate_level > 0:
            return candidate_level / required_level
        return 0.3  # No detectable education

    def _score_semantic(self, resume: dict, job: dict) -> float:
        """Cosine similarity of sentence-transformer embeddings."""
        resume_text = resume.get("raw_text", "")[:2000]
        jd_text = job.get("raw_text", "")[:2000]

        if not resume_text or not jd_text:
            return 0.0

        embeddings = self._embed_model.encode(
            [resume_text, jd_text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        similarity = float(
            cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        )
        # Normalize from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def _score_keywords(self, resume: dict, job: dict) -> float:
        """TF-IDF cosine similarity for keyword/domain overlap."""
        resume_text = resume.get("raw_text", "")
        jd_text = job.get("raw_text", "")

        if not resume_text or not jd_text:
            return 0.0

        try:
            tfidf_matrix = self._tfidf.fit_transform([resume_text, jd_text])
            score = float(
                cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            )
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.0

    # ── Explanation generation ──────────────

    def _generate_explanations(
        self,
        resume: dict,
        job: dict,
        matched: set[str],
        missing: set[str],
        skills_score: float,
        exp_score: float,
        edu_score: float,
        semantic_score: float,
        keyword_score: float,
    ) -> list[str]:
        """Generate human-readable explanations for the score."""
        explanations: list[str] = []
        total_skills = len(matched) + len(missing)

        # Skills
        if total_skills > 0:
            explanations.append(
                f"Matches {len(matched)}/{total_skills} "
                f"required/preferred skills ({skills_score:.0%})"
            )
            if matched:
                skill_list = ", ".join(sorted(matched)[:8])
                if len(matched) > 8:
                    skill_list += f", +{len(matched) - 8} more"
                explanations.append(f"✓ Matched: {skill_list}")
            if missing:
                miss_list = ", ".join(sorted(missing)[:5])
                if len(missing) > 5:
                    miss_list += f", +{len(missing) - 5} more"
                explanations.append(f"✗ Missing: {miss_list}")

        # Experience
        cand_years = resume.get("total_experience_years", 0)
        min_years = job.get("min_experience_years", 0)
        pref_years = job.get("preferred_experience_years", 0)

        if pref_years > 0 or min_years > 0:
            if cand_years >= pref_years > 0:
                explanations.append(
                    f"✓ {cand_years:.1f}y experience exceeds "
                    f"preferred {pref_years}y"
                )
            elif cand_years >= min_years > 0:
                explanations.append(
                    f"~ {cand_years:.1f}y experience meets minimum "
                    f"({min_years}y), below preferred ({pref_years}y)"
                )
            elif min_years > 0:
                explanations.append(
                    f"⚠ {cand_years:.1f}y experience below "
                    f"minimum {min_years}y required"
                )

        # Education
        if edu_score < 1.0 and job.get("education_requirements"):
            explanations.append(
                f"⚠ Education partially matches requirements "
                f"({edu_score:.0%})"
            )
        elif edu_score == 1.0 and job.get("education_requirements"):
            explanations.append("✓ Education meets/exceeds requirements")

        # Semantic
        explanations.append(
            f"Semantic relevance: {semantic_score:.0%}"
        )

        return explanations

    # ── Flag generation ─────────────────────

    def _generate_flags(
        self,
        resume: dict,
        job: dict,
        exp_score: float,
        skills_score: float,
    ) -> list[str]:
        """Generate warning flags for recruiter review."""
        flags: list[str] = []

        if exp_score < 0.3:
            flags.append("LOW_EXPERIENCE")
        if skills_score < 0.3:
            flags.append("LOW_SKILL_MATCH")
        if not resume.get("education"):
            flags.append("NO_EDUCATION_DETECTED")
        if resume.get("total_experience_years", 0) == 0:
            flags.append("NO_EXPERIENCE_DETECTED")

        return flags
