"""
Job description parser: extract structured requirement profiles
from free-text job descriptions.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from resume_ranker.parsers.skill_taxonomy import find_skills_in_text

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class SkillRequirement:
    skill: str
    weight: float = 1.0  # 1.0 = required, 0.5 = preferred/nice-to-have


@dataclass
class ParsedJobDescription:
    job_id: str
    title: str
    seniority_level: str
    required_skills: list[SkillRequirement]
    preferred_skills: list[SkillRequirement]
    min_experience_years: int
    preferred_experience_years: int
    education_requirements: list[str]
    domain_terms: list[str]
    raw_text: str

    def to_dict(self) -> dict:
        """Serialize for storage and scoring."""
        return {
            "job_id": self.job_id,
            "title": self.title,
            "seniority_level": self.seniority_level,
            "required_skills": [
                {"skill": s.skill, "weight": s.weight}
                for s in self.required_skills
            ],
            "preferred_skills": [
                {"skill": s.skill, "weight": s.weight}
                for s in self.preferred_skills
            ],
            "min_experience_years": self.min_experience_years,
            "preferred_experience_years": self.preferred_experience_years,
            "education_requirements": self.education_requirements,
            "domain_terms": self.domain_terms,
            "raw_text": self.raw_text,
        }


# ──────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────

SENIORITY_PATTERNS = {
    "intern": re.compile(r"(?i)\b(intern|internship|co-?op)\b"),
    "junior": re.compile(r"(?i)\b(junior|jr\.?|entry[- ]level|associate)\b"),
    "mid": re.compile(r"(?i)\b(mid[- ]?level|intermediate)\b"),
    "senior": re.compile(r"(?i)\b(senior|sr\.?|lead|principal|staff)\b"),
    "director": re.compile(r"(?i)\b(director|vp|vice president|head of|chief)\b"),
}

EXPERIENCE_YEAR_RE = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?",
    re.IGNORECASE,
)

DEGREE_RE = re.compile(
    r"(?i)\b(Ph\.?D|Doctorate|Master(?:'?s)?|M\.?S\.?|M\.?A\.?|MBA|"
    r"Bachelor(?:'?s)?|B\.?S\.?|B\.?A\.?|B\.?Eng|B\.?Tech|"
    r"Associate(?:'?s)?)\b"
)

# Markers that separate "required" from "nice-to-have" in JD text
PREFERRED_MARKERS = re.compile(
    r"(?i)\b(nice[- ]to[- ]have|preferred|bonus|plus|desired|optional|ideally)\b"
)

REQUIRED_MARKERS = re.compile(
    r"(?i)\b(required|must[- ]have|essential|mandatory|minimum|qualifications?|"
    r"requirements?|what you need|what we look for)\b"
)


# ──────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────

class JobDescriptionParser:
    """
    Parses free-text job descriptions into structured requirement profiles.

    Extracts:
    - Seniority level from title and body
    - Required vs. preferred skills (using section markers)
    - Experience year requirements
    - Education requirements
    - Domain-specific terms
    """

    def parse(
        self,
        title: str,
        description: str,
        overrides: dict | None = None,
    ) -> ParsedJobDescription:
        """
        Parse a job description into a structured profile.

        Args:
            title: Job title string.
            description: Full text of the job description.
            overrides: Optional pre-structured data (e.g., from a form)
                       that takes precedence over text parsing.
        """
        logger.info("Parsing JD: %s", title[:60])

        full_text = f"{title}\n{description}"
        overrides = overrides or {}

        # Seniority
        seniority = overrides.get(
            "seniority_level", self._detect_seniority(full_text)
        )

        # Skills — split into required vs preferred
        all_skills = find_skills_in_text(full_text)
        required_skills, preferred_skills = self._split_required_preferred(
            description, all_skills
        )

        # Apply overrides if provided
        if overrides.get("required_skills"):
            required_skills = [
                SkillRequirement(s["skill"], s.get("weight", 1.0))
                for s in overrides["required_skills"]
            ]
        if overrides.get("preferred_skills"):
            preferred_skills = [
                SkillRequirement(s["skill"], s.get("weight", 0.5))
                for s in overrides["preferred_skills"]
            ]

        # Experience
        min_exp = overrides.get(
            "min_experience_years",
            self._extract_experience_years(description),
        )
        pref_exp = overrides.get(
            "preferred_experience_years",
            self._extract_preferred_experience(description, min_exp),
        )

        # Education
        education = overrides.get(
            "education_requirements",
            self._extract_education(description),
        )

        # Domain terms
        domain_terms = self._extract_domain_terms(description)

        return ParsedJobDescription(
            job_id=self._generate_id(full_text),
            title=title,
            seniority_level=seniority,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_experience_years=min_exp,
            preferred_experience_years=pref_exp,
            education_requirements=education,
            domain_terms=domain_terms,
            raw_text=full_text,
        )

    # ── Seniority detection ─────────────────

    def _detect_seniority(self, text: str) -> str:
        """Detect seniority level from text (title + body)."""
        # Check title area first (more weight), then body
        for level in ("director", "senior", "mid", "junior", "intern"):
            if SENIORITY_PATTERNS[level].search(text[:200]):
                return level

        # Body-wide search
        for level in ("director", "senior", "mid", "junior", "intern"):
            if SENIORITY_PATTERNS[level].search(text):
                return level

        return "mid"  # Default

    # ── Skill splitting ─────────────────────

    def _split_required_preferred(
        self, text: str, all_skills: set[str]
    ) -> tuple[list[SkillRequirement], list[SkillRequirement]]:
        """
        Split skills into required vs preferred based on JD section markers.

        Strategy:
        1. Find "preferred/nice-to-have" section boundaries.
        2. Skills found before that section are required.
        3. Skills found in/after that section are preferred.
        """
        lines = text.split("\n")
        in_preferred_section = False
        required_zone_text: list[str] = []
        preferred_zone_text: list[str] = []

        for line in lines:
            if PREFERRED_MARKERS.search(line):
                in_preferred_section = True
            elif REQUIRED_MARKERS.search(line):
                in_preferred_section = False

            if in_preferred_section:
                preferred_zone_text.append(line)
            else:
                required_zone_text.append(line)

        required_text = "\n".join(required_zone_text)
        preferred_text = "\n".join(preferred_zone_text)

        required_skills_set = find_skills_in_text(required_text)
        preferred_skills_set = find_skills_in_text(preferred_text)

        # Remove overlap (if a skill appears in both, it's required)
        preferred_only = preferred_skills_set - required_skills_set

        # If no section split found, treat all as required
        if not preferred_zone_text:
            required_skills_set = all_skills
            preferred_only = set()

        required = [SkillRequirement(s, 1.0) for s in sorted(required_skills_set)]
        preferred = [SkillRequirement(s, 0.5) for s in sorted(preferred_only)]

        return required, preferred

    # ── Experience parsing ──────────────────

    def _extract_experience_years(self, text: str) -> int:
        """Extract minimum years of experience from JD text."""
        matches = EXPERIENCE_YEAR_RE.findall(text)
        if matches:
            return int(min(matches))
        return 0

    def _extract_preferred_experience(
        self, text: str, min_years: int
    ) -> int:
        """Extract preferred/ideal years (often higher than minimum)."""
        matches = EXPERIENCE_YEAR_RE.findall(text)
        if len(matches) >= 2:
            years = sorted(int(m) for m in matches)
            return years[-1]  # Highest mentioned
        return max(min_years, min_years + 2)  # Default: min + 2

    # ── Education parsing ───────────────────

    def _extract_education(self, text: str) -> list[str]:
        """Extract education requirements."""
        degrees_found: list[str] = []
        for match in DEGREE_RE.finditer(text):
            deg = match.group(1)
            if deg not in degrees_found:
                degrees_found.append(deg)

        # Build requirement strings with context
        requirements: list[str] = []
        for deg in degrees_found:
            # Try to find the field of study near the degree mention
            context_re = re.compile(
                rf"(?i){re.escape(deg)}\s+(?:in\s+|of\s+)?([A-Z][\w\s,]+)",
            )
            ctx_match = context_re.search(text)
            if ctx_match:
                field_text = ctx_match.group(1).strip()[:60]
                requirements.append(f"{deg} in {field_text}")
            else:
                requirements.append(deg)

        return requirements if requirements else []

    # ── Domain terms ────────────────────────

    def _extract_domain_terms(self, text: str) -> list[str]:
        """
        Extract domain-specific terms that aren't in the skill taxonomy.

        These are multi-word noun phrases that may indicate specialized
        knowledge areas (e.g., "fraud detection", "supply chain optimization").
        """
        # Simple approach: extract capitalized multi-word phrases
        domain_pattern = re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Za-z]+){1,3})\b"
        )
        candidates = set()
        for match in domain_pattern.finditer(text):
            phrase = match.group(1).lower()
            # Filter out common non-domain phrases
            stopwords = {
                "the company", "we are", "you will", "this role",
                "our team", "your work", "the team", "new york",
            }
            if phrase not in stopwords and len(phrase) > 5:
                candidates.add(phrase)

        return sorted(candidates)[:20]  # Cap at 20 terms

    # ── Helpers ─────────────────────────────

    def _generate_id(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"jd-{digest}"
