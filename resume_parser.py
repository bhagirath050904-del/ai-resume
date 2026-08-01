"""
Resume parser: extract structured data from PDF, DOCX, and TXT resumes.

Pipeline: raw file → text extraction → section segmentation
          → entity extraction → structured dict
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from docx import Document

from resume_ranker.parsers.skill_taxonomy import find_skills_in_text

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class Education:
    degree: str
    field_of_study: str = ""
    institution: str = ""
    year: int | None = None


@dataclass
class Experience:
    title: str
    company: str = ""
    start_year: int | None = None
    end_year: int | None = None  # None = present
    duration_months: int | None = None
    description: str = ""
    skills_mentioned: list[str] = field(default_factory=list)


@dataclass
class ParsedResume:
    candidate_id: str
    name: str
    email: str | None
    phone: str | None
    education: list[Education]
    experience: list[Experience]
    skills: list[str]
    certifications: list[str]
    total_experience_years: float
    raw_text: str

    def to_dict(self) -> dict:
        """Serialize to dict for storage and scoring."""
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "education": [
                {
                    "degree": e.degree,
                    "field": e.field_of_study,
                    "institution": e.institution,
                    "year": e.year,
                }
                for e in self.education
            ],
            "experience": [
                {
                    "title": ex.title,
                    "company": ex.company,
                    "start_year": ex.start_year,
                    "end_year": ex.end_year,
                    "duration_months": ex.duration_months,
                    "description": ex.description,
                    "skills_mentioned": ex.skills_mentioned,
                }
                for ex in self.experience
            ],
            "skills": self.skills,
            "certifications": self.certifications,
            "total_experience_years": self.total_experience_years,
            "raw_text": self.raw_text,
        }


# ──────────────────────────────────────────────
# Section patterns
# ──────────────────────────────────────────────

SECTION_PATTERNS: dict[str, re.Pattern] = {
    "education": re.compile(
        r"(?i)^\s*(education|academic|qualifications?|degrees?)\s*$"
    ),
    "experience": re.compile(
        r"(?i)^\s*(experience|employment|work\s*history|"
        r"professional\s*experience|career\s*history)\s*$"
    ),
    "skills": re.compile(
        r"(?i)^\s*(skills|technical\s*skills|technologies|"
        r"proficiencies|competencies|tech\s*stack|core\s*skills)\s*$"
    ),
    "certifications": re.compile(
        r"(?i)^\s*(certifications?|licenses?|credentials?|"
        r"professional\s*development)\s*$"
    ),
    "summary": re.compile(
        r"(?i)^\s*(summary|objective|profile|about\s*me|"
        r"professional\s*summary)\s*$"
    ),
    "projects": re.compile(
        r"(?i)^\s*(projects?|personal\s*projects?|key\s*projects?)\s*$"
    ),
}

# Regex patterns for extraction
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
)
DEGREE_RE = re.compile(
    r"(?i)\b(Ph\.?D|Doctorate|M\.?S\.?|M\.?A\.?|M\.?Eng|MBA|"
    r"M\.?Tech|B\.?S\.?|B\.?A\.?|B\.?Eng|B\.?Tech|"
    r"Master(?:'s)?|Bachelor(?:'s)?|Associate(?:'s)?)\b"
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*[-–—to]+\s*((?:19|20)\d{2}|[Pp]resent|[Cc]urrent)"
)


# ──────────────────────────────────────────────
# Parser class
# ──────────────────────────────────────────────

class ResumeParser:
    """
    Extracts structured data from resume files.

    Supports PDF, DOCX, and plain text formats. Uses regex-based section
    detection + skill taxonomy matching for robust extraction without
    requiring a GPU-based NER model.
    """

    def __init__(self, spacy_nlp=None):
        """
        Args:
            spacy_nlp: Optional loaded spaCy model for NER.
                       If None, falls back to regex-only extraction.
        """
        self._nlp = spacy_nlp

    def parse(self, file_path: str | Path) -> ParsedResume:
        """Parse a resume file into structured data."""
        path = Path(file_path)
        logger.info("Parsing resume: %s", path.name)

        raw_text = self._extract_text(path)
        if not raw_text.strip():
            raise ValueError(f"No text could be extracted from {path.name}")

        sections = self._segment_sections(raw_text)
        education = self._extract_education(sections.get("education", ""))
        experience = self._extract_experience(sections.get("experience", ""))
        skills = self._extract_skills(raw_text)
        certifications = self._extract_certifications(
            sections.get("certifications", "")
        )
        total_years = self._calc_total_experience(experience)

        return ParsedResume(
            candidate_id=self._generate_id(raw_text),
            name=self._extract_name(raw_text),
            email=self._extract_email(raw_text),
            phone=self._extract_phone(raw_text),
            education=education,
            experience=experience,
            skills=sorted(skills),
            certifications=certifications,
            total_experience_years=total_years,
            raw_text=raw_text,
        )

    # ── Text extraction ─────────────────────

    def _extract_text(self, path: Path) -> str:
        """Extract raw text from PDF, DOCX, or TXT."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix in (".docx", ".doc"):
            return self._extract_docx(path)
        elif suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from a PDF using pdfplumber."""
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    def _extract_docx(self, path: Path) -> str:
        """Extract text from a DOCX file."""
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    # ── Section segmentation ────────────────

    def _segment_sections(self, text: str) -> dict[str, str]:
        """Split resume text into named sections based on header patterns."""
        sections: dict[str, list[str]] = {"header": []}
        current = "header"

        for line in text.split("\n"):
            stripped = line.strip()
            matched = False

            for section_name, pattern in SECTION_PATTERNS.items():
                if pattern.match(stripped):
                    current = section_name
                    if current not in sections:
                        sections[current] = []
                    matched = True
                    break

            if not matched:
                if current not in sections:
                    sections[current] = []
                sections[current].append(line)

        return {k: "\n".join(v) for k, v in sections.items()}

    # ── Entity extraction ───────────────────

    def _extract_name(self, text: str) -> str:
        """Extract candidate name — first try spaCy NER, fallback to heuristic."""
        if self._nlp:
            doc = self._nlp(text[:500])  # Names are in the header
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text.strip()

        # Heuristic: first non-empty line that isn't an email/phone
        for line in text.split("\n")[:5]:
            line = line.strip()
            if (
                line
                and not EMAIL_RE.search(line)
                and not PHONE_RE.search(line)
                and not any(c.isdigit() for c in line[:3])
                and len(line) < 60
            ):
                return line

        return "Unknown"

    def _extract_email(self, text: str) -> str | None:
        match = EMAIL_RE.search(text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> str | None:
        match = PHONE_RE.search(text)
        return match.group(0).strip() if match else None

    # ── Education ───────────────────────────

    def _extract_education(self, section_text: str) -> list[Education]:
        """Parse education entries from the education section."""
        if not section_text.strip():
            return []

        entries: list[Education] = []
        lines = section_text.strip().split("\n")
        current_block: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_block:
                    edu = self._parse_education_block("\n".join(current_block))
                    if edu:
                        entries.append(edu)
                    current_block = []
                continue
            current_block.append(stripped)

        # Don't forget the last block
        if current_block:
            edu = self._parse_education_block("\n".join(current_block))
            if edu:
                entries.append(edu)

        # If no blocks found, try the entire section
        if not entries:
            for match in DEGREE_RE.finditer(section_text):
                entries.append(Education(degree=match.group(1)))

        return entries

    def _parse_education_block(self, block: str) -> Education | None:
        """Parse a single education entry block."""
        degree_match = DEGREE_RE.search(block)
        if not degree_match:
            return None

        year_matches = YEAR_RE.findall(block)
        year = int(year_matches[-1]) if year_matches else None

        return Education(
            degree=degree_match.group(1),
            field_of_study="",  # Could be enhanced with NER
            institution="",     # Could be enhanced with NER
            year=year,
        )

    # ── Experience ──────────────────────────

    def _extract_experience(self, section_text: str) -> list[Experience]:
        """Parse experience entries from the experience section."""
        if not section_text.strip():
            return []

        entries: list[Experience] = []
        lines = section_text.strip().split("\n")
        current_block: list[str] = []

        for line in lines:
            stripped = line.strip()
            # New entry heuristic: line with a date range
            if DATE_RANGE_RE.search(stripped) and current_block:
                exp = self._parse_experience_block("\n".join(current_block))
                if exp:
                    entries.append(exp)
                current_block = [stripped]
            elif stripped:
                current_block.append(stripped)

        if current_block:
            exp = self._parse_experience_block("\n".join(current_block))
            if exp:
                entries.append(exp)

        return entries

    def _parse_experience_block(self, block: str) -> Experience | None:
        """Parse a single experience entry block."""
        lines = block.strip().split("\n")
        if not lines:
            return None

        # Extract date range
        date_match = DATE_RANGE_RE.search(block)
        start_year = int(date_match.group(1)) if date_match else None
        end_str = date_match.group(2) if date_match else None
        end_year = None
        if end_str and end_str.isdigit():
            end_year = int(end_str)

        # Estimate duration
        duration = None
        if start_year:
            end = end_year or 2026  # Current year approximation
            duration = (end - start_year) * 12

        # First line is usually the title or company
        title = lines[0].strip()
        # Remove the date range from the title
        if date_match:
            title = DATE_RANGE_RE.sub("", title).strip(" -–—|,")

        # Description is the remaining lines
        description = "\n".join(lines[1:]).strip()

        # Find skills mentioned in this role
        role_skills = list(find_skills_in_text(block))

        return Experience(
            title=title or "Unknown Role",
            company="",
            start_year=start_year,
            end_year=end_year,
            duration_months=duration,
            description=description,
            skills_mentioned=role_skills,
        )

    # ── Skills ──────────────────────────────

    def _extract_skills(self, text: str) -> set[str]:
        """Extract skills using taxonomy matching + optional spaCy NER."""
        found = find_skills_in_text(text)

        # Optionally augment with spaCy entities
        if self._nlp:
            doc = self._nlp(text[:5000])
            for ent in doc.ents:
                if ent.label_ in ("PRODUCT", "ORG", "SKILL"):
                    from resume_ranker.parsers.skill_taxonomy import normalize_skill
                    canonical = normalize_skill(ent.text)
                    if canonical:
                        found.add(canonical)

        return found

    # ── Certifications ──────────────────────

    def _extract_certifications(self, section_text: str) -> list[str]:
        """Extract certification names from the certifications section."""
        if not section_text.strip():
            return []
        return [
            line.strip("•-– ").strip()
            for line in section_text.split("\n")
            if line.strip("•-– ").strip() and len(line.strip()) > 3
        ]

    # ── Helpers ─────────────────────────────

    def _calc_total_experience(self, experiences: list[Experience]) -> float:
        """Sum up total experience years from parsed experience entries."""
        total_months = 0
        for exp in experiences:
            if exp.duration_months:
                total_months += exp.duration_months
            elif exp.start_year:
                end = exp.end_year or 2026
                total_months += (end - exp.start_year) * 12

        return round(total_months / 12, 1)

    def _generate_id(self, text: str) -> str:
        """Generate a deterministic candidate ID from content hash."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"cand-{digest}"
