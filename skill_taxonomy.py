"""
Curated skill taxonomy with canonical names, synonyms, abbreviations,
and related terms for accurate resume ↔ JD matching.
"""

# Canonical skill name → set of known aliases (all lowercase)
SKILL_TAXONOMY: dict[str, set[str]] = {
    # ── Programming Languages ──
    "python": {"py", "python3", "python 3", "cpython"},
    "javascript": {"js", "es6", "es2015", "es2020", "ecmascript", "vanilla js"},
    "typescript": {"ts"},
    "java": {"j2ee", "j2se", "java se", "java ee"},
    "c++": {"cpp", "c plus plus"},
    "c#": {"csharp", "c sharp", "dotnet c#"},
    "go": {"golang"},
    "rust": set(),
    "ruby": {"rb"},
    "php": set(),
    "scala": set(),
    "kotlin": {"kt"},
    "swift": set(),
    "r": {"r language", "r programming"},
    "sql": {"structured query language"},
    "bash": {"shell", "shell scripting", "sh", "zsh"},

    # ── ML / AI ──
    "machine learning": {"ml", "machine-learning"},
    "deep learning": {"dl", "deep-learning"},
    "natural language processing": {"nlp", "natural-language-processing"},
    "computer vision": {"cv", "image recognition"},
    "reinforcement learning": {"rl"},
    "large language models": {"llm", "llms", "large language model"},
    "generative ai": {"gen ai", "genai"},

    # ── ML Frameworks ──
    "tensorflow": {"tf"},
    "pytorch": {"torch"},
    "scikit-learn": {"sklearn", "scikit learn"},
    "keras": set(),
    "hugging face": {"huggingface", "hf", "transformers library"},
    "spacy": set(),
    "xgboost": {"xgb"},
    "lightgbm": {"lgbm"},
    "pandas": set(),
    "numpy": set(),
    "opencv": {"cv2"},

    # ── Cloud & Infrastructure ──
    "amazon web services": {"aws"},
    "google cloud platform": {"gcp", "google cloud"},
    "microsoft azure": {"azure"},
    "docker": {"containerization", "containers"},
    "kubernetes": {"k8s", "kube"},
    "terraform": {"tf", "iac"},
    "ansible": set(),
    "ci/cd": {"cicd", "ci cd", "continuous integration", "continuous delivery"},
    "jenkins": set(),
    "github actions": {"gha"},
    "gitlab ci": set(),

    # ── Databases ──
    "postgresql": {"postgres", "psql", "pg"},
    "mysql": set(),
    "mongodb": {"mongo"},
    "redis": set(),
    "elasticsearch": {"es", "elastic"},
    "apache cassandra": {"cassandra"},
    "dynamodb": {"dynamo db"},
    "sqlite": set(),
    "neo4j": set(),

    # ── Web Frameworks ──
    "react": {"reactjs", "react.js"},
    "angular": {"angularjs", "angular.js"},
    "vue.js": {"vue", "vuejs"},
    "next.js": {"nextjs", "next"},
    "node.js": {"nodejs", "node"},
    "express.js": {"express", "expressjs"},
    "django": set(),
    "flask": set(),
    "fastapi": {"fast api"},
    "spring boot": {"spring", "springboot"},
    "ruby on rails": {"rails", "ror"},

    # ── Data Engineering ──
    "apache spark": {"spark", "pyspark"},
    "apache kafka": {"kafka"},
    "apache airflow": {"airflow"},
    "hadoop": {"hdfs", "mapreduce"},
    "dbt": {"data build tool"},
    "snowflake": set(),
    "databricks": set(),
    "etl": {"extract transform load"},

    # ── DevOps / SRE ──
    "linux": {"unix", "centos", "ubuntu", "debian", "rhel"},
    "monitoring": {"observability"},
    "prometheus": set(),
    "grafana": set(),
    "datadog": set(),
    "splunk": set(),

    # ── Methodologies ──
    "agile": {"scrum", "kanban"},
    "test-driven development": {"tdd"},
    "microservices": {"micro-services", "microservice architecture"},
    "rest api": {"rest", "restful", "restful api"},
    "graphql": {"gql"},
    "grpc": set(),
    "event-driven architecture": {"eda", "event driven"},

    # ── Security ──
    "cybersecurity": {"infosec", "information security"},
    "oauth": {"oauth2", "oauth 2.0"},
    "jwt": {"json web token"},
    "ssl/tls": {"ssl", "tls", "https"},
    "soc 2": {"soc2"},

    # ── Other ──
    "git": {"github", "gitlab", "version control"},
    "jira": set(),
    "confluence": set(),
    "figma": set(),
    "tableau": set(),
    "power bi": {"powerbi"},
}


def build_reverse_index() -> dict[str, str]:
    """
    Build a reverse lookup: alias → canonical skill name.

    Returns a dict where each key is a lowercase alias and the value
    is the canonical skill name.
    """
    reverse: dict[str, str] = {}
    for canonical, aliases in SKILL_TAXONOMY.items():
        canonical_lower = canonical.lower()
        reverse[canonical_lower] = canonical_lower
        for alias in aliases:
            reverse[alias.lower()] = canonical_lower
    return reverse


# Pre-built reverse index for fast lookups
ALIAS_TO_CANONICAL: dict[str, str] = build_reverse_index()


def normalize_skill(raw: str) -> str | None:
    """
    Normalize a raw skill mention to its canonical form.

    Returns the canonical name if found, else None.
    """
    return ALIAS_TO_CANONICAL.get(raw.strip().lower())


def find_skills_in_text(text: str) -> set[str]:
    """
    Scan free text for known skills using the taxonomy.

    Returns a set of canonical skill names found in the text.
    Matches multi-word skills and single-word aliases.
    """
    text_lower = text.lower()
    found: set[str] = set()

    # Check all aliases (including canonical names) against the text
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        # Use simple substring matching with word-boundary awareness
        # For very short aliases (≤2 chars), require word boundaries
        if len(alias) <= 2:
            import re
            if re.search(rf"\b{re.escape(alias)}\b", text_lower):
                found.add(canonical)
        else:
            if alias in text_lower:
                found.add(canonical)

    return found
