from __future__ import annotations

from typing import Dict, List, Optional
import re

try:
    import spacy  # type: ignore
except Exception:
    spacy = None  # type: ignore

_NLP = None


def _load_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    if spacy is None:
        _NLP = None
        return None
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = None
    return _NLP


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        l = line.strip()
        if l:
            return l
    return ""


def _guess_role_title(text: str) -> str:
    # Heuristics: look for explicit title labels, else first non-empty line (trim noisy prefixes)
    candidates = []
    for pat in [
        r"(?im)^\s*(role|position|job title)\s*[:\-]\s*(.+)$",
        r"(?im)^\s*title\s*[:\-]\s*(.+)$",
    ]:
        m = re.search(pat, text)
        if m:
            candidates.append(m.group(2).strip())
    if candidates:
        return candidates[0]
    first = _first_nonempty_line(text)
    # Remove company/location suffixes separated by "-" or "|"
    first = re.split(r"\s[\-|\|]\s", first)[0].strip()
    return first


def _extract_experience_level(text: str) -> str:
    # Capture "X+ years", and seniority words
    yrs = re.findall(r"(\d+\s*\+?\s*years?)", text, re.I)
    seniority = None
    if re.search(r"\bsenior|lead|principal|staff\b", text, re.I):
        seniority = "Senior"
    elif re.search(r"\b(junior|entry)\b", text, re.I):
        seniority = "Junior"
    return (", ".join(sorted(set(yrs))) + ("; " + seniority if seniority else "")) if yrs else (seniority or "")


def _extract_education(text: str) -> List[str]:
    edu_patterns = [
        r"b\.?tech|bachelor|b\.s\.|bs\b|be\b|bsc",
        r"m\.?tech|master|m\.s\.|ms\b|msc|mca",
        r"phd|doctorate",
        r"diploma|degree",
        r"computer science|information technology|engineering",
    ]
    found = set()
    for pat in edu_patterns:
        for m in re.finditer(pat, text, re.I):
            found.add(m.group(0).lower())
    return sorted(found)


def _extract_responsibilities(text: str) -> List[str]:
    # Look for responsibilities / duties sections and bullet lines
    items: List[str] = []
    sections = re.split(r"(?im)^\s*(responsibilities|duties|what you will do|about the role)\s*$", text)
    segment = text
    if len(sections) > 1:
        # Prefer the text after the first header
        segment = sections[2] if len(sections) >= 3 else sections[1]
    for line in segment.splitlines():
        l = line.strip()
        if re.match(r"^[\-*•]\s+|^(\d+\.)\s+", l):
            items.append(re.sub(r"^[\-*•\d\.]+\s*", "", l))
    # Fallback: sentences with verbs like build/develop/implement
    if not items:
        items = [s.strip() for s in re.split(r"[\.!?]\s+", segment) if re.search(r"\b(build|develop|design|implement|deploy|manage|lead|configure|monitor|automate|secure|test|analy[sz]e)\b", s, re.I)]
    return items[:15]


def extract_jd_core_info(job_description: str) -> Dict[str, object]:
    """Extract core info from a job description: role title, responsibilities, skills/tools, experience level, education, and keywords/buzzwords."""
    text = job_description or ""
    nlp = _load_nlp()

    role_title = _guess_role_title(text)
    responsibilities = _extract_responsibilities(text)
    experience_level = _extract_experience_level(text)
    education = _extract_education(text)

    # Skills/tools and keywords
    skills: List[str] = []
    keywords: List[str] = []
    try:
        from .ats_matching import extract_keywords, extract_keywords_structured
        structured = extract_keywords_structured(text, max_keywords=80)
        skills = structured.get("skills", [])
        # Additional keywords beyond skills (tasks etc.)
        keywords = extract_keywords(text, max_keywords=100)
    except Exception:
        # minimal fallback: pick word-like tokens with tech hints by regex
        tokens = re.findall(r"[a-zA-Z0-9+.#]{3,}", text.lower())
        keywords = sorted(set(tokens))[:80]

    return {
        "role_title": role_title,
        "responsibilities": responsibilities,
        "skills_tools": skills,
        "experience_level": experience_level,
        "education": education,
        "keywords": keywords,
    }
