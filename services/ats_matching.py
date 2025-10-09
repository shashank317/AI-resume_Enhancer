"""
Smarter ATS keyword matching utilities.

Features:
- Normalization and simple synonym canonicalization
- Lemma-based token matching using spaCy (with graceful fallback)
- Fuzzy matching via RapidFuzz token_set_ratio

Public API:
- normalize(text: str) -> str
- extract_keywords(job_description: str, max_keywords: int = 80) -> list[str]
- ats_match(resume_text: str, jd_keywords: list[str], threshold: int = 80) -> tuple[int, list[str], list[str]]
"""
from __future__ import annotations

from typing import List, Tuple, Set, Dict, Optional
import re

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency handled
    spacy = None  # type: ignore

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dependency handled
    fuzz = None  # type: ignore

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
        # Model not installed; degrade gracefully
        _NLP = None
    return _NLP


# A light stopword set; spaCy's stop words are used when available
_BASIC_STOP = {
    'the','a','an','to','of','and','or','in','on','for','with','by','at','from','as','is','are','was','were','be','been','being',
    'this','that','these','those','it','its','you','your','i','we','they','our','their','he','she','him','her','his','hers','them',
    'but','if','then','else','than','so','such','not','no','yes','do','does','did','done','can','could','should','would','may','might',
    'will','shall','have','has','had','having','into','about','across','over','under','per','vs','via','etc','eg','ie','an','per'
}


def normalize(text: str) -> str:
    """Lowercase and collapse hyphens/underscores and extra spaces."""
    if not text:
        return ""
    s = text.lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_SYN_MAP: list[tuple[re.Pattern, str]] = [
    # --- Software & APIs ---
    (re.compile(r"\brest\s*ful\b"), "rest"),
    (re.compile(r"\brestful apis?\b"), "rest api"),
    (re.compile(r"\bapi\b"), "api"),
    (re.compile(r"\bgraphql\b"), "graphql"),
    (re.compile(r"\bgrpc\b"), "grpc"),
    # --- Shell & OS ---
    (re.compile(r"\bbash( scripting)?\b"), "shell scripting"),
    (re.compile(r"\bshell( scripting)?\b"), "shell scripting"),
    (re.compile(r"\bsh\b|\bzsh\b"), "shell scripting"),
    # --- Databases ---
    (re.compile(r"\bpostgres(?:ql)?\b"), "postgresql"),
    (re.compile(r"\bmongo ?db\b|\bmongo\b"), "mongodb"),
    (re.compile(r"\bms sql server\b|\bsql server\b"), "mssql"),
    # --- Cloud & Platforms ---
    (re.compile(r"\bgoogle cloud platform\b|\bgcp\b"), "google cloud"),
    (re.compile(r"\bamazon web services\b|\baws\b"), "aws"),
    (re.compile(r"\bmicrosoft azure\b|\bazure\b"), "azure"),
    (re.compile(r"\bheroku\b"), "heroku"),
    (re.compile(r"\brender\b"), "render"),
    # --- Programming Languages ---
    (re.compile(r"\bjs\b|\bjava script\b"), "javascript"),
    (re.compile(r"\bts\b"), "typescript"),
    (re.compile(r"\bnode(?:\.js|js)?\b"), "node.js"),
    (re.compile(r"\breact(?:\.js|js)?\b"), "react"),
    (re.compile(r"\bvue(?:\.js|js)?\b"), "vue"),
    (re.compile(r"\bnext(?:\.js|js)?\b"), "next.js"),
    (re.compile(r"\bnest(?:\.js|js)?\b"), "nest.js"),
    (re.compile(r"\bc\+\+\b"), "c++"),
    (re.compile(r"\bc#\b"), "c#"),
    # --- DevOps / Infra ---
    (re.compile(r"\bci\s*/\s*cd\b|\bcicd\b|\bci cd\b"), "ci/cd"),
    (re.compile(r"\bk8s\b"), "kubernetes"),
    (re.compile(r"\bterraform\b"), "terraform"),
    (re.compile(r"\bansible\b"), "ansible"),
    (re.compile(r"\bjenkins\b"), "jenkins"),
    # --- AI / ML / Data ---
    (re.compile(r"\bmachine learning\b|\bml\b"), "machine learning"),
    (re.compile(r"\bnatural language processing\b|\bnlp\b"), "nlp"),
    (re.compile(r"\bartificial intelligence\b|\bai\b"), "ai"),
    (re.compile(r"\bdeep learning\b"), "deep learning"),
    (re.compile(r"\bcomputer vision\b"), "computer vision"),
    (re.compile(r"\bdata analysis\b|\bdata analytics\b"), "data analytics"),
    (re.compile(r"\bdata visuali[sz]ation\b"), "data visualization"),
    # --- Security / Cyber ---
    (re.compile(r"\bpenetration testing\b|\bpentesting\b"), "vulnerability assessment"),
    (re.compile(r"\bincident response\b|\bsecurity incident\b"), "incident response"),
    (re.compile(r"\bmalware analysis\b|\bthreat analysis\b"), "malware analysis"),
    (re.compile(r"\bvulnerability scanning\b|\bvulnerability testing\b"), "vulnerability assessment"),
    (re.compile(r"\brisk management\b|\brisk assessment\b"), "risk management"),
    (re.compile(r"\bforensics\b|\bdigital forensics\b"), "digital forensics"),
    (re.compile(r"\bsecurity operations center\b|\bsoc\b"), "soc"),
    (re.compile(r"\bintrusion detection\b|\bids\b"), "ids"),
    (re.compile(r"\bintrusion prevention\b|\bips\b"), "ips"),
    # --- Misc / Soft Skills ---
    (re.compile(r"\bcommunication skills?\b"), "communication"),
    (re.compile(r"\bproblem solving\b"), "problem solving"),
    (re.compile(r"\bteamwork\b"), "team collaboration"),
    (re.compile(r"\btime management\b"), "time management"),
]


def _canonicalize(term: str) -> str:
    s = normalize(term)
    # Apply synonym replacements
    for pat, repl in _SYN_MAP:
        s = pat.sub(repl, s)
    return s


def _lemma_tokens(text: str) -> Set[str]:
    nlp = _load_nlp()
    if nlp is not None:
        doc = nlp(text)
        toks = {t.lemma_.lower().strip() for t in doc if not (t.is_stop or t.is_punct or t.is_space)}
        # also include raw lower
        toks |= {t.text.lower().strip() for t in doc if not (t.is_punct or t.is_space)}
        return {t for t in toks if t and t not in _BASIC_STOP}
    # Fallback: regex tokenization
    toks = re.split(r"[^a-z0-9+.#]+", text.lower())
    return {t for t in toks if t and t not in _BASIC_STOP}


def _phrase_tokens(phrase: str) -> Set[str]:
    return {w for w in re.split(r"\s+", normalize(phrase)) if w}


# Family groups for broader co-matching (e.g., SQL family, cloud providers)
FAMILY_GROUPS: list[Set[str]] = [
    {"sql", "postgresql", "mysql", "mssql", "sqlite", "oracle"},
    {"aws", "azure", "google cloud", "gcp"},
    {"docker", "kubernetes", "k8s", "containerization"},
    {"bash", "shell scripting", "shell", "powershell"},
    {"rest", "rest apis", "api", "graphql", "grpc", "soap"},
]


def _group_co_match(kw_norm: str, resume_lemmas: Set[str]) -> bool:
    for group in FAMILY_GROUPS:
        if kw_norm in group and (group & resume_lemmas):
            return True
    return False


# Heuristic hints to keep only relevant JD keywords (reduce noise)
DOMAIN_HINTS = {
    # Programming & Frameworks
    "python","java","c++","c#","go","ruby","php","javascript","typescript","html","css",
    "fastapi","django","flask","spring","express","react","vue","next.js","node.js","nest.js",
    "git","github","gitlab","rest api","graphql","grpc","soap","microservices","oop","design patterns",
    # Cloud & DevOps
    "aws","azure","google cloud","gcp","render","heroku","docker","kubernetes","terraform","ansible",
    "jenkins","helm","ci/cd","linux","bash","shell scripting","powershell","monitoring","deployment",
    # Databases & Data
    "sql","nosql","postgresql","mysql","mongodb","sqlite","oracle","redis","elasticsearch","kafka",
    "etl","data pipeline","data analytics","data science","pandas","numpy","tableau","power bi",
    "data warehouse","spark","airflow","hadoop",
    # Cybersecurity
    "cyber security","information security","vulnerability assessment","penetration testing",
    "incident response","soc","siem","firewall","vpn","ids","ips","threat intelligence",
    "risk management","malware analysis","digital forensics","owasp","burp suite","nessus","nmap",
    "wireshark","metasploit","security monitoring","network security","access control","zero trust",
    "data breach","encryption","ssl","tls","web application security","security compliance",
    # AI / ML
    "machine learning","deep learning","nlp","computer vision","ai","model training","inference",
    "pytorch","tensorflow","scikit","transformers","huggingface","openai","gemini",
    # Networking / IT
    "tcp","udp","http","https","dns","ip","vpn","routing","switching","networking","firewall","load balancer",
    "wireshark","protocols","packet inspection","network architecture",
    # Soft Skills / Business
    "problem solving","team collaboration","communication","leadership","documentation",
    "project management","agile","scrum","jira","time management","client communication",
}

# Default fuzzy/semantic threshold (lowered from previous 75 to improve recall on near-matches like
# "predictive modeling" vs "predictive model"; still high enough to avoid spurious matches)
def adaptive_threshold(keyword: str) -> int:
    length = len(keyword.split())
    if length >= 3:
        return 45
    elif length == 2:
        return 55
    return 65

DEFAULT_THRESHOLD = 60  # fallback


def _is_relevant_kw(word: str) -> bool:
    w = normalize(word)
    if not w:
        return False
    if re.search(r"[+.#]", w):
        return True
    for h in DOMAIN_HINTS:
        if h in w:
            return True
    return False


SECTION_PATTERNS: Dict[str, re.Pattern] = {
    "skills": re.compile(r"\bskills?\b|\btechnical skills\b|\btech stack\b", re.I),
    "experience": re.compile(r"\bexperience\b|\bwork experience\b|\bemployment\b|\bprofessional experience\b", re.I),
    "projects": re.compile(r"\bprojects?\b|\bselected projects\b", re.I),
    "education": re.compile(r"\beducation\b|\bacademics\b", re.I),
    "certifications": re.compile(r"\bcertifications?\b|\blicenses?\b", re.I),
    "summary": re.compile(r"\bsummary\b|\bprofile\b|\bobjective\b", re.I),
}


def split_resume_sections(resume_text: str) -> Dict[str, str]:
    """Split resume text into coarse sections using header regexes. Returns lowercased text blocks."""
    text = resume_text or ""
    # Find headers with their indices
    headers: List[Tuple[str, int]] = []
    for name, pat in SECTION_PATTERNS.items():
        m = pat.search(text)
        if m:
            headers.append((name, m.start()))
    if not headers:
        return {"full": text}
    headers.sort(key=lambda x: x[1])
    sections: Dict[str, str] = {}
    for i, (name, start) in enumerate(headers):
        end = headers[i+1][1] if i+1 < len(headers) else len(text)
        sections[name] = text[start:end].strip()
    # Always include full for fallback
    sections["full"] = text
    return sections


def ats_match(resume_text: str, jd_keywords: List[str], threshold: int = DEFAULT_THRESHOLD, resume_sections: Optional[Dict[str, str]] = None,
              prefer_sectioned: bool = True) -> Tuple[int, List[str], List[str]]:
    """
    Compute ATS-style match between resume_text and a list of JD keywords/phrases.

    Returns: (score_percent, matched_keywords, missing_keywords)
    """
    full_text = _canonicalize(resume_text)
    if resume_sections is None:
        resume_sections = split_resume_sections(resume_text)
    # Section texts for targeted matching
    skills_text = _canonicalize(resume_sections.get("skills", ""))
    exp_text = _canonicalize(
        resume_sections.get("experience", "")
        + "\n" + resume_sections.get("projects", "")
        + "\n" + resume_sections.get("summary", "")
    )
    resume_lemmas = _lemma_tokens(full_text)

    matched: List[str] = []
    missing: List[str] = []

    for kw in jd_keywords:
        kw_norm = _canonicalize(kw)

        # Prefer matching in relevant sections first (skills/tasks)
        # Heuristic: if keyword looks like a tech/tool, check skills section first
        looks_like_skill = _is_relevant_kw(kw_norm)

        # Direct substring in preferred section
        targeted_texts = []
        if prefer_sectioned and looks_like_skill and skills_text:
            targeted_texts.append(skills_text)
        if prefer_sectioned and not looks_like_skill and exp_text:
            targeted_texts.append(exp_text)
        # Always add full text as fallback last
        targeted_texts.append(full_text)

        direct_hit = any(kw_norm in t for t in targeted_texts)
        if kw_norm and direct_hit:
            matched.append(kw)
            continue

        # All tokens present (semantic-ish via lemmas)
        kw_toks = _phrase_tokens(kw_norm)
        if kw_toks and kw_toks.issubset(resume_lemmas):
            matched.append(kw)
            continue

        # Family group co-match (e.g., SQL family, cloud family)
        if _group_co_match(kw_norm, resume_lemmas):
            matched.append(kw)
            continue

        # Fuzzy match: try against targeted sections first, then full
        if fuzz is not None:
            score = 0
            for t in targeted_texts:
                try:
                    score = max(score, fuzz.token_set_ratio(kw_norm, t))
                except Exception:
                    pass
            thresh = adaptive_threshold(kw_norm)
            if score >= thresh:
                matched.append(kw)
                continue

        missing.append(kw)

    score_pct = int(round((len(matched) / max(1, len(jd_keywords))) * 100)) if jd_keywords else 0
    return score_pct, matched, missing


def extract_keywords(job_description: str, max_keywords: int = 80) -> List[str]:
    """
    Rough keyword extraction from a JD: prioritize noun chunks, tech-looking tokens, and comma/line-separated phrases.
    Returns up to max_keywords keywords/phrases (original surface forms where possible).
    """
    text = job_description or ""
    nlp = _load_nlp()

    candidates: list[str] = []

    # Split on common separators to capture listed skills/phrases
    for part in re.split(r"[\n\r;,\/]+", text):
        p = part.strip()
        if len(p) > 2:
            candidates.append(p)

    if nlp is not None:
        doc = nlp(text)
        # noun chunks as phrases
        for chunk in getattr(doc, "noun_chunks", []):  # type: ignore[attr-defined]
            t = chunk.text.strip()
            if len(t) > 2:
                candidates.append(t)
        # Proper nouns and potential tech tokens
        for t in doc:
            if t.is_stop or t.is_punct or t.is_space:
                continue
            if t.pos_ in {"PROPN", "NOUN", "ADJ"} or re.search(r"[+.#]", t.text):
                candidates.append(t.text)
    else:
        # Fallback: regex tokens with tech hints
        tokens = re.split(r"[^a-z0-9+.#]+", text.lower())
        for t in tokens:
            if t and t not in _BASIC_STOP and len(t) > 2:
                candidates.append(t)

    # Normalize and canonicalize; keep unique order
    seen: Set[str] = set()
    keywords: list[str] = []
    for c in candidates:
        norm = _canonicalize(c)
        # prune very short or numeric-only
        if not norm or len(norm) < 2 or re.fullmatch(r"\d+", norm):
            continue
        # Relevance filter to avoid over-extraction
        if not _is_relevant_kw(norm):
            continue
        if norm not in seen:
            seen.add(norm)
            keywords.append(norm)
        if len(keywords) >= max_keywords:
            break

    return keywords


def extract_keywords_structured(job_description: str, max_keywords: int = 80) -> Dict[str, List[str]]:
    """Return structured JD keywords with 'skills' and 'tasks' lists after filtering and canonicalization."""
    text = job_description or ""
    nlp = _load_nlp()
    skills: list[str] = []
    tasks: list[str] = []

    # Seed from simple splits
    parts = [p.strip() for p in re.split(r"[\n\r;,/]+", text) if len(p.strip()) > 2]

    if nlp is not None:
        doc = nlp(text)
        for chunk in getattr(doc, "noun_chunks", []):  # type: ignore[attr-defined]
            if len(chunk.text) > 2:
                parts.append(chunk.text)
        for t in doc:
            if t.is_stop or t.is_punct or t.is_space:
                continue
            if t.pos_ in {"VERB", "AUX", "ADV"}:
                parts.append(t.lemma_)
            elif t.pos_ in {"PROPN", "NOUN", "ADJ"} or re.search(r"[+.#]", t.text):
                parts.append(t.text)
    else:
        tokens = re.split(r"[^a-z0-9+.#]+", text.lower())
        parts.extend([t for t in tokens if t and t not in _BASIC_STOP and len(t) > 2])

    seen_s: Set[str] = set()
    seen_t: Set[str] = set()
    for p in parts:
        norm = _canonicalize(p)
        if not norm or not _is_relevant_kw(norm):
            continue
        # crude classification: multi-word with verbs or ending with -ing to tasks; else skills
        if re.search(r"\b(build|develop|design|implement|deploy|manage|lead|configure|monitor|automate|secure|test|analy[sz]e)\b", norm) or norm.endswith("ing"):
            if norm not in seen_t:
                tasks.append(norm)
                seen_t.add(norm)
        else:
            if norm not in seen_s:
                skills.append(norm)
                seen_s.add(norm)
        if len(skills) >= max_keywords and len(tasks) >= max_keywords:
            break

    return {
        "skills": skills[:max_keywords],
        "tasks": tasks[:max_keywords],
    }
