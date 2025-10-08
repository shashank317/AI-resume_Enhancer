from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from httpx import HTTPError

try:  # Prefer config module but allow fallback when running tests standalone
    from config import (
        MODEL_NAME,
        OPENROUTER_API_KEY,
        GEMINI_API_KEY,  # newly added
        get_gemini_models,
    )
except ImportError:  # pragma: no cover - defensive fallback
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:free")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    def get_gemini_models() -> List[str]:  # minimal fallback
        raw = os.getenv("GEMINI_MODEL_PRIORITY", "")
        return [m.strip() for m in raw.split(",") if m.strip()]

try:
    from .ats_matching import (
        ats_match,
        extract_keywords,
        extract_keywords_structured,
        split_resume_sections,
    )
except Exception:  # pragma: no cover - provide resilient fallbacks
    logging.warning("services.ats_matching unavailable; using simplified keyword logic.")

    def split_resume_sections(resume_text: str) -> Dict[str, str]:
        text = resume_text or ""
        return {"full": text}

    def extract_keywords(job_description: str, max_keywords: int = 80) -> List[str]:
        tokens = re.split(r"[^a-z0-9+#.]+", (job_description or "").lower())
        seen: List[str] = []
        for tok in tokens:
            if tok and tok not in seen:
                seen.append(tok)
            if len(seen) >= max_keywords:
                break
        return seen

    def extract_keywords_structured(job_description: str, max_keywords: int = 80) -> Dict[str, List[str]]:
        kws = extract_keywords(job_description, max_keywords)
        return {"skills": kws, "tasks": kws}

    def ats_match(resume_text: str, jd_keywords: List[str], threshold: int = 75, resume_sections: Optional[Dict[str, str]] = None,
                  prefer_sectioned: bool = True) -> Tuple[int, List[str], List[str]]:
        resume_words = set(re.split(r"[^a-z0-9+#.]+", (resume_text or "").lower()))
        matched = [kw for kw in jd_keywords if kw.lower() in resume_words]
        missing = [kw for kw in jd_keywords if kw.lower() not in resume_words]
        score = int(round(100 * len(matched) / max(1, len(jd_keywords)))) if jd_keywords else 0
        return score, matched, missing


DEFAULT_MODEL_SEQUENCE: List[str] = []
for candidate in (
    # First: user-specified / primary configured model
    MODEL_NAME,
    # Paid / higher quality (will be skipped silently if not accessible under key tier)

    "google/gemini-pro-1.5",
    # Existing general fallbacks
    "deepseek/deepseek-chat",
    "openai/gpt-oss-20b:free",
    # Newly added free-tier cascade (requested) — tried in this order if earlier fail
    "deepseek/deepseek-chat-v3.1:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "z-ai/glm-4.5-air:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1:free",
    "microsoft/mai-ds-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-4-maverick:free",
):
    if candidate and candidate not in DEFAULT_MODEL_SEQUENCE:
        DEFAULT_MODEL_SEQUENCE.append(candidate)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_REMOTE_ATTEMPTS = 4

SUMMARY_SENTENCE_LIMIT = 3
FALLBACK_SUMMARY_LENGTH = 240
MAX_HEURISTIC_BULLETS = 6
DEFAULT_ATS_THRESHOLD = 75

_METRIC_PLACEHOLDER_PATTERN = re.compile(r"\[(?:METRIC|METRIC_REQUIRED)\]")
_EDUCATION_PATTERNS = (
    r"b\.?(?:s|sc|tech)\b",
    r"m\.?(?:s|sc|tech)\b",
    r"bachelor",
    r"master",
    r"phd",
    r"doctorate",
    r"computer science",
    r"information technology",
)


@dataclass
class ModelAttempt:
    model: str
    status: str
    detail: Optional[str] = None

    def as_dict(self) -> Dict[str, str]:
        payload = {"model": self.model, "status": self.status}
        if self.detail:
            payload["detail"] = self.detail
        return payload


## (Removed duplicate OpenRouterClient definition)


class OpenRouterClient:
    """Thin wrapper around OpenRouter chat completion endpoint."""

    def __init__(self, api_key: Optional[str], models: Sequence[str]):
        self.api_key = api_key
        self.models = [m for m in models if m]

    def chat(self,
             model: str,
             messages: Sequence[Dict[str, str]],
             max_tokens: int,
             temperature: float) -> Tuple[Optional[str], ModelAttempt]:
        if not self.api_key:
            return None, ModelAttempt(model, "api_key_missing", "OPENROUTER_API_KEY not configured")

        attempt = ModelAttempt(model, "request")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "AI Resume Optimizer",
        }
        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(60)) as client:
                response = client.post(OPENROUTER_URL, headers=headers, json=payload)
                if response.status_code >= 400:
                    text = response.text[:200]
                    attempt.status = "http_error"
                    attempt.detail = f"{response.status_code}: {text}"
                    return None, attempt
                data = response.json()
        except HTTPError as exc:  # pragma: no cover - network error
            attempt.status = "network_error"
            attempt.detail = str(exc)
            return None, attempt
        except Exception as exc:  # pragma: no cover - JSON parsing issues
            attempt.status = "unexpected_error"
            attempt.detail = str(exc)
            return None, attempt

        choices = data.get("choices") or []
        if not choices:
            attempt.status = "empty_response"
            return None, attempt

        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        if not isinstance(content, str):
            attempt.status = "invalid_content"
            return None, attempt

        attempt.status = "success"
        attempt.detail = f"chars={len(content)}"
        return content.strip(), attempt


class GeminiClient:
    """Minimal Gemini Generative Language API client (chat-like wrapper).

    We flatten the role-based messages into a single user turn because Gemini's public
    API (v1beta generateContent) expects a list of content parts. This keeps parity
    with our OpenRouterClient.chat interface.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: Optional[str], models: Sequence[str]):
        self.api_key = api_key
        self.models = [m for m in models if m]

    def chat(self,
             model: str,
             messages: Sequence[Dict[str, str]],
             max_tokens: int,
             temperature: float) -> Tuple[Optional[str], ModelAttempt]:
        if not self.api_key:
            return None, ModelAttempt(model, "api_key_missing", "GEMINI_API_KEY not configured")

        attempt = ModelAttempt(model, "request")
        # Separate system + user for better instruction adherence
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        user_parts = [m.get("content", "") for m in messages if m.get("role") == "user"]

        system_text = "\n\n".join(system_parts).strip()
        user_text = "\n\n".join(user_parts).strip()

        contents: List[Dict[str, object]] = []
        if system_text:
            # Some Gemini model variants accept 'system' role; if rejected, API returns 400 and we will retry below without it.
            contents.append({"role": "system", "parts": [{"text": system_text}]})
        if user_text:
            contents.append({"role": "user", "parts": [{"text": user_text}]})
        if not contents:  # fallback to merged
            merged_text = "\n\n".join(m.get("content", "") for m in messages)
            contents = [{"role": "user", "parts": [{"text": merged_text}]}]

        url = f"{self.BASE_URL}/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(60)) as client:
                response = client.post(url, json=payload)
                if response.status_code >= 400:
                    # Retry once without system role if that's the cause
                    if any(c.get("role") == "system" for c in contents):
                        alt_contents = [c for c in contents if c.get("role") != "system"]
                        if alt_contents:
                            response = client.post(url, json={"contents": alt_contents, "generationConfig": payload["generationConfig"]})
                    if response.status_code >= 400:
                        attempt.status = "http_error"
                        attempt.detail = f"{response.status_code}: {response.text[:160]}"
                        return None, attempt
                data = response.json()
        except HTTPError as exc:  # pragma: no cover
            attempt.status = "network_error"
            attempt.detail = str(exc)
            return None, attempt
        except Exception as exc:  # pragma: no cover
            attempt.status = "unexpected_error"
            attempt.detail = str(exc)
            return None, attempt

        candidates = data.get("candidates") or []
        if not candidates:
            attempt.status = "empty_response"
            return None, attempt
        parts = candidates[0].get("content", {}).get("parts") or []
        text_parts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
        content = "\n".join(text_parts).strip()
        if not content:
            attempt.status = "invalid_content"
            return None, attempt
        attempt.status = "success"
        attempt.detail = f"chars={len(content)}"
        return content, attempt


class ResumePromptBuilder:
    """Generates prompts for the optimizer."""

    @staticmethod
    def optimisation_messages(resume_text: str,
                              job_description: str,
                              preserve_template: bool) -> List[Dict[str, str]]:
        preserve_block = (
            "Maintain the same section order, layout, and formatting as the original resume. Rewrite only the content inside each section (Summary, Skills, Projects, etc.)."
            if preserve_template else
            "You may restructure sections slightly for better clarity or ATS optimization."
        )

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert ATS resume optimizer and professional recruiter. "
                "Your job is to enhance resumes for *any* job description by tailoring content intelligently, not copying it. "
                "Always keep the resume realistic, human, and concise. Think like a hiring manager reading 200 resumes — clarity, impact, and alignment matter."
            ),
        }

        # Build user content with escaped braces for literal JSON structure
        user_content = (
            "Your task: analyze the following resume and job description, then output enhanced content.\n\n"
            "Return valid JSON only:\n"
            "{{\n"
            "  \"enhanced_resume\": string,\n"
            "  \"ats_breakdown\": {{\n"
            "    \"Skills Match\": int,\n"
            "    \"Experience Relevance\": int,\n"
            "    \"Education Alignment\": int,\n"
            "    \"Keyword Coverage\": int\n"
            "  }},\n"
            "  \"matched_keywords\": [string],\n"
            "  \"missing_keywords\": [string],\n"
            "  \"feedback\": {{\n"
            "    \"strengths\": [string],\n"
            "    \"weaknesses\": [string],\n"
            "    \"suggestions\": [string]\n"
            "  }},\n"
            "  \"interview_questions\": [string]\n"
            "}}\n\n"
            "Rules:\n"
            "- Read and understand the job description deeply, infer the key skills, mindset, and responsibilities required.\n"
            "- Rewrite the Summary, Skills, and Project/Experience sections in a natural, role-adaptive way.\n"
            "- The rewritten text must sound like the candidate already fits that role, but DO NOT copy or restate JD lines.\n"
            "- Instead, rephrase in your own words and connect transferable experience to the JD.\n"
            "- If the JD is in a specialized field (e.g., cybersecurity, finance, marketing, ML, etc.), adjust language naturally — let your AI reasoning adapt to that role.\n"
            "- Use professional, confident tone — avoid fluff.\n"
            "- You can infer implied strengths (e.g., problem-solving, teamwork, adaptability) from context.\n"
            "- Focus on ATS readability and natural keyword embedding.\n"
            f"- {preserve_block}\n"
            "- If data is missing (e.g., metrics or company names), use placeholders like [METRIC] or [PROJECT].\n\n"
            "RESUME (Original):\n{resume}\n\n"
            "JOB DESCRIPTION (for context):\n{jd}\n"
        )

        user_msg = {
            "role": "user",
            "content": user_content.format(resume=resume_text, jd=job_description).strip(),
        }
        return [system_msg, user_msg]


class AIResponseParser:
    """Parses JSON returned by the model."""

    REQUIRED_KEYS = ("enhanced_resume", "ats_breakdown", "matched_keywords", "missing_keywords", "feedback")

    def parse(self, payload: str) -> Optional[Dict[str, object]]:
        candidate = self._try_parse(payload)
        if candidate is None:
            return None
        result = self._normalise(candidate)
        if not result["enhanced_resume"]:
            return None
        return result

    def _try_parse(self, raw: str) -> Optional[Dict[str, object]]:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None

    def _normalise(self, payload: Dict[str, object]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        enhanced = str(payload.get("enhanced_resume") or "").strip()
        result["enhanced_resume"] = sanitize_resume_text(enhanced)

        ats = payload.get("ats_breakdown") or {}
        result["ats_breakdown"] = self._coerce_mapping(ats)

        result["matched_keywords"] = self._as_list(payload.get("matched_keywords"))
        result["missing_keywords"] = self._as_list(payload.get("missing_keywords"))

        feedback = payload.get("feedback") or {}
        fb = {
            "strengths": self._as_list(getattr(feedback, "get", lambda _: [])("strengths") if isinstance(feedback, dict) else []),
            "weaknesses": self._as_list(getattr(feedback, "get", lambda _: [])("weaknesses") if isinstance(feedback, dict) else []),
            "suggestions": self._as_list(getattr(feedback, "get", lambda _: [])("suggestions") if isinstance(feedback, dict) else []),
        }
        result["feedback"] = fb

        interview = payload.get("interview_questions")
        if interview:
            result["interview_questions"] = self._as_list(interview)

        for key in self.REQUIRED_KEYS:
            result.setdefault(key, [] if key.endswith("keywords") else {} if key in {"ats_breakdown", "feedback"} else "")
        return result

    @staticmethod
    def _as_list(value: object) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _coerce_mapping(value: object) -> Dict[str, int]:
        if isinstance(value, dict):
            out: Dict[str, int] = {}
            for key, val in value.items():
                try:
                    out[str(key)] = int(float(val))
                except Exception:
                    continue
            return out
        return {}


class HeuristicOptimizer:
    """Local fallback when remote AI is unavailable."""

    def optimise(self,
                 resume_text: str,
                 job_description: str,
                 user_metrics: Optional[Dict[str, str]],
                 preserve_template: bool) -> Dict[str, object]:
        jd_struct = extract_keywords_structured(job_description or "", max_keywords=80)
        jd_keywords = jd_struct.get("skills", []) + jd_struct.get("tasks", [])
        if not jd_keywords:
            jd_keywords = extract_keywords(job_description or "", max_keywords=40)

        sections = split_resume_sections(resume_text or "")
        full_text = sections.get("full") or resume_text or ""

        summary = self._build_summary(full_text, job_description)
        skills_block = self._build_skills_block(full_text, jd_struct)
        experience_block = self._build_experience_block(sections)
        education_block = self._build_education_block(full_text)

        if preserve_template and resume_text.strip():
            enhanced_body = resume_text.strip()
        else:
            parts = [segment for segment in (
                "SUMMARY\n" + summary if summary else None,
                skills_block,
                experience_block,
                education_block,
            ) if segment]
            enhanced_body = "\n\n".join(parts) or resume_text or job_description or "Optimized resume"

        enhanced_body = sanitize_resume_text(enhanced_body)
        enhanced_body = apply_metric_overrides(enhanced_body, user_metrics)

        ats_score, matched, missing = ats_match(enhanced_body, jd_keywords, threshold=DEFAULT_ATS_THRESHOLD,
                                               resume_sections=sections)

        skills_matches = [kw for kw in matched if kw in jd_struct.get("skills", [])]
        task_matches = [kw for kw in matched if kw in jd_struct.get("tasks", [])]
        skills_total = max(1, len(jd_struct.get("skills", []) or matched))
        task_total = max(1, len(jd_struct.get("tasks", []) or matched))
        ats_breakdown = {
            "Skills Match": int(round(100 * len(skills_matches) / skills_total)),
            "Experience Relevance": int(round(100 * len(task_matches) / task_total)),
            "Education Alignment": self._education_alignment_score(full_text, job_description),
            "Keyword Coverage": ats_score,
        }

        feedback = self._build_feedback(summary, matched, missing, job_description)
        interview_questions = [f"Can you describe your experience with {kw}?" for kw in missing[:3]]

        return {
            "enhanced_resume": enhanced_body,
            "ats_breakdown": ats_breakdown,
            "matched_keywords": matched,
            "missing_keywords": missing,
            "feedback": feedback,
            "interview_questions": interview_questions,
            "source": "heuristic",
        }

    @staticmethod
    def _build_summary(resume_text: str, job_description: str) -> str:
        """
        Build a generic, transferable summary. Incorporates top JD keywords if available.
        """
        summary_candidates: List[str] = []

        # Use first 3 meaningful lines from resume text
        for line in resume_text.splitlines():
            if len(summary_candidates) >= SUMMARY_SENTENCE_LIMIT:
                break
            stripped = line.strip()
            if stripped and len(stripped) > 8:
                summary_candidates.append(stripped)

        # Highlight 3-4 top keywords from job description (generic)
        jd_highlights = extract_keywords(job_description or "", max_keywords=6)
        if jd_highlights:
            summary_candidates.append("Highlights: " + ", ".join(jd_highlights[:4]))

        combined = " ".join(summary_candidates)
        return combined[:FALLBACK_SUMMARY_LENGTH].strip()

    @staticmethod
    def _build_skills_block(resume_text: str, jd_struct: Dict[str, List[str]]) -> Optional[str]:
        """
        Build a transferable key skills section for any role.
        """
        # Merge skills from resume and JD
        skills = jd_struct.get("skills", []) + jd_struct.get("tasks", [])
        seen = set()
        filtered_skills: List[str] = []

        for kw in skills:
            low = kw.lower()
            if low in seen:
                continue
            seen.add(low)
            filtered_skills.append(kw)
            if len(filtered_skills) >= 15:
                break

        if not filtered_skills:
            # Fallback: extract top meaningful tokens from resume
            tokens = re.split(r"[^a-z0-9+#.]+", resume_text.lower())
            filtered_skills = list(dict.fromkeys(t for t in tokens if t and len(t) > 2))[:10]

        bullets = "\n".join(f"- {skill}" for skill in filtered_skills)
        return "KEY SKILLS\n" + bullets if bullets else None

    @staticmethod
    def _build_experience_block(sections: Dict[str, str]) -> Optional[str]:
        """
        Extract experience/projects in a generic, role-agnostic way.
        """
        for key, content in sections.items():
            if not content.strip():
                continue
            bullets: List[str] = []
            for line in content.splitlines():
                cleaned = line.strip()
                if not cleaned:
                    continue
                if not cleaned.startswith("-"):
                    cleaned = f"- {cleaned}"
                bullets.append(cleaned)
                if len(bullets) >= MAX_HEURISTIC_BULLETS:
                    break
            if bullets:
                return "EXPERIENCE / PROJECTS\n" + "\n".join(bullets)
        return None

    @staticmethod
    def _build_education_block(resume_text: str) -> Optional[str]:
        lines = [ln.strip() for ln in resume_text.splitlines() if any(re.search(pat, ln, re.I) for pat in _EDUCATION_PATTERNS)]
        if not lines:
            return None
        return "EDUCATION\n" + "\n".join(f"- {line}" if not line.startswith("-") else line for line in lines[:4])

    @staticmethod
    def _education_alignment_score(resume_text: str, job_description: str) -> int:
        resume_hits = sum(bool(re.search(pat, resume_text, re.I)) for pat in _EDUCATION_PATTERNS)
        jd_hits = sum(bool(re.search(pat, job_description, re.I)) for pat in _EDUCATION_PATTERNS)
        if jd_hits == 0:
            return 100 if resume_hits else 70
        return int(round(100 * min(resume_hits, jd_hits) / max(1, jd_hits)))

    @staticmethod
    def _build_feedback(summary: str, matched: List[str], missing: List[str], job_description: str) -> Dict[str, List[str]]:
        strengths = []
        if summary:
            strengths.append("Includes a concise summary highlighting relevant tools.")
        if matched:
            strengths.append(f"Covers {len(matched)} job-aligned keywords such as {', '.join(matched[:3])}.")
        weaknesses = []
        if missing:
            weaknesses.append(f"Missing coverage for {', '.join(missing[:3])}.")
        else:
            weaknesses.append("Consider adding quantified metrics to emphasise achievements.")
        suggestions = []
        for kw in missing[:2]:
            suggestions.append(f"Add a bullet describing practical experience with {kw} or related accomplishments.")
        if not suggestions:
            suggestions.append("Review each section for opportunities to add measurable impact statements.")

        # Domain-aware (but template-preserving) guidance.
        jd_lower = (job_description or "").lower()
        security_indicators = (
            "cyber", "infosec", "security", "siem", "threat", "vulnerability",
            "incident response", "cloudsek", "owasp", "zero trust", "soc2", "nist"
        )
        if any(tok in jd_lower for tok in security_indicators):
            suggestions.append(
                "Add 1–2 bullets with concrete security impact (e.g., reduced incident MTTR, patched CVEs, implemented SIEM dashboards)."
            )
            if not any("security" in s.lower() for s in strengths):
                strengths.append("Strong transferable foundation; could further emphasise security-specific outcomes.")

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
        }


def sanitize_resume_text(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s+$", "", cleaned, flags=re.M)
    return cleaned.strip()


def apply_metric_overrides(text: str, user_metrics: Optional[Dict[str, str]]) -> str:
    if not text or not user_metrics:
        return text
    replacement_values = [str(v) for v in user_metrics.values() if str(v).strip()]
    if not replacement_values:
        return text
    def replacer(match_iter: Iterable[re.Match]) -> str:
        remaining = replacement_values.copy()
        chars = list(text)
        for match in match_iter:
            if not remaining:
                break
            start, end = match.span()
            value = remaining.pop(0)
            chars[start:end] = list(value)
        return "".join(chars)
    matches = list(_METRIC_PLACEHOLDER_PATTERN.finditer(text))
    if not matches:
        return text
    return replacer(matches)


def format_resume_as_html(plain: str) -> str:
    """Convert plain-text enhanced resume into simple semantic HTML.

    Rules:
      - Detect section headings (lines in ALL CAPS or ending with ':') -> <h3>
      - Bullet lines starting with -, *, • -> <li>
      - Consecutive bullet lines grouped into <ul> blocks
      - Blank lines create paragraph breaks
      - Other lines become <p>
    Very lightweight; meant for safe rendering (no inline styles).
    """
    if not plain:
        return "<p>(empty)</p>"
    lines = plain.splitlines()
    html_parts: List[str] = []
    bullets: List[str] = []
    heading_pat = re.compile(r"^[A-Z0-9 &/()'.,-]{3,}$")
    bullet_pat = re.compile(r"^\s*[-*•]\s+(.*)$")

    def flush_bullets():
        nonlocal bullets
        if bullets:
            html_parts.append("<ul>" + "".join(f"<li>{re.escape(b)}</li>" for b in bullets) + "</ul>")
            bullets = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush_bullets()
            continue
        m = bullet_pat.match(line)
        if m:
            bullets.append(m.group(1).strip())
            continue
        # Non-bullet line
        flush_bullets()
        stripped = line.strip()
        if heading_pat.match(stripped) and len(stripped.split()) <= 7:
            html_parts.append(f"<h3>{re.escape(stripped)}</h3>")
        else:
            html_parts.append(f"<p>{re.escape(stripped)}</p>")
    flush_bullets()
    # Join and unescape minimal punctuation replacements (re.escape adds backslashes)
    html = "".join(html_parts)
    html = html.replace(r"\-", "-").replace(r"\.", ".").replace(r"\,", ",")
    html = html.replace(r"\(", "(").replace(r"\)", ")").replace(r"\/", "/")
    return html


class ResumeOptimizer:
    """Coordinates remote AI call with multi-provider + local fallbacks.

    Order:
      1. OpenRouter model cascade (if configured)
      2. Gemini model cascade (if configured & OpenRouter failed)
      3. Heuristic fallback
    """

    def __init__(self,
                 openrouter_client: Optional[OpenRouterClient],
                 gemini_client: Optional[GeminiClient]):
        self.openrouter_client = openrouter_client
        self.gemini_client = gemini_client
        self.prompt_builder = ResumePromptBuilder()
        self.parser = AIResponseParser()
        self.fallback = HeuristicOptimizer()

    def optimise(self,
                 resume_text: str,
                 job_description: str,
                 max_tokens: int,
                 temperature: float,
                 user_metrics: Optional[Dict[str, str]],
                 preserve_template: bool) -> Dict[str, object]:
        messages = self.prompt_builder.optimisation_messages(resume_text, job_description, preserve_template)
        attempts: List[ModelAttempt] = []
        parsed_payload: Optional[Dict[str, object]] = None

        # 1. OpenRouter cascade
        if self.openrouter_client:
            for attempt_index, model in enumerate(self.openrouter_client.models, start=1):
                if attempt_index > MAX_REMOTE_ATTEMPTS:
                    break
                response_text, attempt = self.openrouter_client.chat(model, messages, max_tokens, temperature)
                attempts.append(attempt)
                if attempt.status != "success" or not response_text:
                    continue
                parsed_payload = self.parser.parse(response_text)
                if parsed_payload:
                    parsed_payload["source"] = "openrouter"
                    break

        # 2. Gemini cascade (only if OR failed entirely)
        if not parsed_payload and self.gemini_client:
            for model in self.gemini_client.models:
                response_text, attempt = self.gemini_client.chat(model, messages, max_tokens, temperature)
                attempts.append(attempt)
                if attempt.status != "success" or not response_text:
                    continue
                parsed_payload = self.parser.parse(response_text)
                if parsed_payload:
                    parsed_payload["source"] = "gemini"
                    break
                # Salvage: if JSON parse failed, treat raw text as enhanced resume to inspect behavior
                if not parsed_payload and response_text:
                    if os.getenv("AI_DEBUG_LOG"):
                        logging.warning("Gemini raw (non-JSON) response used as enhanced resume; model=%s len=%d", model, len(response_text))
                    # Provide heuristic structure but substitute enhanced body with raw model text
                    heur = self.fallback.optimise(resume_text, job_description, user_metrics, preserve_template)
                    # Basic anti-copy filtering: drop lines that appear verbatim in JD beyond a threshold
                    jd_lower = (job_description or "").lower()
                    filtered_lines: List[str] = []
                    for ln in response_text.splitlines():
                        clean = ln.strip()
                        if not clean:
                            continue
                        # If line appears largely in JD, skip
                        if clean.lower() in jd_lower and len(clean) > 40:
                            continue
                        filtered_lines.append(clean)
                    enhanced_raw = "\n".join(filtered_lines) or response_text
                    heur["enhanced_resume"] = sanitize_resume_text(enhanced_raw)
                    heur["source"] = "gemini_raw"
                    parsed_payload = heur
                    break

        if parsed_payload:
            enhanced = apply_metric_overrides(parsed_payload.get("enhanced_resume", ""), user_metrics)
            parsed_payload["enhanced_resume"] = enhanced
            if os.getenv("GENERATE_HTML", "0") not in ("0", "false", "False", ""):
                try:
                    parsed_payload["html_resume"] = format_resume_as_html(enhanced)
                except Exception as html_exc:  # pragma: no cover
                    logging.warning("HTML formatting failed: %s", html_exc)
            parsed_payload.setdefault("model_attempts", [a.as_dict() for a in attempts])
            return parsed_payload

        # 3. Local heuristic fallback
        fallback_result = self.fallback.optimise(resume_text, job_description, user_metrics, preserve_template)
        fallback_result["model_attempts"] = [a.as_dict() for a in attempts]
        if os.getenv("GENERATE_HTML", "0") not in ("0", "false", "False", ""):
            try:
                fallback_result["html_resume"] = format_resume_as_html(fallback_result.get("enhanced_resume", ""))
            except Exception as html_exc:  # pragma: no cover
                logging.warning("HTML formatting failed: %s", html_exc)
        return fallback_result


## (Removed malformed duplicate ResumeOptimizer definition; using corrected definition above)


def call_gemini_optimize_resume(
    resume_text: str,
    job_description: str,
    max_tokens: int = 3072,
    temperature: float = 0.35,
    user_metrics: Optional[Dict[str, str]] = None,
    preserve_template: bool = False,
) -> Dict[str, object]:
    """Public API consumed by the FastAPI routes.

    Returns a dict containing enhanced resume text, ATS insights, keyword analysis, structured feedback,
    and optionally interview questions. Falls back to deterministic heuristics when remote AI is unavailable.
    """
    openrouter_client = OpenRouterClient(OPENROUTER_API_KEY, DEFAULT_MODEL_SEQUENCE) if DEFAULT_MODEL_SEQUENCE else None
    gemini_models = get_gemini_models() if callable(get_gemini_models) else []
    gemini_client = GeminiClient(GEMINI_API_KEY, gemini_models) if gemini_models else None
    optimizer = ResumeOptimizer(openrouter_client, gemini_client)
    try:
        result = optimizer.optimise(
            resume_text=resume_text,
            job_description=job_description,
            max_tokens=max_tokens,
            temperature=temperature,
            user_metrics=user_metrics,
            preserve_template=preserve_template,
        )
        if not result.get("enhanced_resume"):
            raise RuntimeError("Unable to generate enhanced resume")
        return result
    except Exception as e:  # pragma: no cover - defensive runtime path
        logging.error(f"AI optimization failed: {e}")
        # Graceful fallback to heuristic optimizer
        return optimizer.fallback.optimise(resume_text, job_description, user_metrics, preserve_template)


def call_gemini_raw(prompt: str, max_tokens: int = 1024, temperature: float = 0.4) -> str:
    """Lightweight wrapper that mirrors the previous Gemini helper."""
    openrouter_client = OpenRouterClient(OPENROUTER_API_KEY, DEFAULT_MODEL_SEQUENCE)
    gemini_models = get_gemini_models() if callable(get_gemini_models) else []
    gemini_client = GeminiClient(GEMINI_API_KEY, gemini_models) if gemini_models else None
    messages = [
        {"role": "system", "content": "Respond succinctly and helpfully."},
        {"role": "user", "content": prompt},
    ]
    attempts: List[ModelAttempt] = []
    # Try OpenRouter first
    for model in openrouter_client.models:
        response_text, attempt = openrouter_client.chat(model, messages, max_tokens, temperature)
        attempts.append(attempt)
        if response_text and attempt.status == "success":
            return response_text
    # Then Gemini
    if gemini_client:
        for model in gemini_client.models:
            response_text, attempt = gemini_client.chat(model, messages, max_tokens, temperature)
            attempts.append(attempt)
            if response_text and attempt.status == "success":
                return response_text
    logging.error("All provider attempts failed; attempts: %s", ", ".join(f"{a.model}:{a.status}" for a in attempts))
    return "Error: Unable to generate response at this time."