from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Body
from fastapi.responses import StreamingResponse, PlainTextResponse
from services.file_parser import extract_resume_text
import json
from config import UPLOAD_DIR
from services.gemini_service import call_gemini_optimize_resume
from services.gemini_service import call_gemini_raw
from services.schemas import EnhancedResumeResponse
from pydantic import ValidationError
from services.docx_generator import create_docx_from_text
from pathlib import Path
from uuid import uuid4
import aiofiles
import io
import re
from collections import Counter
from services.ats_matching import extract_keywords, ats_match
from services.jd_extractor import extract_jd_core_info

router = APIRouter()

# Simple in-memory cache for analyze endpoint (fast mode). Not persistent and resets on process restart.
_ANALYZE_CACHE: dict[str, dict] = {}


@router.post('/api/analyze')
async def api_analyze(
    file: UploadFile = File(...),
    job_description: str = Query(..., description="Job description text"),
    mode: str = Query("fast", description="Analysis mode: fast (heuristics only) | ai (LLM augmented)"),
):
    """
    Analyze resume vs job description.

    Modes:
      - fast: heuristic ATS keyword extraction & matching only (no LLM call) -> returns quickly.
      - ai: attempts LLM analysis (slower) with fallback to heuristics if model fails.
    Adds lightweight in‑memory caching to avoid recomputation for identical inputs.
    """
    mode = (mode or "fast").lower()
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed types are {list(ALLOWED_EXTENSIONS)}")

    temp_path = UPLOAD_DIR / f"temp_{uuid4()}{ext}"
    await save_upload_file(file, temp_path)
    try:
        resume_text = extract_resume_text(temp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

        # --- In-memory LRU cache (very small & simple) ---
        cache_key = None
        try:
            from hashlib import sha256
            cache_key = sha256(f"analyze::{mode}::{resume_text[:5000]}||{job_description[:5000]}".encode('utf-8', 'ignore')).hexdigest()
        except Exception:
            pass

        result = None
        if cache_key and cache_key in _ANALYZE_CACHE:
            result = _ANALYZE_CACHE[cache_key]
        if result:
            result['cached'] = True
            return result

        def heuristic_analysis():
            """Fast heuristic analysis (no network)."""
            jd_keywords = extract_keywords(job_description, max_keywords=80)
            score_pct, matched_kw, _missing = ats_match(resume_text, jd_keywords, threshold=75)
            stop = {
                'the','a','an','to','of','and','or','in','on','for','with','by','at','from','as','is','are','was','were','be','been','being',
                'this','that','these','those','it','its','you','your','i','we','they','our','their','he','she','him','her','his','hers','them',
                'but','if','then','else','than','so','such','not','no','yes','do','does','did','done','can','could','should','would','may','might',
                'will','shall','have','has','had','having','into','about','across','over','under','per','vs','via','etc','eg','ie'
            }
            def tokenize(s: str):
                tokens = re.split(r"[^a-z0-9+.#]+", s.lower())
                return [t for t in tokens if t and t not in stop]
            cv_toks = tokenize(resume_text)
            cv_counts = Counter(cv_toks)
            matched_norm = {m.lower() for m in matched_kw}
            unwanted_local = [tok for tok, _ in cv_counts.most_common(120) if tok not in matched_norm][:25]
            return {
                'unwanted_sections': unwanted_local,
                'matched_sections': matched_kw[:50],
                'extracted_text': resume_text,
                'ats_score': score_pct,
                'ai_used': False,
                'cached': False,
                'mode': 'fast'
            }

        if mode != 'ai':
            result = heuristic_analysis()
        else:
            analyze_prompt = (
                "You are an expert recruiter. Analyze the resume vs job description. Return JSON with keys: "
                "unwanted_sections (phrases in resume irrelevant to JD), matched_sections (phrases/skills from resume strongly aligned). "
                "Keep lists short (<=30 items). Return ONLY JSON.\n\nResume:\n" + resume_text + "\n\nJob Description:\n" + job_description
            )
            try:
                raw = call_gemini_raw(analyze_prompt, max_tokens=512, temperature=0.0)
                try:
                    parsed = json.loads(raw)
                except Exception:
                    start = raw.find('{')
                    end = raw.rfind('}')
                    if start != -1 and end != -1:
                        parsed = json.loads(raw[start:end+1])
                    else:
                        raise ValueError('invalid json from ai')
                unwanted_ai = parsed.get('unwanted_sections') or parsed.get('unwanted') or []
                matched_ai = parsed.get('matched_sections') or parsed.get('matched') or []
                # compute ats score heuristically for consistency
                jd_keywords = extract_keywords(job_description, max_keywords=80)
                score_pct, _matched_tmp, _missing_tmp = ats_match(resume_text, jd_keywords, threshold=75)
                result = {
                    'unwanted_sections': unwanted_ai[:30],
                    'matched_sections': matched_ai[:50],
                    'extracted_text': resume_text,
                    'ats_score': score_pct,
                    'ai_used': True,
                    'cached': False,
                    'mode': 'ai'
                }
            except Exception:
                # Fallback to heuristic
                result = heuristic_analysis()
                result['ai_fallback_used'] = True

        if cache_key:
            _ANALYZE_CACHE[cache_key] = result
            # trim LRU if too big
            if len(_ANALYZE_CACHE) > 100:
                # pop first inserted (Python 3.7+ dict preserves insertion order)
                _ANALYZE_CACHE.pop(next(iter(_ANALYZE_CACHE)))
        return result
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.post('/api/generate')
async def api_generate(
    file: UploadFile = File(...),
    job_description: str = Query(..., description="Job description text"),
    matched_sections: str = Query(None, description="Optional matched sections JSON"),
    user_metrics: str = Query(None, description="Optional JSON object of metric substitutions keyed by keyword (e.g., {\"FastAPI\": \"30%\"})"),
    preserve_template: bool = Query(False, description="If true, keep original template structure and only minimally tailor text"),
    output_format: str = Query("json", description="Response format: json | text | docx"),
    plain: bool = Query(False, description="Force returning only enhanced resume plain text regardless of output_format")
):
    """
    Generate a tailored resume emphasizing matched_sections and removing unwanted content. Returns enhanced_resume and matched_sections.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed types are {list(ALLOWED_EXTENSIONS)}")

    temp_path = UPLOAD_DIR / f"temp_{uuid4()}{ext}"
    await save_upload_file(file, temp_path)
    try:
        resume_text = extract_resume_text(temp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

        # matched_sections may be a JSON string; parse if provided
        matched = []
        if matched_sections:
            try:
                matched = json.loads(matched_sections)
            except Exception:
                matched = [s.strip() for s in matched_sections.split(',') if s.strip()]

        preserve_instruction = (
            "\n\nIMPORTANT: Preserve the original resume template and formatting. "
            "When returning the enhanced resume, keep the same headings, bullet style, ordering, and spacing as the provided resume. "
            "Do not change the visual template — only rewrite content to emphasize matched skills and remove irrelevant sections."
        )
        job_desc_with_preserve = job_description + preserve_instruction

        parsed_metrics = None
        if user_metrics:
            try:
                parsed_metrics = json.loads(user_metrics)
                if not isinstance(parsed_metrics, dict):
                    parsed_metrics = None
            except Exception:
                parsed_metrics = None

        raw = call_gemini_optimize_resume(
            resume_text,
            job_desc_with_preserve,
            user_metrics=parsed_metrics,
            preserve_template=preserve_template
        )

        parsed = raw if isinstance(raw, dict) else None
        if parsed is None:
            try:
                parsed = json.loads(raw)
            except Exception:
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1:
                    parsed = json.loads(raw[start:end+1])
                else:
                    raise HTTPException(status_code=502, detail="AI generation returned invalid JSON")

        enhanced = parsed.get('enhanced_resume') or parsed.get('resume') or ''
        matched_out = parsed.get('matched_sections') or parsed.get('matched') or matched
        ats = parsed.get('ats_breakdown') or {}
        feedback = parsed.get('feedback') or {}
        breakdown_values = list(ats.values())
        total_score = int(round(sum(breakdown_values) / len(breakdown_values))) if breakdown_values else 0
        # Output formatting options
        fmt = (output_format or "json").lower()
        if plain or fmt == "text":
            # Plain text only (newbie friendly)
            if not enhanced:
                raise HTTPException(status_code=502, detail="Empty enhanced resume output")
            return PlainTextResponse(enhanced)
        if fmt == "docx":
            if not enhanced:
                raise HTTPException(status_code=502, detail="Empty enhanced resume output")
            docx_buffer = create_docx_from_text(enhanced)
            return StreamingResponse(
                docx_buffer,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": "attachment; filename=enhanced_resume.docx"}
            )
        return {
            'enhanced_resume': enhanced,
            'matched_sections': matched_out,
            'ats_breakdown': ats,
            'feedback': feedback,
            'total_score': total_score
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.post('/api/generate_text', response_class=PlainTextResponse)
async def api_generate_text(
    file: UploadFile = File(...),
    job_description: str = Query(..., description="Job description text"),
    matched_sections: str = Query(None, description="Optional matched sections JSON"),
    user_metrics: str = Query(None, description="Optional JSON object of metric substitutions keyed by keyword (e.g., {\"FastAPI\": \"30%\"})"),
    preserve_template: bool = Query(False, description="If true, keep original template structure and only minimally tailor text")
):
    """Text-only generation endpoint. Always returns text/plain enhanced resume.

    This avoids any ambiguity with content negotiation on certain hosts.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed types are {list(ALLOWED_EXTENSIONS)}")

    temp_path = UPLOAD_DIR / f"temp_{uuid4()}{ext}"
    await save_upload_file(file, temp_path)
    try:
        resume_text = extract_resume_text(temp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

        matched = []
        if matched_sections:
            try:
                matched = json.loads(matched_sections)
            except Exception:
                matched = [s.strip() for s in matched_sections.split(',') if s.strip()]

        preserve_instruction = (
            "\n\nIMPORTANT: Preserve the original resume template and formatting. "
            "When returning the enhanced resume, keep the same headings, bullet style, ordering, and spacing as the provided resume. "
            "Do not change the visual template — only rewrite content to emphasize matched skills and remove irrelevant sections."
        )
        job_desc_with_preserve = job_description + preserve_instruction

        parsed_metrics = None
        if user_metrics:
            try:
                parsed_metrics = json.loads(user_metrics)
                if not isinstance(parsed_metrics, dict):
                    parsed_metrics = None
            except Exception:
                parsed_metrics = None

        raw = call_gemini_optimize_resume(
            resume_text,
            job_desc_with_preserve,
            user_metrics=parsed_metrics,
            preserve_template=preserve_template
        )

        parsed = raw if isinstance(raw, dict) else None
        if parsed is None:
            try:
                parsed = json.loads(raw)
            except Exception:
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1:
                    parsed = json.loads(raw[start:end+1])
                else:
                    raise HTTPException(status_code=502, detail="AI generation returned invalid JSON")

        enhanced = parsed.get('enhanced_resume') or parsed.get('resume') or ''
        if not enhanced:
            raise HTTPException(status_code=502, detail="Empty enhanced resume output")

        # Response class on decorator ensures text/plain
        try:
            import logging as _logging
            _logging.getLogger(__name__).info("Returning enhanced resume as text/plain (%d chars)", len(enhanced))
        except Exception:
            pass
        return enhanced
    finally:
        if temp_path.exists():
            temp_path.unlink()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

async def save_upload_file(upload_file: UploadFile, destination: Path):
    """Save uploaded file asynchronously."""
    try:
        async with aiofiles.open(destination, "wb") as out_file:
            while content := await upload_file.read(1024):
                await out_file.write(content)
    finally:
        await upload_file.close()

@router.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume file and return a resume_id for later enhancement."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed types are {list(ALLOWED_EXTENSIONS)}")
    resume_id = f"{uuid4()}{ext}"
    dest_path = UPLOAD_DIR / resume_id
    await save_upload_file(file, dest_path)
    return {"resume_id": resume_id, "filename": file.filename}


@router.post("/enhance_resume")
async def enhance_resume(
    job_description: str = Query(..., description="Job description text"),
    file: UploadFile | None = File(None),
    resume_id: str | None = Query(None, description="Previously uploaded resume_id returned from /upload_resume"),
    user_metrics: str = Query(None, description="Optional JSON object of metric substitutions keyed by keyword (e.g., {\"FastAPI\": \"30%\"})"),
    preserve_template: bool = Query(False, description="If true, keep original template structure and only minimally tailor text")
):
    """Enhance an uploaded resume (via file or resume_id). Returns summary message and enhanced file reference.

    Maintains backward compatibility with earlier tests expecting message & enhanced_filename.
    """
    if not file and not resume_id:
        raise HTTPException(status_code=400, detail="Provide either a file or resume_id")

    temp_path: Path | None = None
    cleanup_paths: list[Path] = []
    try:
        if file:
            ext = Path(file.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed types are {list(ALLOWED_EXTENSIONS)}")
            temp_path = UPLOAD_DIR / f"temp_{uuid4()}{ext}"
            await save_upload_file(file, temp_path)
            source_path = temp_path
        else:
            source_path = UPLOAD_DIR / resume_id  # resume_id includes extension
            if not source_path.exists():
                raise HTTPException(status_code=404, detail="Resume file not found for provided resume_id")

        resume_text = extract_resume_text(source_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the resume. The file might be empty or corrupted.")

        parsed_metrics = None
        if user_metrics:
            try:
                parsed_metrics = json.loads(user_metrics)
                if not isinstance(parsed_metrics, dict):
                    parsed_metrics = None
            except Exception:
                parsed_metrics = None

        try:
            ai_response = call_gemini_optimize_resume(
                resume_text,
                job_description,
                user_metrics=parsed_metrics,
                preserve_template=preserve_template
            )
        except RuntimeError as re_err:
            raise HTTPException(status_code=502, detail=str(re_err))

        # Validate structure (best effort)
        enhanced_resume = ai_response.get("enhanced_resume") if isinstance(ai_response, dict) else ""
        if not isinstance(enhanced_resume, str):
            enhanced_resume = ""
        if not enhanced_resume:
            # No offline fallback anymore; treat empty as error
            raise HTTPException(status_code=502, detail="Model returned empty enhanced resume.")

        enhanced_filename = f"enhanced_{uuid4()}.txt"
        enhanced_path = UPLOAD_DIR / enhanced_filename
        enhanced_path.write_text(enhanced_resume, encoding="utf-8")

        # Provide legacy-style response fields plus enriched data
        response_payload = {
            "message": "Resume enhanced successfully.",
            "enhanced_filename": enhanced_filename,
            "enhanced_resume": enhanced_resume,
        }
        # Merge additional structured info if present
        for k in ("ats_breakdown", "matched_keywords", "missing_keywords", "feedback", "model_attempts"):
            if isinstance(ai_response, dict) and k in ai_response:
                response_payload[k] = ai_response[k]
        return response_payload
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@router.get("/download/{filename}")
async def download(filename: str):
    """Download an enhanced (or original) text file."""
    target = UPLOAD_DIR / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
    return PlainTextResponse(content)

@router.post("/generate_docx")
async def generate_docx(
    resume_text: str = Body(..., embed=True, description="The tailored resume text to be converted to DOCX.")
):
    """
    Generates a DOCX file from the provided text and returns it for download.
    """
    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")

@router.post('/ats/score_text')
async def ats_score_text(
    resume_text: str = Body(..., embed=True, description="Resume text to score"),
    job_description: str = Query(..., description="Job description text")
):
    """Return ATS score, matched and missing keywords for an arbitrary resume text.

    Used after AI generation to show updated score without re-uploading file.
    """
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty")
    jd_keywords = extract_keywords(job_description, max_keywords=80)
    score_pct, matched_kw, missing_kw = ats_match(resume_text, jd_keywords, threshold=75)
    return {
        'ats_score': score_pct,
        'matched_keywords': matched_kw,
        'missing_keywords': missing_kw,
        'keyword_count': len(jd_keywords)
    }

    try:
        doc_buffer = create_docx_from_text(resume_text)
        
        headers = {
            'Content-Disposition': 'attachment; filename="tailored_resume.docx"'
        }
        
        return StreamingResponse(doc_buffer, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate DOCX file: {str(e)}")


@router.post('/generate')
async def generate_from_prompt(prompt: str = Body(..., embed=True)):
    """
    Generic endpoint to generate text from a user-provided prompt using Gemini.
    This keeps the API key off the client and centralizes calls on the server.
    """
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    try:
        result_text = call_gemini_raw(prompt)
        return {"result": result_text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {str(e)}")


@router.post('/jd/extract')
async def jd_extract(job_description: str = Body(..., embed=True)):
    """Extract core info from a job description (role_title, responsibilities, skills/tools, experience level, education, keywords)."""
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is required")
    core = extract_jd_core_info(job_description)
    return core


@router.post('/ats/map')
async def ats_map(file: UploadFile = File(...), job_description: str = Query(..., description="Job description text")):
    """Map resume to JD: returns JD core info, extracted resume text, ATS matched/missing lists (section-aware)."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed types are {list(ALLOWED_EXTENSIONS)}")

    temp_path = UPLOAD_DIR / f"temp_{uuid4()}{ext}"
    await save_upload_file(file, temp_path)
    try:
        resume_text = extract_resume_text(temp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

        core = extract_jd_core_info(job_description)
        jd_keywords = core.get("keywords") or extract_keywords(job_description, max_keywords=80)
        _, matched, missing = ats_match(resume_text, jd_keywords, threshold=75)
        return {
            "core": core,
            "extracted_text": resume_text,
            "matched_keywords": matched,
            "missing_keywords": missing,
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()
