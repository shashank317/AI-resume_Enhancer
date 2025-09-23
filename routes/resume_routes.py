from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
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

router = APIRouter()


@router.post('/api/analyze')
async def api_analyze(file: UploadFile = File(...), job_description: str = Query(..., description="Job description text")):
    """
    Analyze resume vs job description. Returns unwanted_sections and matched_sections for frontend highlighting.
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

        # Use Gemini to analyze and return unwanted/matched sections. We will reuse the optimize call but instruct the model
        analyze_prompt = (
            f"Analyze the following resume and job description. Return JSON with 'unwanted_sections' (list of phrases present in the resume but irrelevant to the job) and 'matched_sections' (list of important phrases from the resume that match or should be emphasized for the job). Respond ONLY with JSON.\n\n"
            f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}"
        )

        raw = call_gemini_raw(analyze_prompt, max_tokens=1024, temperature=0.0)

        # Parse AI response robustly
        try:
            parsed = json.loads(raw)
        except Exception:
            # If raw is not JSON, attempt to extract JSON substring
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start:end+1])
                except Exception:
                    raise HTTPException(status_code=502, detail="AI analysis returned invalid JSON")
            else:
                raise HTTPException(status_code=502, detail="AI analysis returned invalid JSON")

        # Ensure lists exist
        unwanted = parsed.get('unwanted_sections') or parsed.get('unwanted') or []
        matched = parsed.get('matched_sections') or parsed.get('matched') or []

        # Return the extracted plain-text resume so the frontend doesn't try to display raw PDF bytes
        return {'unwanted_sections': unwanted, 'matched_sections': matched, 'extracted_text': resume_text}
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.post('/api/generate')
async def api_generate(file: UploadFile = File(...), job_description: str = Query(..., description="Job description text"), matched_sections: str = Query(None, description="Optional matched sections JSON")):
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
                # if it's simple csv-like, try splitting
                matched = [s.strip() for s in matched_sections.split(',') if s.strip()]

        # Compose prompt for full tailored resume generation.
        # IMPORTANT: instruct the model to preserve the original resume template and formatting.
        preserve_instruction = (
            "\n\nIMPORTANT: Preserve the original resume template and formatting. "
            "When returning the enhanced resume, keep the same headings, bullet style, ordering, and spacing as the provided resume. "
            "Do not change the visual template — only rewrite content to emphasize matched skills and remove irrelevant sections."
        )

        job_desc_with_preserve = job_description + preserve_instruction

        # Call the optimizer with the resume text and the modified job description which requests template preservation
        raw = call_gemini_optimize_resume(resume_text, job_desc_with_preserve)

        # call_gemini_optimize_resume is expected to return parsed JSON already (dict)
        if isinstance(raw, dict):
            parsed = raw
        else:
            try:
                parsed = json.loads(raw)
            except Exception:
                # fallback
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

        # Optionally compute total_score as earlier
        breakdown_values = list(ats.values())
        total_score = int(round(sum(breakdown_values) / len(breakdown_values))) if breakdown_values else 0

        return { 'enhanced_resume': enhanced, 'matched_sections': matched_out, 'ats_breakdown': ats, 'feedback': feedback, 'total_score': total_score }
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

@router.post("/enhance_resume")
async def enhance_resume(
    file: UploadFile = File(...),
    job_description: str = Query(..., description="Job description text")
):
    """
    Enhances a resume based on a job description using the Gemini API.
    It extracts text from the uploaded resume, calls the Gemini service,
    and returns the full structured JSON response.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed types are {list(ALLOWED_EXTENSIONS)}")

    # Use a temporary path for processing
    temp_path = UPLOAD_DIR / f"temp_{uuid4()}{ext}"
    await save_upload_file(file, temp_path)
    
    try:
        resume_text = extract_resume_text(temp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the resume. The file might be empty or corrupted.")

        # Call Gemini to get the structured data
        gemini_response = call_gemini_optimize_resume(resume_text, job_description)

        # Validate and coerce the response using Pydantic so frontend gets a reliable contract
        try:
            validated = EnhancedResumeResponse.parse_obj(gemini_response)
        except ValidationError as ve:
            # Return a 502 with helpful debug info but avoid leaking secrets
            raise HTTPException(status_code=502, detail=f"Invalid AI response structure: {ve}",)

        # Compute an overall total_score (simple average of breakdown values) and include it
        breakdown_values = list(validated.ats_breakdown.values())
        total_score = int(round(sum(breakdown_values) / len(breakdown_values))) if breakdown_values else 0

        result = validated.model_dump()
        result['total_score'] = total_score
        return result

    except HTTPException as e:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise e
    except Exception as e:
        # Catch-all for other potential errors (e.g., Gemini API call failure)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        if temp_path.exists():
            temp_path.unlink()

@router.post("/generate_docx")
async def generate_docx(
    resume_text: str = Body(..., embed=True, description="The tailored resume text to be converted to DOCX.")
):
    """
    Generates a DOCX file from the provided text and returns it for download.
    """
    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")

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
