from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from uuid import uuid4
from pathlib import Path
from services.file_parser import extract_resume_text
from config import UPLOAD_DIR

from services.gemini_service import call_gemini_optimize_resume
import aiofiles
import shutil
import os

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


async def save_upload_file(upload_file: UploadFile, destination: Path):
    """Save uploaded file asynchronously."""
    try:
        async with aiofiles.open(destination, "wb") as out_file:
            content = await upload_file.read()
            await out_file.write(content)
    finally:
        await upload_file.close()


@router.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume file and store it."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")

    resume_id = str(uuid4())
    filename = f"{resume_id}{ext}"
    file_path = UPLOAD_DIR / filename

    await save_upload_file(file, file_path)

    return {
        "message": "Resume uploaded successfully",
        "resume_id": resume_id,
        "filename": filename
    }


@router.post("/enhance_resume")
async def enhance_resume(
    file: UploadFile = File(None),
    job_description: str = Query(..., description="Job description text"),
    resume_id: str = Query(None, description="Resume ID from upload")
):
    """
    Enhance a resume using GPT.  
    Either provide a file OR a resume_id from a previous upload.
    """
    # Determine where to get resume text
    if file:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file type")
        temp_path = UPLOAD_DIR / f"{uuid4()}{ext}"
        await save_upload_file(file, temp_path)
        resume_text = extract_resume_text(temp_path)
    elif resume_id:
        # Use previously uploaded file
        matches = list(UPLOAD_DIR.glob(f"{resume_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail="Resume not found")
        resume_text = extract_resume_text(matches[0])
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or resume_id")

    # Call GPT to enhance
    try:
        enhanced_text = call_gemini_optimize_resume(resume_text, job_description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

    # Save enhanced resume
    enhanced_filename = f"enhanced_{uuid4()}.txt"
    enhanced_path = UPLOAD_DIR / enhanced_filename
    async with aiofiles.open(enhanced_path, "w", encoding="utf-8") as f:
        await f.write(enhanced_text)

    return {
        "message": enhanced_text,
        "enhanced_filename": enhanced_filename
    }


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Securely download a file from the uploads directory."""
    safe_filename = Path(filename).name  # Prevent path traversal
    file_path = UPLOAD_DIR / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=safe_filename)