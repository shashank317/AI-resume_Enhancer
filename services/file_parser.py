# services/file_parser.py
from pathlib import Path
import fitz  # PyMuPDF
import docx

def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    with fitz.open(path) as pdf:
        for page in pdf:
            text = page.get_text("text")
            if text:
                text_parts.append(text)
    return "\n".join(text_parts).strip()

def extract_text_from_docx(path: Path) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs).strip()

def extract_resume_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported resume file format: {ext}")
