# services/file_parser.py
from pathlib import Path
import fitz  # PyMuPDF
import docx
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    try:
        with fitz.open(path) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text("text")
                if text:
                    text_parts.append(text)
                    logger.info(f"Extracted text from page {page_num + 1} of {path.name}")
                else:
                    logger.warning(f"No text extracted from page {page_num + 1} of {path.name}")
        full_text = "\n".join(text_parts).strip()
        if not full_text:
            logger.warning(f"No text extracted from PDF: {path.name}. It might be an image-based PDF.")
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {path.name}: {e}")
        raise

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
