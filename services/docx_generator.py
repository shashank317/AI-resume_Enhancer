# services/docx_generator.py
from docx import Document
from pathlib import Path

def create_docx_from_text(text: str, out_path: Path):
    doc = Document()
    # Split into paragraphs by two newlines preserving reasonable formatting
    for block in text.split("\n\n"):
        for line in block.splitlines():
            doc.add_paragraph(line)
        # add an empty paragraph between blocks
        doc.add_paragraph()
    doc.save(out_path)
