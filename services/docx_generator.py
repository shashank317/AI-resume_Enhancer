from docx import Document
from docx.shared import Inches
import io

def create_docx_from_text(text: str):
    """
    Creates a DOCX document in memory from a string of text.

    Args:
        text: The text content to be included in the document.

    Returns:
        An in-memory buffer (io.BytesIO) containing the DOCX file.
    """
    document = Document()
    document.add_paragraph(text)
    
    # Create an in-memory buffer
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0) # Rewind the buffer to the beginning
    
    return buffer