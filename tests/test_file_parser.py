import pytest
from pathlib import Path
from services.file_parser import extract_resume_text
import docx
import fitz # PyMuPDF

# Define the path to the test files 
UPLOAD_DIR = Path("uploads")

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_file_parser_test_files():
    # Create dummy files
    (UPLOAD_DIR / "test_resume.txt").write_text("This is a test resume in a text file.")
    
    doc = docx.Document()
    doc.add_paragraph("This is a test resume in a docx file.")
    doc.save(UPLOAD_DIR / "test_resume.docx")

    pdf_doc = fitz.open()
    pdf_page = pdf_doc.new_page()
    pdf_page.insert_text((72, 72), "This is a test resume in a PDF file.")
    pdf_doc.save(UPLOAD_DIR / "test_resume.pdf")
    pdf_doc.close()

    yield

    # Clean up dummy files
    (UPLOAD_DIR / "test_resume.txt").unlink(missing_ok=True)
    (UPLOAD_DIR / "test_resume.docx").unlink(missing_ok=True)
    (UPLOAD_DIR / "test_resume.pdf").unlink(missing_ok=True)


def test_extract_text_from_txt():
    # Test extracting text from a .txt file
    txt_path = UPLOAD_DIR / "test_resume.txt"
    assert extract_resume_text(txt_path) == "This is a test resume in a text file."


def test_extract_text_from_docx():
    # Test extracting text from a .docx file
    docx_path = UPLOAD_DIR / "test_resume.docx"
    assert extract_resume_text(docx_path) == "This is a test resume in a docx file."


def test_extract_text_from_pdf():
    # Test extracting text from a .pdf file
    pdf_path = UPLOAD_DIR / "test_resume.pdf"
    assert extract_resume_text(pdf_path) == "This is a test resume in a PDF file."


def test_unsupported_file_type():
    # Test that an unsupported file type raises a ValueError
    unsupported_path = UPLOAD_DIR / "unsupported.zip"
    with open(unsupported_path, "w") as f:
        f.write("dummy content")
    with pytest.raises(ValueError):
        extract_resume_text(unsupported_path)
    unsupported_path.unlink() # clean up the dummy file
