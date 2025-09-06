import pytest
from fastapi.testclient import TestClient
from main import app
from pathlib import Path
import os

client = TestClient(app)

# Define the path to the test files
UPLOAD_DIR = Path("uploads")

# Create dummy files for testing
@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_test_files():
    # Create dummy files
    (UPLOAD_DIR / "test_upload.txt").write_text("This is a test upload.")
    (UPLOAD_DIR / "test_upload.pdf").write_text("This is a test upload.") # Dummy PDF
    (UPLOAD_DIR / "test_upload.docx").write_text("This is a test upload.") # Dummy DOCX

    yield

    # Clean up dummy files
    for f in UPLOAD_DIR.iterdir():
        if f.name.startswith("test_upload") or f.name.startswith("enhanced_") or f.name.startswith("temp_"):
            f.unlink()


def test_upload_resume_txt():
    with open(UPLOAD_DIR / "test_upload.txt", "rb") as f:
        response = client.post("/resume/upload_resume", files={"file": ("test_upload.txt", f, "text/plain")})
    assert response.status_code == 200
    assert "resume_id" in response.json()
    assert "filename" in response.json()


def test_upload_resume_unsupported_type():
    with open(UPLOAD_DIR / "test_upload.txt", "rb") as f:
        response = client.post("/resume/upload_resume", files={"file": ("test_upload.xyz", f, "application/octet-stream")})
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_enhance_resume_with_file():
    job_description = "Software Engineer"
    with open(UPLOAD_DIR / "test_upload.txt", "rb") as f:
        response = client.post(
            f"/resume/enhance_resume?job_description={job_description}",
            files={"file": ("test_upload.txt", f, "text/plain")}
        )
    assert response.status_code == 200
    assert "message" in response.json()
    assert "enhanced_filename" in response.json()
    assert response.json()["message"] != ""


def test_enhance_resume_with_resume_id():
    # First, upload a resume to get a resume_id
    with open(UPLOAD_DIR / "test_upload.txt", "rb") as f:
        upload_response = client.post("/resume/upload_resume", files={"file": ("test_upload.txt", f, "text/plain")})
    assert upload_response.status_code == 200
    resume_id = upload_response.json()["resume_id"]

    job_description = "Software Engineer"
    response = client.post(
        f"/resume/enhance_resume?job_description={job_description}&resume_id={resume_id}"
    )
    assert response.status_code == 200
    assert "message" in response.json()
    assert "enhanced_filename" in response.json()
    assert response.json()["message"] != ""


def test_enhance_resume_no_file_or_id():
    job_description = "Software Engineer"
    response = client.post(f"/resume/enhance_resume?job_description={job_description}")
    assert response.status_code == 400
    assert "Provide either a file or resume_id" in response.json()["detail"]


def test_download_file():
    # First, enhance a resume to get an enhanced_filename
    job_description = "Software Engineer"
    with open(UPLOAD_DIR / "test_upload.txt", "rb") as f:
        enhance_response = client.post(
            f"/resume/enhance_resume?job_description={job_description}",
            files={"file": ("test_upload.txt", f, "text/plain")}
        )
    assert enhance_response.status_code == 200
    enhanced_filename = enhance_response.json()["enhanced_filename"]

    response = client.get(f"/resume/download/{enhanced_filename}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.content.decode("utf-8") != ""


def test_download_non_existent_file():
    response = client.get("/resume/download/non_existent_file.txt")
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]
