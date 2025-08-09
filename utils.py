# utils.py
from pathlib import Path
import uuid
import shutil

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def unique_file_path(upload_dir: Path, original_name: str) -> Path:
    safe_name = original_name.replace(" ", "_")
    return upload_dir / f"{uuid.uuid4().hex}_{safe_name}"

def save_upload_file(upload_file, dest_path: Path):
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
