# config.py
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables. Please ensure it is set.")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

# Ensure upload dir exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
