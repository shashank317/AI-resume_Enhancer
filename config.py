# config.py
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

load_dotenv()

# Read GEMINI API key from environment but do not raise at import-time.
# Render (or other hosts) should provide this as a secret environment variable.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY is not set. Gemini API calls will fail until the secret is configured in the environment.")

# Upload directory (can be overridden via env). Keep creating the directory if running locally.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

# Ensure upload dir exists locally. In production you should mount a persistent disk or use S3.
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    logging.debug(f"Could not create upload dir {UPLOAD_DIR}, it may be on a read-only filesystem in production.")
