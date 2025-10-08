# config.py
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

load_dotenv()

# ===== API Keys =====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logging.warning("⚠️ OPENROUTER_API_KEY not set — OpenRouter calls may fail.")

# Separate Gemini API key (Google AI / Gemini models)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.info("ℹ️ GEMINI_API_KEY not set — Gemini fallback will be skipped if OpenRouter fails.")

# ===== Upload Directory =====
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    logging.debug(f"Could not create upload dir {UPLOAD_DIR} — may be read-only in production.")

# ===== Model Fallback Cascade =====
# Ordered list — system will try these models in sequence if one fails.
MODEL_PRIORITY = [
    "deepseek/deepseek-chat",
    "openai/gpt-oss-20b:free",
    # Newly added free-tier cascade (requested)
    "deepseek/deepseek-chat-v3.1:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "z-ai/glm-4.5-air:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1:free",
    "microsoft/mai-ds-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-4-maverick:free",
]

# Default to the first model in the list
MODEL_NAME = os.getenv("MODEL_NAME", MODEL_PRIORITY[0])

# Gemini model fallback list (first item can be overridden via GEMINI_MODEL env var)
GEMINI_MODEL_PRIORITY = [
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

def get_available_model():
    """Returns the best available model (can be extended for health-check logic)."""
    for model in MODEL_PRIORITY:
        if model:  # You can add API call test here if needed
            return model
    return MODEL_PRIORITY[0]

def get_gemini_models() -> list[str]:
    return [m for m in GEMINI_MODEL_PRIORITY if m]
