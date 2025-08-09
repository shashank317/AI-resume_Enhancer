import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import resume_routes

# ------------------- Logging Setup -------------------
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

setup_logging()

# ------------------- FastAPI App -------------------
app = FastAPI(title="AI Resume Enhancer")

# ------------------- CORS -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- API Routes -------------------
app.include_router(resume_routes.router, prefix="/resume", tags=["resume"])

# ------------------- Serve Frontend -------------------
# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))
