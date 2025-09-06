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

# ------------------- Health Check -------------------
@app.get("/health", tags=["Health"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "AI Resume Enhancer API is running!"}

# ------------------- Serve Frontend -------------------
# This should come after all other API routes.
# It serves the frontend application, defaulting to index.html for any path not handled by the API.
app.mount("/", StaticFiles(directory="frontend", html=True), name="static-frontend")
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static-frontend")
