import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import resume_routes
from fastapi import HTTPException

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
async def health_check():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "AI Resume Enhancer API is running!"}

# ------------------- Serve Frontend -------------------
# This should come after all other API routes.
# It serves the frontend application, defaulting to index.html for any path not
# handled by the API. We check if the directory exists to avoid crashing during
# local development if the frontend hasn't been built.
static_files_dir = "frontend"
if os.path.isdir(static_files_dir):
    # Register the root route first so requests to '/' return landing.html (if present).
    @app.get("/")
    async def serve_index():
        # Serve the landing page at the root so users see it first
        landing_path = os.path.join(static_files_dir, "landing.html")
        if os.path.isfile(landing_path):
            return FileResponse(landing_path)
        return FileResponse(os.path.join(static_files_dir, "index.html"))

    # Backwards-compatible route: serve files requested under /frontend/* paths.
    # This helps older links or bookmarks that include the 'frontend/' prefix.
    @app.get('/frontend/{path:path}')
    async def serve_frontend_path(path: str):
        requested = os.path.join(static_files_dir, path)
        if os.path.isfile(requested):
            return FileResponse(requested)
        # Not found — raise 404 so the client knows
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    # Mount static files after routes so the root route above takes precedence.
    app.mount("/", StaticFiles(directory=static_files_dir, html=True), name="static")
else:
    logging.warning(f"Static files directory '{static_files_dir}' not found. Frontend will not be served.")
