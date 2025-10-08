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
    # Root (landing page) route
    @app.get("/")
    async def serve_landing():
        landing_path = os.path.join(static_files_dir, "landing.html")
        if os.path.isfile(landing_path):
            return FileResponse(landing_path)
        # Fallback to app if landing missing
        return FileResponse(os.path.join(static_files_dir, "index.html"))

    # Dedicated app route for the interactive optimizer UI
    @app.get("/app")
    async def serve_app():
        index_path = os.path.join(static_files_dir, "index.html")
        if not os.path.isfile(index_path):
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_path)

    # Backwards-compatible route: serve files requested under /frontend/* paths.
    @app.get('/frontend/{path:path}')
    async def serve_frontend_path(path: str):
        requested = os.path.join(static_files_dir, path)
        if os.path.isfile(requested):
            return FileResponse(requested)
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    # Mount static assets at /static to avoid overshadowing explicit routes
    app.mount("/static", StaticFiles(directory=static_files_dir, html=True), name="static")
else:
    logging.warning(f"Static files directory '{static_files_dir}' not found. Frontend will not be served.")
