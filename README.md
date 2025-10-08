# AI Resume Enhancer (FastAPI + HTML/JS)

An AI‑powered resume enhancer that tailors a candidate’s resume to a Job Description (JD), computes an ATS match score with semantic + fuzzy matching, highlights matched / irrelevant parts, and outputs a clean plain‑text or structured optimized version. Backend: FastAPI + multi‑model OpenRouter + optional Gemini fallback + heuristic safety net. Frontend: static HTML/JS (Tailwind CDN) served by FastAPI (`/` landing, `/app` main UI).

## ✨ Key Features

- Resume upload (PDF / DOCX / TXT; PDF text extraction via PyMuPDF)
- Pre‑Generation Analysis:
  - Irrelevant / removable fragments highlighted (red)
  - Matched resume phrases harvested for optimization
- AI Generation (Tailored Resume):
  - Multi‑model cascade (OpenRouter models → Gemini → heuristic fallback)
  - Optional template preservation (`preserve_template=true`)
  - Plain Text Only mode (`plain=1`) or full ATS JSON
- ATS Intelligence:
  - Lemma + synonym + fuzzy + family grouping (cloud/sql/etc.)
  - Breakdown + strengths + suggestions + missing keywords
  - Tuned fuzzy threshold (65) for better recall
- Plain Text Mode: returns strict `text/plain` (no JSON envelope)
- Utilities: Copy, PDF export (client-side), DOCX (server), history panel
- Dark/Light theme toggle; graceful offline / heuristic fallbacks
- Health check endpoint: `/health`

## 🚀 Getting Started (Local)

### Prerequisites

- Python 3.10+ (3.11 recommended)
- pip
- OpenRouter API key (for remote models) — optional but recommended

### 1) Clone & Enter Project

```bash
cd C:\Users\sumit\Desktop\ai-resume
```

### 2) Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # (Windows)
```

### 3) Install Dependencies

```bash
pip install -r requirements.txt
```

### 4) Environment Variables

Option A – ad‑hoc (current terminal):

```bash
set OPENROUTER_API_KEY=your_openrouter_key_here
set GEMINI_API_KEY=your_gemini_key_here        # optional
set MODEL_NAME=deepseek/deepseek-chat
set GEMINI_MODEL=gemini-2.5-flash              # optional
set UPLOAD_DIR=uploads
```

Option B – `.env` file (auto‑loaded by `python-dotenv`):

```
OPENROUTER_API_KEY=your_openrouter_key_here
GEMINI_API_KEY=your_gemini_key_here
MODEL_NAME=deepseek/deepseek-chat
GEMINI_MODEL=gemini-2.5-flash
UPLOAD_DIR=uploads
```

### 5) Run the App

```bash
uvicorn main:app --reload
```

Open: http://127.0.0.1:8000/ (landing) then http://127.0.0.1:8000/app (app interface).

Optional (improved ATS lemma quality):

```bash
python -m spacy download en_core_web_sm
```

## 🧪 Quick Local Restart (No Docker)

If you change dependencies:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## 📒 Usage Guide

1. Visit landing (`/`), then open app (`/app`).
2. Upload resume & paste JD.
3. Click “Initiate ATS Analysis” → red highlights (unwanted) + collects matched phrases.
4. Click “AI Generate Optimized Version” → tailored resume.
5. Copy or export as PDF/DOCX; switch plain vs ATS mode by editing frontend constant for now.

### Plain Text vs Detailed ATS Mode

Frontend constant in `frontend/index.html`:

```js
const FORCE_PLAIN_TEXT_OUTPUT = true;
```

Set to `false` for full JSON (ATS panels). Backend options:

| Parameter | Effect |
|-----------|--------|
| `plain=1` | Forces raw `text/plain` body (ignores `output_format`) |
| `output_format=json` | JSON envelope (enhanced_resume + ats_breakdown etc.) |
| `output_format=text` | (Legacy) plain text (prefer `plain=1`) |
| `output_format=docx` | Returns DOCX stream |

### API Examples (curl)

Health:

```bash
curl http://127.0.0.1:8000/health
```

Analyze (JSON response):

```bash
curl -F "file=@C:\\path\\to\\resume.pdf" \
  "http://127.0.0.1:8000/resume/api/analyze?job_description=Your%20JD%20text"
```

Generate (plain text only):

```bash
curl -F "file=@C:\\path\\to\\resume.pdf" \
  "http://127.0.0.1:8000/resume/api/generate?job_description=Your%20JD%20text&plain=1"
```

Generate (structured JSON):

```bash
curl -F "file=@C:\\path\\to\\resume.pdf" \
  "http://127.0.0.1:8000/resume/api/generate?job_description=Your%20JD%20text&output_format=json"
```

Legacy enhance:

```bash
curl -F "file=@C:\\path\\to\\resume.pdf" \
  "http://127.0.0.1:8000/resume/enhance_resume?job_description=Your%20JD%20text"
```

Notes:

- Image‑only PDFs need OCR (not included).
- Keep secrets in env vars / `.env` (never commit).
- History persists in browser `localStorage` only.

## 🗂️ Project Structure

```
.
├─ config.py                # Env + model priority config
├─ main.py                  # FastAPI app, CORS, /health, landing '/', app '/app'
├─ routes/
│  └─ resume_routes.py      # Core resume/ATS endpoints
├─ services/
│  ├─ gemini_service.py     # Model cascade + prompt builder + heuristic fallback
│  ├─ ats_matching.py       # Keyword/lemma/fuzzy ATS logic
│  ├─ file_parser.py        # PDF/DOCX/TXT extraction
│  ├─ docx_generator.py     # DOCX export
│  └─ schemas.py            # Pydantic response models
├─ frontend/
│  ├─ landing.html          # Landing page (marketing)
│  └─ index.html            # App UI (analysis + generation)
├─ uploads/                 # Uploaded + generated artifacts
├─ logs/app.log             # Runtime log file
├─ tests/                   # Pytest suite
├─ start.sh                 # Simple startup script (non-Docker)
├─ requirements.txt         # Dependencies
└─ README.md
```

## 🧰 Technologies & Dependencies

- Backend: FastAPI, Uvicorn
- AI: OpenRouter (multi‑provider) + optional Gemini + heuristic fallback
- NLP / Matching: spaCy (optional model), rapidfuzz, custom normalization families
- Parsing: PyMuPDF, python-docx, python-multipart
- Frontend: Vanilla JS, Tailwind CDN, jsPDF, html2canvas
- Testing: pytest, httpx
- Infra: Plain Uvicorn (no Docker), dotenv

## 🚀 Deployment (Render Recommended)

This repo now includes `Dockerfile` and `render.yaml` for one‑click Render deployment.

### Render Steps
1. Push repo to GitHub.
2. In Render dashboard: New → Blueprint → point to this repo (uses `render.yaml`).
3. Add secrets in the service settings (Environment):
  - `OPENROUTER_API_KEY=...`
  - `GEMINI_API_KEY=...` (optional)
  - `MODEL_NAME=deepseek/deepseek-chat` (optional override)
4. Deploy. Health endpoint: `/health`.
5. Visit `https://<your-app>.onrender.com/app` for UI.

### Environment Variables (summary)
| Key | Purpose |
|-----|---------|
| OPENROUTER_API_KEY | Enables multi‑model LLM cascade |
| GEMINI_API_KEY | Gemini fallback layer |
| MODEL_NAME | Primary preferred model |
| GEMINI_MODEL | Optional Gemini override |
| UPLOAD_DIR | Storage path (default `uploads`) |
| PORT | Render internal port (auto) |

### Local VM (Alternative, No Render)
If you prefer a bare server:
```bash
python -m venv venv
venv/Scripts/activate  # Windows adjust
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Systemd unit (Linux) similar to earlier—see git history if needed.

### Static Frontend Split
Host `frontend/` on a static host (GitHub Pages / Netlify) and set a global `BACKEND_BASE_URL` in your JS.

### Smoke Test
```bash
curl https://<render-host>/health
curl -F "file=@resume.pdf" "https://<render-host>/resume/api/analyze?job_description=Test"
curl -F "file=@resume.pdf" "https://<render-host>/resume/api/generate?job_description=Test&plain=1"
```

## 🧪 AI Model Cascade & Fallback Logic

Sequence (first successful valid JSON wins):

1. `MODEL_NAME` (env) – defaults to first element in priority (`deepseek/deepseek-chat`)
2. `google/gemini-pro-1.5`
3. `deepseek/deepseek-chat`
4. `openai/gpt-oss-20b:free`
5. `deepseek/deepseek-chat-v3.1:free`
6. `tngtech/deepseek-r1t2-chimera:free`
7. `z-ai/glm-4.5-air:free`
8. `deepseek/deepseek-r1-0528:free`
9. `deepseek/deepseek-r1:free`
10. `microsoft/mai-ds-r1:free`
11. `qwen/qwen3-235b-a22b:free`
12. `google/gemini-2.0-flash-exp:free`
13. `meta-llama/llama-4-maverick:free`

Gemini fallback iterates internal list (`gemini-2.5-flash`, `gemini-2.5-pro`, etc.). If all remote attempts fail (keys missing / rate limit) a deterministic heuristic builder generates:

- Summary, Skills, Experience, Education blocks
- Matched vs missing keywords
- Approximate ATS breakdown

Inspect `model_attempts` (when JSON mode enabled) to debug failures.

Prefer a specific primary model:

```bash
set MODEL_NAME=deepseek/deepseek-chat
```

Heuristic‑only mode: omit both `OPENROUTER_API_KEY` and `GEMINI_API_KEY`.

## 🤝 Contributing

1. Branch from `main`
2. Keep secrets out of commits
3. Add/update tests for behavioral changes
4. Submit PR with clear summary + screenshots (if UI)

## 📄 License

MIT License — provided “as is” without warranty.

---

Feel free to open issues for: UI toggle for plain/detailed ATS mode, rate limiting, multi-user sessions, or improved metrics injection.
