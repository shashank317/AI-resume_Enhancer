# ==============================
# Dockerfile for AI Resume Enhancer
# ==============================

# 1️⃣ Use official Python slim image (small & fast)
FROM python:3.11-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Environment variables
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# 4️⃣ Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5️⃣ Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm || true

# 6️⃣ Copy project files into container
COPY . .

# 7️⃣ Expose the app port
EXPOSE 8000

# 8️⃣ Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
