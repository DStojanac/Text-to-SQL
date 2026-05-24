# HuggingFace Docker Space — FastAPI inference backend (GPU).
# HF Spaces expose port 7860 by default.

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY databases/ databases/

EXPOSE 7860

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
