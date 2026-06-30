# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for pandas / numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="LLM Product Normalizer"
LABEL description="Offline LLM product data normalization pipeline"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY . .

# Create required directories
RUN mkdir -p data logs

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# ── Environment defaults (override via docker-compose or -e flags) ─────────────
ENV OLLAMA_HOST=http://ollama:11434
ENV OLLAMA_MODEL=llama3.2
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Default: start FastAPI (override in docker-compose for other services)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
