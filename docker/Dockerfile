# Multi-stage build for the trading agent system.

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download FinBERT model to avoid cold-start latency at runtime
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert')"


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
# Copy cached HuggingFace models
COPY --from=builder /root/.cache /root/.cache

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy source
COPY agents/      ./agents/
COPY config/      ./config/
COPY storage/     ./storage/
COPY scripts/     ./scripts/
COPY models/      ./models/
COPY main.py      .

# SQLite data volume (mounted at runtime)
VOLUME /app/data

# Healthcheck: orchestrator must heartbeat every 2 minutes
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s \
  CMD python scripts/healthcheck.py --max-age-seconds=180 || exit 1

CMD ["python", "main.py"]
