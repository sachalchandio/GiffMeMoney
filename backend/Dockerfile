# ---------------------------------------------------------------------------
# GiffMeMoney backend image (FastAPI + uvicorn).
#
# Build context: the repository root (so `backend/` is copied as a whole).
#   docker build -f backend/Dockerfile -t giffmemoney-backend .
#
# Runtime is the safe default path out of the box: simulated market data,
# the simulated (paper) broker, and in-memory storage. Every live capability
# is OPT-IN via environment variables — see `.env.example` and `docs/DEPLOY.md`.
# Nothing in this image enables real-money trading; that stays a deliberate,
# documented env/config action (ALPACA_LIVE + live base URL + BROKER_ACK).
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc files in the image layer.
# - PYTHONUNBUFFERED: stream logs straight to stdout (so `docker logs` is live).
# - PIP_NO_CACHE_DIR: keep the image small.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first (a separate layer) so application code changes
# don't bust the pip cache. requirements.txt is pinned to mutually-compatible
# minor ranges for Python 3.11.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend package itself (the `app` module + run.py).
COPY backend/ ./

# Run as a non-root user. Create it after the copy so it owns /app (needed when
# GIFF persistence is enabled and SQLite writes giffmemoney.db into the workdir).
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Lightweight liveness probe against the public health route (no extra deps:
# Python's stdlib urllib is always present).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else sys.exit(1)" \
    || exit 1

# Production entrypoint: run uvicorn directly (no --reload). run.py is for local
# development only. Bind to all interfaces so the container is reachable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
