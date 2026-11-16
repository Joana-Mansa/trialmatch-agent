# trialmatch-agent HTTP API.
# Build:  docker build -t trialmatch-agent .
# Run:    docker run --rm -p 127.0.0.1:8000:8000 -e ANTHROPIC_API_KEY -e TRIALMATCH_API_TOKEN trialmatch-agent
# Set TRIALMATCH_API_TOKEN to require a bearer token; without it the API is open.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package (non-editable) first so code-only changes reuse this layer
# when dependencies are unchanged.
COPY pyproject.toml ./
COPY src ./src
RUN pip install . && rm -rf /app/src

# Synthetic patient bundles only. The installed package cannot locate the repo's
# data/ folder, so point the store at it explicitly.
COPY data ./data
ENV TRIALMATCH_PATIENT_DIR=/app/data/patients \
    TRIALMATCH_HTTP=live

RUN useradd --create-home --uid 10001 trialmatch
USER trialmatch

EXPOSE 8000

# python:3.12-slim has no curl; use the stdlib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4); sys.exit(0 if r.status == 200 else 1)"

# One worker on purpose: the job store is in-memory and per-process (see api.py).
CMD ["uvicorn", "trialmatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
