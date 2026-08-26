FROM python:3.12-slim

# Non-root container, no privileged capabilities required.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY pyproject.toml README.md uv.lock* ./
COPY src ./src
RUN pip install --no-cache-dir ".[telemetry,service]"

ENV RE_SERVICE_ARTIFACT_DIR=/app/data/artifacts \
    RE_CHECKPOINT_DB=/app/data/checkpoints.db \
    RE_SERVICE_DB_PATH=/app/data/run_service.db

RUN mkdir -p /app/data/artifacts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default command runs the API; workers use:
#   python -m research_engineer.service.serve worker
CMD ["python", "-m", "research_engineer.service.serve", "api"]
