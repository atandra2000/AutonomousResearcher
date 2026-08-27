FROM python:3.12-slim

# Non-root container, no privileged capabilities required.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

# P1: deployment image runs the deterministic benchmark/service stack only.
# All dependencies are installed OFFLINE from vendored linux/arm64 wheels so
# sentence-transformers satisfies torch with a CPU wheel (no multi-GB CUDA
# packages) and container builds never depend on flaky in-VM network access.
# Refresh wheels on an arm64 host with:
#   /tmp/pipenv/bin/python -m pip download --only-binary=:all: \
#     --python-version 312 --platform manylinux2014_aarch64 \
#     --platform manylinux_2_17_aarch64 --platform manylinux_2_28_aarch64 \
#     --platform linux_aarch64 -d deploy/wheels ".[telemetry,service]" \
#     "deploy/wheels/torch-...+cpu....whl"
#   /tmp/pipenv/bin/python -m pip download --only-binary=:all: -d deploy/wheels \
#     hatchling pathspec pluggy packaging tomlkit trove-classifiers
COPY pyproject.toml README.md uv.lock* ./
COPY deploy/wheels/ /wheels/

RUN pip install --no-cache-dir --no-index --find-links=/wheels hatchling

COPY src ./src
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
        --no-build-isolation ".[telemetry,service]" \
    && rm -rf /wheels

ENV RE_SERVICE_ARTIFACT_DIR=/app/data/artifacts \
    RE_CHECKPOINT_DB=/app/data/checkpoints.db \
    RE_SERVICE_DB_PATH=/app/data/run_service.db

RUN mkdir -p /app/data/artifacts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default command runs the API; workers use:
#   python -m research_engineer.service.serve worker
CMD ["python", "-m", "research_engineer.service.serve", "api"]
