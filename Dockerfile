# TacitGraph — chat app and pipeline CLIs in one image (local NLP mode, CPU only).
#
#   docker compose up                       # chat app on http://localhost:7861
#   docker compose run --rm tacitgraph tacitgraph-ingest --pst data/archive.pst --output data

# ---- build stage: resolve and compile dependencies ---------------------------
FROM python:3.11-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

# Compiler toolchain for libpff-python (PST parsing)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
        --extra ingest --extra nlp --group models

COPY README.md LICENSE NOTICE ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ingest --extra nlp --group models

# ---- runtime stage -------------------------------------------------------------
FROM python:3.11-slim

# antiword: legacy .doc attachments; poppler-utils: PDF page rendering
RUN apt-get update \
    && apt-get install -y --no-install-recommends antiword poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

ENV PYTHONUNBUFFERED=1 \
    TACITGRAPH_HOME=/app \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY --from=build --chown=app:app /app /app
COPY --chown=app:app config/ config/
# The model cache must exist and be owned by `app` before Docker mounts a volume on it
RUN mkdir -p data logs /home/app/.cache/huggingface \
    && chown -R app:app data logs /home/app/.cache

USER app

EXPOSE 7861

CMD ["tacitgraph-app", "--mode", "local", "--port", "7861"]
