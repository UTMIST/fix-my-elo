# syntax=docker/dockerfile:1
# Engine service: torch + model + MCTS. Kept slim by using the CPU-only torch
# wheel instead of the default CUDA build.
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Non-privileged user to run the app.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install the CPU-only torch build first (avoids multi-GB CUDA libraries).
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# Then the remaining engine dependencies from PyPI.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=engine_requirements.txt,target=engine_requirements.txt \
    python -m pip install -r engine_requirements.txt

USER appuser

COPY . .

EXPOSE 8001
CMD uvicorn 'services.engine_service.engine_server:app' --host=0.0.0.0 --port=8001
