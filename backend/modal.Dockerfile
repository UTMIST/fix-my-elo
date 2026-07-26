# Engine service: torch + model + MCTS. Kept slim by using the CPU-only torch
# wheel instead of the default CUDA build.
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.11.0

COPY engine_requirements.txt api_requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=engine_requirements.txt,target=engine_requirements.txt \
    python -m pip install -r engine_requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=api_requirements.txt,target=api_requirements.txt \
    python -m pip install -r api_requirements.txt

COPY . .


