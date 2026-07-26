ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /app

RUN pip install modal python-dotenv

COPY . .

CMD modal serve --env=FME-engine  services/engine_service/modal_api.py