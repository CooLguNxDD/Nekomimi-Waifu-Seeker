# Slim web image — keyword decision fallback by default.
# For Laya: docker compose --profile laya up --build
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WAIFU_FORCE_FALLBACK=0 \
    USE_TF=0

COPY requirements-docker.txt .
RUN pip install --upgrade pip && pip install -r requirements-docker.txt

COPY pyproject.toml README.md ./
COPY data ./data
COPY waifu_engine ./waifu_engine

RUN pip install -e .

EXPOSE 7860

CMD ["uvicorn", "waifu_engine.web:app", "--host", "0.0.0.0", "--port", "7860"]
