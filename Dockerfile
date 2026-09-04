FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TTS_HOME=/cache/tts

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && python3.11 -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data /cache/huggingface /cache/tts \
    && chown -R appuser:appuser /app /cache

USER appuser

CMD ["python3.11", "-m", "app.main"]
