FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAPSTONE_ASR_PROVIDER=whisper \
    CAPSTONE_TTS_PROVIDER=pocket_tts

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg espeak-ng && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY orchestrator/ ./orchestrator/
COPY services/ ./services/
COPY ml/ ./ml/
COPY web/ ./web/
RUN mkdir -p storage/temp_audio storage/cloned_voices storage/benchmarks models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health').read()"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
