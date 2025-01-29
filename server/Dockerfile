FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY api.py .
COPY generate_speech.py .
COPY process_audio.py .
COPY save_embeddings.py .
COPY cache_voice_latents.py .
COPY generate_voice.py .

# Copy voice profiles
COPY voices/ /app/voices/

# Pre-download models
RUN python -c "from generate_speech import VoiceConditioningPipeline; VoiceConditioningPipeline()"

# Create and set permissions for required directories
RUN mkdir -p /app/output /app/train_voices && \
    chmod -R 755 /app/output /app/train_voices /app/voices

EXPOSE 8000

# Add environment variables for debugging
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV UVICORN_LOG_LEVEL=debug
ENV UVICORN_ACCESS_LOG=1

# Run with reload for development
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug", "--access-log"]
