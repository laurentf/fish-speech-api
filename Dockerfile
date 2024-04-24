FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn huggingface_hub python-multipart

# Copy app code
COPY fish_speech/ ./fish_speech/
COPY tools/ ./tools/
COPY server.py .
COPY .project-root .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create checkpoints directory
RUN mkdir -p checkpoints

# Expose port
EXPOSE 8080

# Entrypoint handles model download + server start
ENTRYPOINT ["./entrypoint.sh"]
