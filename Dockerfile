FROM python:3.10-slim

WORKDIR /app

# Install system dependencies: ffmpeg + fonts + build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    fonts-dejavu-core \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENTRYPOINT ["python", "handler.py"]
