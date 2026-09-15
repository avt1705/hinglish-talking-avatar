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
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Pre-download Stable Diffusion model into image
RUN git lfs install && \
    git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 /models/sd15

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Keep container alive to process jobs
CMD ["python", "handler.py"]
