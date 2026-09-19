FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies, FFmpeg, and graphic rendering libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    ca-certificates \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

# Download Google's Noto Sans Devanagari font for Hindi rendering
RUN curl -L -o /app/NotoSansDevanagari.ttf "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari-Bold.ttf"

# Upgrade pip and wheel ONLY. Do NOT upgrade setuptools here. 
RUN python3 -m pip install --upgrade pip wheel

# Install dependencies (this will install the safe, pinned version of setuptools)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your transparent custom character sprites
COPY my_scene1.png /app/my_scene1.png
COPY my_scene2.png /app/my_scene2.png
COPY my_scene3.png /app/my_scene3.png

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
