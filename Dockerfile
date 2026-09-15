FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 fonts-dejavu-core \
    git curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Install CPU-only torch wheel from official index
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.0+cpu --trusted-host download.pytorch.org

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3", "handler.py"]
