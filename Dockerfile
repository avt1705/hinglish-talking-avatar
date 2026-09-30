FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip "setuptools<70.0.0" wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["python3", "-u", "handler.py"]
