FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY main.py .
COPY best.pt .

# Port - Railway uses dynamic PORT
ENV PORT=8000
EXPOSE $PORT

# Run - Use shell form to read PORT env variable
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
