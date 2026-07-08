# Invoice OCR sidecar — CPU build.
# Model weights (~20 MB) download on first OCR request; mount /root/.paddleocr
# as a volume so they survive container restarts.
FROM python:3.12-slim

# PaddleOCR/OpenCV runtime libs (not present in -slim).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8100/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
