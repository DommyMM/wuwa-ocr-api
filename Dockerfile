# =======================
# Builder stage
# =======================
FROM python:3.13-slim AS builder

WORKDIR /build

# Copy requirements and install as user (for easy copying later)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =======================
# Runtime stage
# =======================
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY Data /app/Data
COPY *.py /app/

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# OpenCV headless mode
ENV OPENCV_HEADLESS=1

EXPOSE 5000

ENTRYPOINT ["python", "server.py"]