FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    autoconf automake build-essential ca-certificates g++ git \
    libarchive-dev libc6 libc6-dev libcurl4-openssl-dev libgif-dev \
    libicu-dev libjpeg-dev libleptonica-dev libopencv-dev libpng-dev \
    libtiff-dev libtool libwebp-dev make pkg-config python3-opencv wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup application
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create and verify data directories
RUN mkdir -p /app/Data/Icons

# Copy data files with explicit paths
COPY Data/*.json /app/Data/
COPY Data/Icons/*.png /app/Data/Icons/

# Copy application code
COPY *.py /app/

# Install Tesseract and verify setup
RUN git clone https://github.com/tesseract-ocr/tesseract.git && \
    cd tesseract && \
    git checkout 5.5.0 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf tesseract && \
    mkdir -p /usr/local/share/tessdata && \
    wget -P /usr/local/share/tessdata https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Verify data files and template loading
RUN echo "Verifying data structure:" && \
    ls -la /app/Data/ && \
    echo "Verifying icons:" && \
    ls -la /app/Data/Icons/ && \
    echo "Testing template loading:" && \
    python -c "from data import TEMPLATE_FEATURES; print(f'Templates loaded: {len(TEMPLATE_FEATURES)}')" && \
    echo "Verifying Tesseract:" && \
    tesseract --version && \
    tesseract --list-langs

ENV TESSDATA_PREFIX=/usr/local/share/tessdata

ENTRYPOINT ["python", "server.py"]