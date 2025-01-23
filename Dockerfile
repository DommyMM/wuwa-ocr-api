FROM python:3.9-slim

# Setup directories and files
WORKDIR /app
COPY requirements.txt .
COPY Data /app/Data
COPY *.py /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    autoconf automake build-essential ca-certificates g++ git \
    libarchive-dev libc6 libc6-dev libcurl4-openssl-dev libgif-dev \
    libicu-dev libjpeg-dev libleptonica-dev libopencv-dev libpng-dev \
    libtiff-dev libtool libwebp-dev make pkg-config python3-opencv wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Tesseract
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

ENV TESSDATA_PREFIX=/usr/local/share/tessdata
ENTRYPOINT ["python", "server.py"]