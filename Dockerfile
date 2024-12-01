FROM python:3.9-slim
RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    g++ \
    git \
    libarchive-dev \
    libc6 \
    libc6-dev \
    libcurl4-openssl-dev \
    libgif-dev \
    libicu-dev \
    libjpeg-dev \
    libleptonica-dev \
    libopencv-dev \
    libpng-dev \
    libtiff-dev \
    libtool \
    libwebp-dev \
    make \
    pkg-config \
    python3-opencv \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tesseract-ocr/tesseract.git && \
    cd tesseract && \
    git checkout 5.5.0 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf tesseract

RUN mkdir -p /usr/local/share/tessdata
RUN wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim_vert.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_tra.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_tra_vert.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/deu.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/enm.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/equ.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/fil.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/fra.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/hin.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/jpn.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/jpn_vert.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/mya.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/spa.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/tha.traineddata && \
    wget -P /usr/local/share/tessdata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/vie.traineddata

ENV TESSDATA_PREFIX=/usr/local/share/tessdata

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN tesseract --version
RUN tesseract --list-langs

ENTRYPOINT ["python", "server.py"]