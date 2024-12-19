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
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata\
    https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim_vert.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/chi_tra.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/chi_tra_vert.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/deu.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/enm.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/equ.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/fil.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/jpn.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/jpn_vert.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/mya.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/spa.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/tha.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/vie.traineddata

RUN mkdir -p /usr/local/share/tessdata/script && \
    wget -P /usr/local/share/tessdata/script \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/Hangul.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/HanS.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/HanS_vert.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/Japanese.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/Japanese_vert.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/Latin.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/Myanmar.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/Thai.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/script/Vietnamese.traineddata

ENV TESSDATA_PREFIX=/usr/local/share/tessdata

WORKDIR /app

RUN mkdir -p /app/Public
COPY Public /app/Public

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN tesseract --version
RUN tesseract --list-langs

ENTRYPOINT ["python", "server.py"]