FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY openqasm/ ./openqasm/

ENV PYTHONPATH=/app
ENV EXCHANGE_ID=binanceus

ENTRYPOINT ["python", "-m", "src.paper_trading.engine"]