# crypto-quant-model Dockerfile
# 利於 24/7 雲端運行

FROM python:3.11-slim

WORKDIR /app

# 系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先複製依賴檔案以利用 Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案原始碼
COPY . .

# 預設啟動命令（可依需求調整）
CMD ["python", "cli.py", "--help"]
