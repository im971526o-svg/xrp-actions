FROM python:3.11-slim
WORKDIR /app

# 安裝相依套件
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY app ./app
COPY openapi.yaml ./openapi.yaml

# 給本機測試用的預設埠（Render 會覆蓋為 $PORT）
EXPOSE 10000

# 關鍵：使用環境變數 PORT（Render 會提供）
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
