FROM python:3.11-slim
WORKDIR /app

# 安裝套件
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY app ./app
COPY openapi.yaml ./openapi.yaml

# 本機測試預設埠；在 Render 會被 $PORT 覆蓋
EXPOSE 10000

# 用 python -m uvicorn（避免 PATH 問題）；一定是 --port（兩個減號）
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
