# XRP Pattern Scanner Actions (FastAPI)

This is a minimal backend you can deploy and connect to your custom GPT via **Actions**.
It fetches OHLCV for **XRP/USDT** from Binance using **CCXT**, detects a handful of classic candlestick patterns,
and exposes two endpoints:
- `GET /scan` — scan recent candles for patterns across intervals (15m, 1h, 4h, 1d)
- `GET /explain` — return evidence and a compact explanation for a specific detection

## Quickstart

```bash
# 1) Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs to try it.

### Example
- Scan the default set (XRP/USDT on 15m,1h,4h,1d):
  `GET http://localhost:8000/scan`

- Ask for details:
  `GET http://localhost:8000/explain?pattern_id=<the id from /scan>`

## Deploy
You can deploy to any place that runs Python (VM, Docker, Render, Fly, Railway). For Docker:

```bash
docker build -t xrp-actions .
docker run -p 8000:8000 xrp-actions
```

## Connect to your Custom GPT
1. Open **GPT Builder → Configure → Actions → Upload OpenAPI schema** (use `openapi.yaml` in this folder).
2. Approve the domain when prompted.
3. In your system instructions, tell the GPT to call `/scan` first and then `/explain` for any found pattern.

## Cron / Scheduled refresh (optional)
This demo fetches market data on-demand. If you want to precompute and cache detections, set up a cron job that calls:
```
curl "<YOUR_DEPLOY_URL>/scan?intervals=15m,1h,4h,1d&lookback=500"
```
Store results on your side if you want long-term statistics.

## Notes
- Exchange limits: Binance allows 1000 candles per call. This demo fetches up to `lookback` (<= 1000) per interval per symbol.
- This demo includes a few patterns for clarity. Extend `app/patterns.py` with your own rules.
- For RAG: Map `pattern_name` to your book passages in your GPT (Knowledge), or build a small retrieval endpoint that returns the relevant snippets for the detected pattern and include them inside `/explain`.
