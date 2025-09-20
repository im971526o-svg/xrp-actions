# app/main.py
# ============================================================================
# XRP Pattern Scanner Actions - single-file backend
# Endpoints:
#   GET  /                -> root OK
#   GET  /price           -> 現價（含本地時間字串）
#   GET  /klines          -> K 線資料 rows (ts/open/high/low/close/volume)
#   GET  /indicators      -> 取得最新一根的技術指標（BBands/KDJ/MACD）
#   GET  /chart           -> 停用（501），避免依賴圖片
#   GET  /chart/quick     -> 停用（501）
#   POST /experience/log  -> 寫入經驗（SQLite）
#   GET  /experience/recent -> 讀取最近經驗
#   GET  /auto/status     -> 自動交易開關狀態
#   POST /auto/enable     -> 開/關自動交易（需 X-AUTO-SECRET）
#   GET  /positions/open  -> (新增) 虛擬/乾跑持倉列表
#   GET  /pnl/trades      -> (新增) 成交明細（虛擬）
#   GET  /pnl/summary     -> (新增) 勝率/盈虧摘要（虛擬）
# ============================================================================

import io
import math
import time
import os
from datetime import datetime, timezone
from typing import List

import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Query, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

# 市場資料 (ccxt)
import ccxt

# SQLite 永久儲存（經驗日誌）
from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

APP_NAME = "XRP Pattern Scanner Actions"
app = FastAPI(title=APP_NAME, version="0.1.3")

# CORS：預設開放（方便你在不同環境測）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Exchange 初始化（Binance）
# -----------------------------------------------------------------------------
exchange = ccxt.binance({"enableRateLimit": True})
try:
    exchange.options["adjustForTimeDifference"] = True
except Exception:
    pass

# -----------------------------------------------------------------------------
# SQLite 初始化（經驗日誌）
# -----------------------------------------------------------------------------
Base = declarative_base()
engine = create_engine("sqlite:///data.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class Experience(Base):
    __tablename__ = "experiences"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False)
    pattern = Column(String(64), nullable=False)   # e.g. hammer / three_white_soldiers ...
    outcome = Column(String(16), nullable=False)   # bullish / bearish / neutral
    notes = Column(Text, nullable=True)
    ts = Column(DateTime(timezone=True), nullable=False)

Base.metadata.create_all(bind=engine)

# -----------------------------------------------------------------------------
# Helper / Utils
# -----------------------------------------------------------------------------
TF_ALLOW = {"5m", "15m", "30m", "1h", "4h", "1d"}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_taipei_ms(ms:int) -> datetime:
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone()

def zh_time(dt: datetime) -> str:
    h = dt.hour
    m = dt.minute
    ap = "上午" if h < 12 else "下午"
    h12 = h % 12 or 12
    return f"{ap} {h12:02d} 點 {m:02d} 分"

def fetch_ohlcv(symbol: str, interval: str, limit: int) -> List[dict]:
    if interval not in TF_ALLOW:
        raise HTTPException(400, f"interval must be one of {sorted(TF_ALLOW)}")
    limit = clamp(limit, 50, 500)
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    except Exception as e:
        raise HTTPException(502, f"fetch_ohlcv error: {e}")

    rows = []
    for ts, o, h, l, c, v in raw:
        rows.append({
            "ts": int(ts),
            "open": float(o), "high": float(h),
            "low": float(l), "close": float(c),
            "volume": float(v)
        })
    return rows

def fetch_price(symbol: str, source: str = "ticker") -> float:
    try:
        if source == "book":
            ob = exchange.fetch_order_book(symbol, limit=5)
            best_bid = ob["bids"][0][0] if ob["bids"] else None
            best_ask = ob["asks"][0][0] if ob["asks"] else None
            if best_bid is None or best_ask is None:
                raise Exception("empty orderbook")
            return float((best_bid + best_ask) / 2)
        else:
            t = exchange.fetch_ticker(symbol)
            px = t.get("last") or t.get("close")
            if px is None:
                raise Exception("ticker has no last/close")
            return float(px)
    except Exception as e:
        raise HTTPException(502, f"fetch_price error: {e}")

# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/", summary="Root")
def root():
    return {"ok": True, "service": APP_NAME}

# -----------------------------------------------------------------------------
# /price
# -----------------------------------------------------------------------------
@app.get("/price", summary="現價")
def get_price(
    symbol: str = Query(..., example="XRP/USDT"),
    precision: int = Query(4, ge=0, le=10, description="小數位數"),
    source: str = Query("ticker", regex="^(ticker|book)$", description="ticker 或 book")
):
    px = fetch_price(symbol, source=source)
    ts = int(time.time() * 1000)
    local = to_taipei_ms(ts)
    return {
        "exchange": "binance",
        "symbol": symbol,
        "price": round(px, precision),
        "ts": ts,
        "time_local": zh_time(local)
    }

# -----------------------------------------------------------------------------
# /klines
# -----------------------------------------------------------------------------
@app.get("/klines", summary="K 線")
def get_klines(
    symbol: str = Query(..., example="XRP/USDT"),
    interval: str = Query("15m", regex="^(5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(200, ge=50, le=500)
):
    rows = fetch_ohlcv(symbol, interval, limit)
    return {"exchange": "binance", "symbol": symbol, "interval": interval, "rows": rows}

# -----------------------------------------------------------------------------
# /indicators（回傳最新一根的指標）
# -----------------------------------------------------------------------------
@app.get("/indicators", summary="指標（最新一根）")
def get_indicators(
    symbol: str = Query(..., example="XRP/USDT"),
    interval: str = Query("15m", regex="^(5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(200, ge=50, le=500),
    bbands: int = Query(20, ge=5, le=100),
    sigma: float = Query(2.0, ge=0.5, le=4.0),
):
    rows = fetch_ohlcv(symbol, interval, limit)
    if len(rows) < max(bbands, 26) + 5:
        raise HTTPException(400, "not enough data for indicators")

    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # BBands
    mid = close.rolling(bbands).mean()
    std = close.rolling(bbands).std(ddof=0)
    upper = mid + sigma * std
    lower = mid - sigma * std

    # KDJ (9,3,3) with safe denominator
    low_n  = low.rolling(9).min()
    high_n = high.rolling(9).max()
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = ((close - low_n) / denom * 100).fillna(0.0)
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3*k - 2*d

    # MACD (12,26,9)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_val = ema_fast - ema_slow
    signal = macd_val.ewm(span=9, adjust=False).mean()
    hist = macd_val - signal

    last = rows[-1]
    return {
        "exchange": "binance",
        "symbol": symbol,
        "interval": interval,
        "last": {
            "open": last["open"], "high": last["high"], "low": last["low"],
            "close": last["close"], "volume": last["volume"],
        },
        "bbands": {
            "upper": round(float(upper.iloc[-1]), 6),
            "middle": round(float(mid.iloc[-1]), 6),
            "lower": round(float(lower.iloc[-1]), 6),
            "sigma": sigma
        },
        "kdj": {
            "k": round(float(k.iloc[-1]), 3),
            "d": round(float(d.iloc[-1]), 3),
            "j": round(float(j.iloc[-1]), 3),
        },
        "macd": {
            "dif": round(float(macd_val.iloc[-1]), 6),
            "dea": round(float(signal.iloc[-1]), 6),
            "hist": round(float(hist.iloc[-1]), 6),
            "fast": 12, "slow": 26, "signal": 9
        }
    }

# -----------------------------------------------------------------------------
# /chart 停用（返回 501）
# -----------------------------------------------------------------------------
@app.get("/chart", summary="(停用) 圖表輸出", tags=["chart"])
def get_chart_disabled(
    symbol: str = Query(..., example="XRP/USDT"),
    interval: str = Query(..., regex="^(5m|15m|30m|1h|4h|1d)$", example="15m"),
    limit: int = Query(200, ge=50, le=200, example=200),
    sma20: int = Query(20, ge=2, le=200, description="SMA 週期"),
    bbands: int = Query(20, ge=5, le=100, description="布林基準週期"),
    sigma: float = Query(2.0, ge=0.5, le=4.0, description="布林標準差倍數"),
    show_kdj: bool = Query(False),
    show_macd: bool = Query(False),
    volume: bool = Query(True),
    width: int = Query(980, ge=320, le=1920),
    height: int = Query(480, ge=320, le=1080),
    dpi: int = Query(110, ge=72, le=220),
    tight: bool = Query(True),
    ext: str = Query("png", regex="^(png|webp|jpeg)$"),
):
    return JSONResponse(
        status_code=501,
        content={
            "error": "chart_disabled",
            "message": "圖表輸出暫時停用。請改用 /klines 與 /indicators 取得 K 線與指標數據；GPT 分析不受影響。"
        },
    )

@app.get("/chart/quick", summary="(停用) 快速 15m 圖表", tags=["chart"])
def get_chart_quick_disabled(
    symbol: str = Query(..., example="XRP/USDT"),
    interval: str = Query("15m", regex="^(5m|15m|30m|1h|4h|1d)$"),
):
    return JSONResponse(
        status_code=501,
        content={
            "error": "chart_disabled",
            "message": "圖表輸出暫時停用。請改用 /klines 與 /indicators。"
        },
    )

# -----------------------------------------------------------------------------
# Experience Log（持久化）
# -----------------------------------------------------------------------------
@app.post("/experience/log", summary="寫入經驗")
def post_experience_log(
    payload: dict = Body(..., example={
        "symbol": "XRP/USDT", "interval": "15m",
        "pattern": "hammer", "outcome": "bullish",
        "notes": "收盤突破MA30，量能溫和放大。"
    })
):
    required = {"symbol", "interval", "pattern", "outcome"}
    if not required.issubset(payload.keys()):
        raise HTTPException(400, f"missing fields: {required - set(payload.keys())}")

    interval = payload["interval"]
    if interval not in TF_ALLOW:
        raise HTTPException(400, f"interval must be in {sorted(TF_ALLOW)}")

    db = SessionLocal()
    try:
        rec = Experience(
            symbol=payload["symbol"],
            interval=payload["interval"],
            pattern=payload["pattern"],
            outcome=payload["outcome"],
            notes=payload.get("notes") or "",
            ts=datetime.now(timezone.utc),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return {
            "id": rec.id,
            "symbol": rec.symbol,
            "interval": rec.interval,
            "pattern": rec.pattern,
            "outcome": rec.outcome,
            "notes": rec.notes,
            "ts": int(rec.ts.timestamp() * 1000),
            "time_local": zh_time(rec.ts.astimezone()),
        }
    finally:
        db.close()

@app.get("/experience/recent", summary="最近經驗")
def get_experience_recent(limit: int = Query(20, ge=1, le=100)):
    db = SessionLocal()
    try:
        q = db.query(Experience).order_by(Experience.id.desc()).limit(limit).all()
        items = []
        for r in q:
            items.append({
                "id": r.id, "symbol": r.symbol, "interval": r.interval,
                "pattern": r.pattern, "outcome": r.outcome,
                "notes": r.notes,
                "ts": int(r.ts.timestamp() * 1000),
                "time_local": zh_time(r.ts.astimezone())
            })
        return {"items": items}
    finally:
        db.close()

# -----------------------------------------------------------------------------
# Auto Trading Toggle (minimal)
# -----------------------------------------------------------------------------
AUTO_SECRET = os.getenv("AUTO_SECRET", "")  # 在 Render 環境變數設定同一個值
_auto_enabled = {"enabled": False}

@app.get("/auto/status", summary="Auto-trading status")
def auto_status():
    return {"enabled": _auto_enabled["enabled"]}

@app.post("/auto/enable", summary="Enable/disable auto-trading")
def auto_enable(enabled: bool, x_auto_secret: str = Header(default="")):
    if not AUTO_SECRET or x_auto_secret != AUTO_SECRET:
        return {"ok": False, "error": "unauthorized"}
    _auto_enabled["enabled"] = bool(enabled)
    return {"ok": True, "enabled": _auto_enabled["enabled"]}

# -----------------------------------------------------------------------------
# >>> PNL API (新增)：配合 DRY_RUN 虛擬倉位檔 VIRTUAL_STATE_PATH
# -----------------------------------------------------------------------------
VSTATE_PATH = os.getenv("VIRTUAL_STATE_PATH", "virtual_state.json")

def _vload():
    try:
        with open(VSTATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"positions": [], "trades": []}

@app.get("/positions/open", summary="目前虛擬持倉（DRY_RUN）", tags=["pnl"])
def list_open_positions():
    s = _vload()
    return {"count": len(s.get("positions", [])), "positions": s.get("positions", [])}

@app.get("/pnl/trades", summary="成交明細（DRY_RUN）", tags=["pnl"])
def list_trades(limit: int = Query(100, ge=1, le=1000)):
    s = _vload()
    trades = s.get("trades", [])
    if limit and len(trades) > limit:
        trades = trades[-limit:]
    return {"count": len(trades), "trades": trades}

@app.get("/pnl/summary", summary="勝率/盈虧摘要（DRY_RUN）", tags=["pnl"])
def pnl_summary():
    s = _vload()
    trades = s.get("trades", [])
    n = len(trades)
    if n == 0:
        return {"count": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "open_positions": len(s.get("positions", []))}
    pnls = [float(t.get("pnl_usdt", 0)) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = sum(pnls)
    wr = (len(wins) / n) if n else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    return {
        "count": n,
        "win_rate": round(wr, 4),
        "total_pnl": round(total, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "open_positions": len(s.get("positions", [])),
        "updated_at": int(time.time()),
    }
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# main（本地測試用）
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
