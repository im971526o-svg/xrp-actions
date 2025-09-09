# app/main.py
# =========================================================
# XRP Pattern Scanner Actions - Full API
# Endpoints:
#   GET /           : health
#   GET /price      : 現價（預設用買賣一中間價），回傳 time_local=「上午/下午 HH 點 mm 分」
#   GET /klines     : 取得 OHLCV
#   GET /indicators : BB(20,2) / KDJ(9,3,3) / MACD(12,26,9)
#   GET /chart      : 圖表（PNG），可開關：BB、KDJ、MACD、成交量（volume）
#   POST /experience/log, GET /experience/recent : 簡易經驗紀錄（示範用、存在記憶體）
# =========================================================
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import ccxt

import matplotlib
matplotlib.use("Agg")  # 重要：Render 無視窗環境要用 Agg
import matplotlib.pyplot as plt

APP_NAME = "XRP Pattern Scanner Actions"

# ---------------------------
# 初始化交易所（Binance Spot）
# ---------------------------
binance = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
        "adjustForTimeDifference": True,
    }
})
binance.load_markets()

# 支援的時間框架（ccxt 對應）
ALLOWED_INTERVALS = {"5m", "15m", "30m", "1h", "4h", "1d"}

# ---------------------------
# 小工具
# ---------------------------
def to_local_str(ts_ms: int, tz: str = "Asia/Taipei") -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, ZoneInfo(tz))
    ampm = "上午" if dt.hour < 12 else "下午"
    hh = dt.hour if dt.hour <= 12 else dt.hour - 12
    if hh == 0:
        hh = 12
    return f"{ampm} {hh:02d} 點 {dt.minute:02d} 分"

def r4(x: float) -> float:
    return float(f"{x:.4f}")

def fetch_price(symbol: str, source: str = "book", precision: int = 4) -> Dict[str, Any]:
    """
    source='book'   : 取 orderbook 買一/賣一中間價（較貼近即時）
    source='ticker' : 取上一筆成交價（last），沒有就 bid/ask 中間
    """
    if source == "book":
        ob = binance.fetch_order_book(symbol, 5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if bid is None or ask is None:
            t = binance.fetch_ticker(symbol)
            last = t.get("last")
            bid = t.get("bid")
            ask = t.get("ask")
            price = last or ((bid or 0) + (ask or 0)) / 2
            ts = t.get("timestamp") or binance.milliseconds()
        else:
            price = (bid + ask) / 2
            ts = ob.get("timestamp") or binance.milliseconds()
    else:
        t = binance.fetch_ticker(symbol)
        last = t.get("last")
        bid = t.get("bid")
        ask = t.get("ask")
        price = last or ((bid or 0) + (ask or 0)) / 2
        ts = t.get("timestamp") or binance.milliseconds()

    if precision is not None:
        price = float(f"{price:.{precision}f}")
    else:
        price = r4(price)

    return {
        "exchange": "binance",
        "symbol": symbol,
        "price": price,
        "ts": ts,
        "time_local": to_local_str(ts),
    }

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval 必須為 {sorted(ALLOWED_INTERVALS)}")
    if limit > 1000:
        raise HTTPException(status_code=400, detail="limit 最多 1000")

    raw = binance.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    # raw: [ts, open, high, low, close, volume]
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def indicators(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if len(df) == 0:
        return out

    # BB(20,2)
    if len(df) >= 20:
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std(ddof=0)
        upper = mid + 2 * std
        lower = mid - 2 * std
        out["bbands"] = {
            "upper": r4(upper.iloc[-1]),
            "middle": r4(mid.iloc[-1]),
            "lower": r4(lower.iloc[-1]),
            "sigma": 2,
        }

    # KDJ(9,3,3)
    if len(df) >= 9:
        low_n = df["low"].rolling(9).min()
        high_n = df["high"].rolling(9).max()
        rsv = (df["close"] - low_n) / (high_n - low_n) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        out["kdj"] = {
            "k": r4(float(k.iloc[-1])),
            "d": r4(float(d.iloc[-1])),
            "j": r4(float(j.iloc[-1])),
        }

    # MACD(12,26,9)
    if len(df) >= 26:
        fast_ema = ema(df["close"], 12)
        slow_ema = ema(df["close"], 26)
        diff = fast_ema - slow_ema
        dea = diff.ewm(span=9, adjust=False).mean()
        hist = diff - dea
        out["macd"] = {
            "diff": r4(float(diff.iloc[-1])),
            "dea": r4(float(dea.iloc[-1])),
            "hist": r4(float(hist.iloc[-1])),
            "fast": 12,
            "slow": 26,
            "signal": 9
        }

    # last candle
    last = df.iloc[-1].to_dict()
    last["open"] = r4(last["open"])
    last["high"] = r4(last["high"])
    last["low"] = r4(last["low"])
    last["close"] = r4(last["close"])
    last["volume"] = float(f"{last['volume']:.2f}")

    out["last"] = {
        "open": last["open"], "high": last["high"], "low": last["low"], "close": last["close"],
        "volume": last["volume"], "ts": int(last["ts"])
    }
    return out

def make_chart(df: pd.DataFrame,
               symbol: str,
               interval: str,
               limit: int,
               sma20: int = 20,
               sma50: int = 50,
               bbands: int = 20,
               show_kdj: bool = False,
               show_macd: bool = False,
               volume: bool = True) -> BytesIO:
    """
    為了避免安裝額外套件，這裡用「收盤線 + BB + SMA」，
    下方可選擇 KDJ/MACD/成交量 子圖。
    """
    if len(df) < 5:
        raise HTTPException(status_code=400, detail="資料不足，無法作圖")

    close = df["close"].copy()
    idx = pd.to_datetime(df["ts"], unit="ms")

    rows = 1
    if show_kdj:
        rows += 1
    if show_macd:
        rows += 1
    if volume:
        rows += 1

    fig, axes = plt.subplots(rows, 1, figsize=(10, 6 + 1.5*rows), sharex=True)
    if rows == 1:
        axes = [axes]

    ax_price = axes[0]
    ax_price.plot(idx, close, label="Close", linewidth=1.2)

    # SMA
    if len(df) >= sma20:
        ax_price.plot(idx, close.rolling(sma20).mean(), label=f"SMA{sma20}", linewidth=1)
    if len(df) >= sma50:
        ax_price.plot(idx, close.rolling(sma50).mean(), label=f"SMA{sma50}", linewidth=1)

    # BB
    if len(df) >= bbands:
        mid = close.rolling(bbands).mean()
        std = close.rolling(bbands).std(ddof=0)
        up = mid + 2*std
        lo = mid - 2*std
        ax_price.plot(idx, up, linewidth=0.8, label=f"BB+2σ")
        ax_price.plot(idx, mid, linewidth=0.8, label=f"BB mid")
        ax_price.plot(idx, lo, linewidth=0.8, label=f"BB-2σ")

    ax_price.set_title(f"{symbol} {interval} (n={limit})")
    ax_price.legend(loc="upper left")

    row_ptr = 1

    # KDJ
    if show_kdj:
        low_n = df["low"].rolling(9).min()
        high_n = df["high"].rolling(9).max()
        rsv = (close - low_n) / (high_n - low_n) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3*k - 2*d
        ax_kdj = axes[row_ptr]
        ax_kdj.plot(idx, k, label="K")
        ax_kdj.plot(idx, d, label="D")
        ax_kdj.plot(idx, j, label="J")
        ax_kdj.set_ylabel("KDJ")
        ax_kdj.legend(loc="upper left")
        row_ptr += 1

    # MACD
    if show_macd:
        fast_ema = ema(close, 12)
        slow_ema = ema(close, 26)
        diff = fast_ema - slow_ema
        dea = diff.ewm(span=9, adjust=False).mean()
        hist = diff - dea
        ax_macd = axes[row_ptr]
        ax_macd.plot(idx, diff, label="DIFF")
        ax_macd.plot(idx, dea, label="DEA")
        ax_macd.bar(idx, hist, width=0.8, alpha=0.5, label="HIST")
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(loc="upper left")
        row_ptr += 1

    # Volume
    if volume:
        ax_vol = axes[row_ptr]
        ax_vol.bar(idx, df["volume"], width=0.8, alpha=0.5)
        ax_vol.set_ylabel("VOL")

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title=APP_NAME)

@app.get("/")
def root():
    return {"ok": True, "name": APP_NAME}

@app.get("/price")
def api_price(
    symbol: str = Query(..., example="XRP/USDT"),
    source: str = Query("book", pattern="^(book|ticker)$"),
    precision: int = Query(4, ge=0, le=10),
):
    try:
        return fetch_price(symbol, source, precision)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/klines")
def api_klines(
    symbol: str = Query(..., example="XRP/USDT"),
    interval: str = Query("15m", description="5m, 15m, 30m, 1h, 4h, 1d"),
    limit: int = Query(200, ge=5, le=1000),
):
    try:
        df = fetch_klines(symbol, interval, limit)
        rows = [
            {
                "ts": int(r.ts),
                "open": r4(r.open),
                "high": r4(r.high),
                "low": r4(r.low),
                "close": r4(r.close),
                "volume": float(f"{r.volume:.2f}")
            }
            for r in df.itertuples(index=False)
        ]
        return {
            "exchange": "binance",
            "symbol": symbol,
            "interval": interval,
            "rows": rows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indicators")
def api_indicators(
    symbol: str = Query(..., example="XRP/USDT"),
    interval: str = Query("15m"),
    limit: int = Query(200, ge=30, le=1000),
):
    try:
        df = fetch_klines(symbol, interval, limit)
        res = indicators(df)
        return {"exchange": "binance", "symbol": symbol, "interval": interval, **res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chart")
def api_chart(
    symbol: str = Query(..., example="XRP/USDT"),
    interval: str = Query("15m"),
    limit: int = Query(200, ge=30, le=300),
    sma20: int = Query(20, ge=2, le=200),
    sma50: int = Query(50, ge=2, le=300),
    bbands: int = Query(20, ge=5, le=100),
    show_kdj: bool = Query(False),
    show_macd: bool = Query(False),
    volume: bool = Query(True),
):
    try:
        df = fetch_klines(symbol, interval, limit)
        buf = make_chart(df, symbol, interval, limit, sma20, sma50, bbands, show_kdj, show_macd, volume)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# 簡易經驗紀錄（示範）
# ---------------------------
class ExperienceIn(BaseModel):
    symbol: str
    interval: str
    pattern: Optional[str] = None
    outcome: Optional[str] = None  # bullish/bearish/neutral
    notes: Optional[str] = None

EXPERIENCE_LOG: List[Dict[str, Any]] = []

@app.post("/experience/log")
def experience_log(payload: ExperienceIn):
    entry = payload.dict()
    entry["ts"] = int(datetime.now(tz=ZoneInfo("Asia/Taipei")).timestamp() * 1000)
    EXPERIENCE_LOG.append(entry)
    # 只保留最近 500 筆
    if len(EXPERIENCE_LOG) > 500:
        del EXPERIENCE_LOG[:-500]
    return {"ok": True, "saved": entry}

@app.get("/experience/recent")
def experience_recent(limit: int = Query(20, ge=1, le=200)):
    return {"items": EXPERIENCE_LOG[-limit:][::-1]}
