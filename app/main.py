# app/main.py
import os
import io
import time
import math
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 無頭環境
import matplotlib.pyplot as plt

from PIL import Image

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

import ccxt

APP_NAME = os.getenv("APP_NAME", "XRP Pattern Scanner Actions")
TZ_NAME = os.getenv("TZ_NAME", "Asia/Taipei")
SYMBOL_DEFAULT = "XRP/USDT"

# ===== FastAPI =====
app = FastAPI(title=APP_NAME)

# ===== DB (SQLite 檔案，免費 Render 也能用；若要永久化，改用外部 Postgres) =====
DB_PATH = os.getenv("SQLITE_PATH", "/tmp/xrp_actions.db")

def db_init():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS experience_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        pattern TEXT,
        outcome TEXT,
        notes TEXT
    );
    """)
    conn.commit()
    conn.close()

db_init()

def db_insert(ts: int, symbol: str, interval: str, pattern: str, outcome: str, notes: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO experience_log (ts, symbol, interval, pattern, outcome, notes) VALUES (?, ?, ?, ?, ?, ?)",
        (ts, symbol, interval, pattern, outcome, notes)
    )
    conn.commit()
    conn.close()

def db_recent(limit: int = 20) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ts, symbol, interval, pattern, outcome, notes FROM experience_log ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    res = []
    for r in rows:
        res.append({
            "ts": r[0],
            "symbol": r[1],
            "interval": r[2],
            "pattern": r[3],
            "outcome": r[4],
            "notes": r[5],
        })
    return res

# ===== Helper：時間、四捨五入、圖片壓縮 =====
def local_str(ts_ms: int, tz_name: str = TZ_NAME) -> str:
    # pandas 讓我們不加額外套件也能轉 tz
    dt = pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_convert(tz_name)
    hh = int(dt.strftime("%I"))  # 12 小時制，前導零去掉
    mm = int(dt.strftime("%M"))
    ap = "上午" if dt.strftime("%p") == "AM" else "下午"
    return f"{ap} {hh} 點 {mm} 分"

def round4(x: float) -> float:
    return float(f"{x:.4f}")

def save_png_under(fig, target_kb: int = 150) -> bytes:
    """將 Matplotlib figure 轉 PNG 並嘗試壓到 ~target_kb。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    data = buf.getvalue()
    buf.close()

    if len(data) <= target_kb * 1024:
        return data

    # 用 PIL 量化壓小
    img = Image.open(io.BytesIO(data)).convert("RGB")
    for colors in (128, 96, 64, 32):
        out = io.BytesIO()
        img_p = img.convert("P", palette=Image.ADAPTIVE, colors=colors)
        img_p.save(out, format="PNG", optimize=True)
        d = out.getvalue()
        out.close()
        if len(d) <= target_kb * 1024:
            return d
    return d  # 實在壓不下就回最後一次

# ===== 交易所：Binance（ccxt）=====
def binance():
    ex = ccxt.binance({"enableRateLimit": True})
    # 避免時間差錯誤
    ex.options["adjustForTimeDifference"] = True
    return ex

# ===== 技術指標 =====
def calc_bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower

def calc_kdj(df: pd.DataFrame, n: int = 9):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

# ===== 取 K 線 =====
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    ex = binance()
    ms = ex.parse_timeframe(interval) * 1000
    now = int(time.time() * 1000)
    since = now - ms * (limit + 5)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=limit)
    cols = ["ts", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv, columns=cols)
    return df

# ====== Endpoints ======
@app.get("/")
def root():
    return {"ok": True}

@app.get("/price")
def get_price(
    symbol: str = Query(SYMBOL_DEFAULT, example="XRP/USDT"),
    precision: int = Query(4, ge=0, le=10),
    source: str = Query("ticker", regex="^(ticker|book)$", description="ticker 或 book")
):
    ex = binance()

    if source == "book":
        ob = ex.fetch_order_book(symbol)
        if not ob["bids"] or not ob["asks"]:
            raise HTTPException(503, "orderbook not ready")
        price = (ob["bids"][0][0] + ob["asks"][0][0]) / 2
    else:
        t = ex.fetch_ticker(symbol)
        price = t["last"] or t["close"] or t["bid"] or t["ask"]

    price = float(f"{price:.{precision}f}")
    ts = int(time.time() * 1000)
    return {
        "exchange": "binance",
        "symbol": symbol,
        "price": price,
        "ts": ts,
        "time_local": local_str(ts),
        "precision": precision
    }

@app.get("/klines")
def get_klines(
    symbol: str = Query(SYMBOL_DEFAULT),
    interval: str = Query("15m", regex="^(5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(200, ge=50, le=500)
):
    df = fetch_klines(symbol, interval, limit)
    rows = df.to_dict(orient="records")
    return {"exchange": "binance", "symbol": symbol, "interval": interval, "rows": rows}

@app.get("/indicators")
def get_indicators(
    symbol: str = Query(SYMBOL_DEFAULT),
    interval: str = Query("15m", regex="^(5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(200, ge=50, le=500)
):
    df = fetch_klines(symbol, interval, limit)
    close = df["close"]
    k, d, j = calc_kdj(df, 9)
    dif, dea, hist = calc_macd(close, 12, 26, 9)
    ma, upper, lower = calc_bbands(close, 20, 2.0)

    last = {
        "ts": int(df["ts"].iloc[-1]),
        "open": round4(df["open"].iloc[-1]),
        "high": round4(df["high"].iloc[-1]),
        "low": round4(df["low"].iloc[-1]),
        "close": round4(df["close"].iloc[-1]),
        "volume": float(df["volume"].iloc[-1])
    }
    out = {
        "exchange": "binance",
        "symbol": symbol,
        "interval": interval,
        "last": last,
        "bbands": {
            "upper": round4(upper.iloc[-1]),
            "middle": round4(ma.iloc[-1]),
            "lower": round4(lower.iloc[-1]),
            "sigma": 2
        },
        "kdj": {
            "k": float(f"{k.iloc[-1]:.2f}"),
            "d": float(f"{d.iloc[-1]:.2f}"),
            "j": float(f"{j.iloc[-1]:.2f}")
        },
        "macd": {
            "dif": float(f"{dif.iloc[-1]:.4f}"),
            "dea": float(f"{dea.iloc[-1]:.4f}"),
            "hist": float(f"{hist.iloc[-1]:.4f}")
        }
    }
    return out

@app.get("/chart")
def get_chart(
    symbol: str = Query(SYMBOL_DEFAULT),
    interval: str = Query("15m", regex="^(5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(300, ge=100, le=500),
    sma: int = Query(20, ge=2, le=200),
    bbands: int = Query(20, ge=5, le=100),
    show_kdj: bool = Query(True),
    show_macd: bool = Query(True),
    volume: bool = Query(True)
):
    df = fetch_klines(symbol, interval, limit)
    close = df["close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma, upper, lower = calc_bbands(close, bbands, 2.0)

    k, d, j = calc_kdj(df, 9)
    dif, dea, hist = calc_macd(close, 12, 26, 9)

    # figure 佈局：上面主圖；下面兩圖（可選）
    rows = 1 + int(show_kdj) + int(show_macd)
    height = 2.8 * rows
    fig, axes = plt.subplots(rows, 1, figsize=(10, height), sharex=True,
                             gridspec_kw={"height_ratios": [2] + [1]*(rows-1)})

    if rows == 1:
        ax = axes
        sub_axes = []
    else:
        ax = axes[0]
        sub_axes = axes[1:]

    x = np.arange(len(df))
    # K 線（以蠟燭條顯示）
    up = df["close"] >= df["open"]
    down = ~up
    # 上漲
    ax.bar(x[up], (df["close"] - df["open"])[up], bottom=df["open"][up], color="#26a69a", width=0.6)
    ax.vlines(x[up], df["low"][up], df["high"][up], color="#26a69a", linewidth=1)
    # 下跌
    ax.bar(x[down], (df["close"] - df["open"])[down], bottom=df["open"][down], color="#ef5350", width=0.6)
    ax.vlines(x[down], df["low"][down], df["high"][down], color="#ef5350", linewidth=1)

    # SMA 與 BB
    ax.plot(x, ma20, linewidth=1, label="SMA20")
    ax.plot(x, ma50, linewidth=1, label="SMA50")
    ax.plot(x, upper, linewidth=1, linestyle="--", label="BB upper")
    ax.plot(x, ma, linewidth=1, linestyle="--", label="BB mid")
    ax.plot(x, lower, linewidth=1, linestyle="--", label="BB lower")

    if volume:
        vol_ax = ax.twinx()
        vol_ax.bar(x, df["volume"], alpha=0.2, width=0.6, color="#90caf9")
        vol_ax.set_yticklabels([])

    ax.set_title(f"{symbol}  {interval}  ({limit} bars)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    idx = 0
    if show_kdj and idx < len(sub_axes):
        ax2 = sub_axes[idx]; idx += 1
        ax2.plot(x, k, label="K")
        ax2.plot(x, d, label="D")
        ax2.plot(x, j, label="J")
        ax2.legend(loc="upper left"); ax2.grid(alpha=0.2)

    if show_macd and idx < len(sub_axes):
        ax3 = sub_axes[idx]; idx += 1
        ax3.plot(x, dif, label="DIF")
        ax3.plot(x, dea, label="DEA")
        ax3.bar(x, hist, label="HIST", color=np.where(hist >= 0, "#26a69a", "#ef5350"), alpha=0.4)
        ax3.legend(loc="upper left"); ax3.grid(alpha=0.2)

    plt.tight_layout()
    data = save_png_under(fig, target_kb=150)
    plt.close(fig)
    return StreamingResponse(io.BytesIO(data), media_type="image/png")

# ===== 經驗紀錄 =====
@app.post("/experience/log")
def post_experience(payload: Dict[str, Any]):
    """
    {
      "symbol": "XRP/USDT",
      "interval": "15m",
      "pattern": "three_white_soldiers",
      "outcome": "bullish",
      "notes": "收斂箱體突破後..."
    }
    """
    symbol = payload.get("symbol", SYMBOL_DEFAULT)
    interval = payload.get("interval", "15m")
    pattern = payload.get("pattern", "")
    outcome = payload.get("outcome", "")
    notes = payload.get("notes", "")

    ts = int(time.time() * 1000)
    db_insert(ts, symbol, interval, pattern, outcome, notes)
    return {"ok": True, "saved": True, "ts": ts, "time_local": local_str(ts)}

@app.get("/experience/recent")
def get_experience_recent(n: int = Query(20, ge=1, le=200)):
    rows = db_recent(n)
    return {"ok": True, "items": rows}

# ===== 可選：每 15 分鐘自動分析（預設關閉）。設環境變數 AUTO_ANALYZE=true 啟用 =====
_auto_started = False

def _auto_worker():
    while True:
        try:
            df = fetch_klines(SYMBOL_DEFAULT, "15m", 120)
            # 簡單範例：連續 3 根紅棒（three white soldiers）偵測
            closes = df["close"].values
            opens = df["open"].values
            cond = (closes[-1] > opens[-1]) and (closes[-2] > opens[-2]) and (closes[-3] > opens[-3]) \
                   and (closes[-1] > closes[-2] > closes[-3])
            if cond:
                db_insert(int(time.time()*1000), SYMBOL_DEFAULT, "15m", "three_white_soldiers", "bullish", "auto")
        except Exception as e:
            print("[auto] error:", e)
        # 15 分鐘跑一次
        time.sleep(900)

def maybe_start_auto():
    global _auto_started
    if _auto_started:
        return
    if os.getenv("AUTO_ANALYZE", "false").lower() == "true":
        t = threading.Thread(target=_auto_worker, daemon=True)
        t.start()
        _auto_started = True

maybe_start_auto()
