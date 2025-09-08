# ── 必須在 pyplot 之前設定無頭後端 ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # 讓伺服器無圖形介面也能畫圖

# ── 標準匯入 ────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Literal
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO
import time

APP_NAME = "XRP Pattern Scanner Actions"

app = FastAPI(title=APP_NAME)

# ── 交易所初始化（幣安 / 四位小數一致化）───────────────────────────────────────
def _ex_binance():
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": False,  # 關閉時間差校正，較穩
        },
    })
    if "adjustForTimeDifference" not in ex.options:
        ex.options["adjustForTimeDifference"] = False
    return ex

def r4(x: float) -> float:
    """四位小數"""
    return float(f"{float(x):.4f}")

ALLOWED_INTERVALS = {"15m", "1h", "4h", "1d"}

# ── 共同資料處理 ─────────────────────────────────────────────────────────────
def fetch_ohlcv_df(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(400, f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
    if limit < 10 or limit > 1000:
        raise HTTPException(400, "limit must be between 10 and 1000")

    ex = _ex_binance()
    try:
        rows = ex.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    except Exception as e:
        raise HTTPException(502, f"failed to fetch_ohlcv: {e}")

    if not rows:
        raise HTTPException(404, "no klines")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    # 四位小數（量能保留原值）
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float).map(r4)
    df.set_index("ts", inplace=True)
    return df

def parse_int_list(csv: str) -> List[int]:
    return [int(x) for x in csv.split(",") if str(x).strip().isdigit()]

# ── 指標（BB、KDJ、MACD）────────────────────────────────────────────────────
def compute_bbands(df: pd.DataFrame, period: int = 20, sigma: float = 2.0) -> pd.DataFrame:
    mid = df["close"].rolling(period, min_periods=period).mean()
    std = df["close"].rolling(period, min_periods=period).std(ddof=0)
    upper = mid + sigma * std
    lower = mid - sigma * std
    out = pd.DataFrame({
        "bb_mid": mid.map(r4),
        "bb_upper": upper.map(r4),
        "bb_lower": lower.map(r4),
    }, index=df.index)
    return out

def compute_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    low_n = df["low"].rolling(n, min_periods=n).min()
    high_n = df["high"].rolling(n, min_periods=n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100.0
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3*k - 2*d
    return pd.DataFrame({"K": k.map(r4), "D": d.map(r4), "J": j.map(r4)}, index=df.index)

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({
        "DIF": dif.map(r4),
        "DEA": dea.map(r4),
        "HIST": hist.map(r4)
    }, index=df.index)

# ── 回應模型（可選，讓 /docs 更清楚）──────────────────────────────────────────
class PriceResponse(BaseModel):
    exchange: str
    symbol: str
    price: float
    ts: int

class KlineRow(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class KlinesResponse(BaseModel):
    exchange: str
    symbol: str
    interval: str
    rows: List[KlineRow]

class IndicatorsResponse(BaseModel):
    exchange: str
    symbol: str
    interval: str
    last: dict

# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/price", response_model=PriceResponse)
def price(symbol: str = "XRP/USDT"):
    ex = _ex_binance()
    try:
        ticker = ex.fetch_ticker(symbol)
    except Exception as e:
        raise HTTPException(502, f"failed to fetch_ticker: {e}")
    last = r4(ticker.get("last") or ticker.get("close") or 0.0)
    return {
        "exchange": "binance",
        "symbol": symbol,
        "price": last,
        "ts": int(time.time()*1000),
    }

@app.get("/klines", response_model=KlinesResponse)
def klines(
    symbol: str = "XRP/USDT",
    interval: Literal["15m", "1h", "4h", "1d"] = "15m",
    limit: int = Query(200, ge=10, le=1000),
):
    df = fetch_ohlcv_df(symbol, interval, limit)
    rows = [
        KlineRow(
            ts=int(idx.value/1e6),
            open=r4(row.open), high=r4(row.high),
            low=r4(row.low), close=r4(row.close),
            volume=float(row.volume),
        )
        for idx, row in df.iterrows()
    ]
    return {
        "exchange": "binance",
        "symbol": symbol,
        "interval": interval,
        "rows": rows,
    }

@app.get("/indicators", response_model=IndicatorsResponse)
def indicators(
    symbol: str = "XRP/USDT",
    interval: Literal["15m", "1h", "4h", "1d"] = "15m",
    limit: int = Query(200, ge=50, le=1000),
):
    df = fetch_ohlcv_df(symbol, interval, limit)
    bb = compute_bbands(df, period=20, sigma=2.0)
    kdj = compute_kdj(df, n=9, m1=3, m2=3)
    macd = compute_macd(df, fast=12, slow=26, signal=9)

    last_ts = df.index[-1]
    out = {
        "exchange": "binance",
        "symbol": symbol,
        "interval": interval,
        "last": {
            "ts": int(last_ts.value/1e6),
            "open": r4(df["open"].iloc[-1]),
            "high": r4(df["high"].iloc[-1]),
            "low": r4(df["low"].iloc[-1]),
            "close": r4(df["close"].iloc[-1]),
            "volume": float(df["volume"].iloc[-1]),
            "bbands": {
                "upper": r4(bb["bb_upper"].iloc[-1]),
                "middle": r4(bb["bb_mid"].iloc[-1]),
                "lower": r4(bb["bb_lower"].iloc[-1]),
                "sigma": 2,
            },
            "kdj": {
                "k": r4(kdj["K"].iloc[-1]),
                "d": r4(kdj["D"].iloc[-1]),
                "j": r4(kdj["J"].iloc[-1]),
                "n": 9, "m1": 3, "m2": 3,
            },
            "macd": {
                "dif": r4(macd["DIF"].iloc[-1]),
                "dea": r4(macd["DEA"].iloc[-1]),
                "hist": r4(macd["HIST"].iloc[-1]),
                "fast": 12, "slow": 26, "signal": 9,
            },
        }
    }
    return out

@app.get("/chart")
def chart(
    symbol: str = "XRP/USDT",
    interval: Literal["15m", "1h", "4h", "1d"] = "15m",
    limit: int = Query(300, ge=50, le=1000),
    sma: str = "20,50",
    bbands: str = "20,2",
    show_kdj: bool = True,
    show_macd: bool = True,
):
    df = fetch_ohlcv_df(symbol, interval, limit)

    # ── addplot 設定 ────────────────────────────────────────────────────────
    aps = []

    # SMA
    sma_periods = parse_int_list(sma)
    for p in sma_periods:
        sma_series = df["close"].rolling(p, min_periods=p).mean().map(r4)
        aps.append(mpf.make_addplot(sma_series, panel=0, width=1.0))

    # BBands
    bb = parse_int_list(bbands)
    if len(bb) >= 1:
        bb_period = bb[0]
        bb_sigma = float(bb[1]) if len(bb) >= 2 else 2.0
        bbdf = compute_bbands(df, period=bb_period, sigma=bb_sigma)
        aps.append(mpf.make_addplot(bbdf["bb_upper"], panel=0, width=0.8))
        aps.append(mpf.make_addplot(bbdf["bb_lower"], panel=0, width=0.8))

    # KDJ panel
    if show_kdj:
        kdj = compute_kdj(df, n=9, m1=3, m2=3)
        aps.append(mpf.make_addplot(kdj["K"], panel=1, width=0.8))
        aps.append(mpf.make_addplot(kdj["D"], panel=1, width=0.8))
        aps.append(mpf.make_addplot(kdj["J"], panel=1, width=0.8))

    # MACD panel
    if show_macd:
        macd = compute_macd(df, fast=12, slow=26, signal=9)
        aps.append(mpf.make_addplot(macd["HIST"], panel=2, type="bar", alpha=0.5))
        aps.append(mpf.make_addplot(macd["DIF"], panel=2, width=0.8))
        aps.append(mpf.make_addplot(macd["DEA"], panel=2, width=0.8))

    # ── 繪圖 & 回傳 PNG（確保關圖：避免記憶體累積）────────────────────────────
    fig = None
    try:
        fig, axlist = mpf.plot(
            df,
            type="candle",
            style="yahoo",
            addplot=aps,
            volume=False,               # 如要顯示量能可改 True（會多一個 panel）
            returnfig=True,             # 取得 fig 才能關掉
            figscale=1.1,
            figratio=(16, 9),
            panel_ratios=(3, 1, 1) if (show_kdj and show_macd) else (3, 1),
            tight_layout=True,
            datetime_format="%Y-%m-%d\n%H:%M"
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"plot failed: {e}")
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")

@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "endpoints": ["/price", "/klines", "/indicators", "/chart"]}
