from fastapi import FastAPI, Query, HTTPException
from typing import List, Dict, Any
from .models import ScanResponse, ExplainResponse, Detection, Interval
from .models import ScanRequest
from .datasource import fetch_ohlcv
from .patterns import run_all_patterns
from . import config
import hashlib, time, math
from fastapi import Query
from fastapi.responses import StreamingResponse
from io import BytesIO
import time
import ccxt
import pandas as pd
import numpy as np
import mplfinance as mpf

app = FastAPI(title=config.APP_NAME)

# Simple in-memory store for last scan (for demo)
DETECTIONS: Dict[str, Detection] = {}

def mk_id(symbol: str, interval: str, idx: int, name: str, t: int) -> str:
    s = f"{symbol}|{interval}|{idx}|{name}|{t}"
    return hashlib.md5(s.encode()).hexdigest()

@app.get("/scan", response_model=ScanResponse)
def scan(symbol: str="XRP/USDT",
         intervals: str="15m,1h,4h,1d",
         lookback: int=200):
    try:
        intervals_list: List[Interval] = [i.strip() for i in intervals.split(",") if i.strip()]
    except Exception:
        raise HTTPException(400, "Invalid intervals")
    if lookback < 10 or lookback > 1000:
        raise HTTPException(400, "lookback must be in [10,1000]")

    detections = []
    meta = {"symbol": symbol, "intervals": intervals_list, "lookback": lookback, "ts": int(time.time()*1000)}
    for itv in intervals_list:
        candles = fetch_ohlcv(symbol, itv, lookback)
        hits = run_all_patterns(candles)
        for (idx, direction, conf, extra) in hits[-50:]:  # cap results
            t_start = candles[max(0, idx-2)].t_open
            t_end = candles[idx].t_open
            pid = mk_id(symbol, itv, idx, extra.get("type","pattern"), t_end)
            det = Detection(
                id=pid,
                symbol=symbol,
                interval=itv, 
                pattern_name=extra.get("type","pattern"),
                direction=direction,
                confidence=float(round(min(1.0, max(0.0, conf)), 3)),
                t_start=t_start,
                t_end=t_end,
                index_range=[max(0, idx-2), idx],
                extra=extra
            )
            DETECTIONS[pid] = det
            detections.append(det)
    # sort by latest first
    detections.sort(key=lambda d: d.t_end, reverse=True)
    return ScanResponse(detections=detections[:200], meta=meta)

@app.get("/explain", response_model=ExplainResponse)
def explain(pattern_id: str):
    det = DETECTIONS.get(pattern_id)
    if not det:
        raise HTTPException(404, "pattern_id not found; call /scan first or use a fresh id")
    # For a compact demo, refetch the window around the detection to provide evidence
    lookback = 120
    candles = fetch_ohlcv(det.symbol, det.interval, lookback)
    start_i, end_i = det.index_range[0], det.index_range[1]
    # slice a small window
    window = candles[max(0, end_i-10): end_i+1]
    evidence = {
        "window_candles": [c.model_dump() for c in window],
        "interval": det.interval,
        "symbol": det.symbol,
        "computed_features": det.extra,
    }
    rationale = (
        f"This looks like a {det.pattern_name.replace('_',' ')} pattern on {det.symbol} "
        f"({det.interval}). Confidence {det.confidence:.2f}. "
        "Use it with broader context (trend, S/R, volume) and risk controls."
    )
    # references: map your books/knowledge here by pattern name
    references = [
        {"title": "Your Book: Candlestick Basics", "section": det.pattern_name, "note": "Map this to your Knowledge entries."}
    ]
    return ExplainResponse(detection=det, evidence=evidence, rationale=rationale, references=references)
# ===== Binance helpers & indicators （貼在檔案最底部）=====
def _ex_binance():
    ex = ccxt.binance()
    ex.options = {"defaultType": "spot"}
    return ex

def _r4(x: float) -> float:
    return float(f"{float(x):.4f}")

def _to_df(ohlcv):
    import pandas as pd
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float).map(_r4)
    df["volume"] = df["volume"].astype(float)
    return df

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    dif = _ema(close, fast) - _ema(close, slow)
    dea = _ema(dif, signal)
    hist = dif - dea
    return dif, dea, hist

def _kdj(df: pd.DataFrame, n=9, m1=3, m2=3):
    low_n  = df["low"].rolling(n, min_periods=n).min()
    high_n = df["high"].rolling(n, min_periods=n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100.0
    rsv = rsv.fillna(50.0)
    k = rsv.ewm(alpha=1.0/m1, adjust=False).mean()
    d = k.ewm(alpha=1.0/m2, adjust=False).mean()
    j = 3*k - 2*d
    return k, d, j

@app.get("/price")
def price(symbol: str = "XRP/USDT"):
    """幣安即時價格（小數 4 位）"""
    ex = _ex_binance()
    t = ex.fetch_ticker(symbol)
    return {"exchange": "binance", "symbol": symbol, "price": _r4(t["last"]), "ts": int(time.time()*1000)}

@app.get("/klines")
def klines(symbol: str = "XRP/USDT", interval: str = "15m", limit: int = 200):
    """幣安 OHLCV（舊→新），所有 OHLC 四位小數"""
    ex = _ex_binance()
    raw = ex.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    df = _to_df(raw)
    rows = [{"ts": int(idx.value//10**6),
             "open": _r4(r.open), "high": _r4(r.high),
             "low": _r4(r.low), "close": _r4(r.close),
             "volume": float(r.volume)} for idx, r in df.iterrows()]
    return {"exchange":"binance","symbol":symbol,"interval":interval,"rows":rows}

@app.get("/chart", responses={200: {"content": {"image/png": {}}}})
def chart(symbol: str = "XRP/USDT", interval: str = "15m", limit: int = 300,
          sma: str = "20,50", bbands: str = "20,2",
          show_kdj: bool = True, show_macd: bool = True,
          width: int = 1200, height: int = 900):
    """K 線 PNG：含 SMA、布林(20,2)、KDJ、MACD"""
    ex = _ex_binance()
    raw = ex.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    df = _to_df(raw)

    addplots = []
    title = f"{symbol} {interval} (last {limit})"

    # SMA
    if sma:
        try:
            periods = [int(x) for x in sma.split(",") if x.strip()]
            for p in periods:
                df[f"SMA{p}"] = df["close"].rolling(p).mean().map(_r4)
                addplots.append(mpf.make_addplot(df[f"SMA{p}"], color=None))
            title += f" | SMA {periods}"
        except:
            pass

    # BB(20,2)
    if bbands:
        try:
            n, s = [int(x) for x in bbands.split(",")]
            mid = df["close"].rolling(n).mean()
            std = df["close"].rolling(n).std(ddof=0)
            up = (mid + s*std).map(_r4)
            lo = (mid - s*std).map(_r4)
            addplots += [mpf.make_addplot(up, color=None),
                         mpf.make_addplot(lo, color=None)]
            title += f" | BB({n},{s})"
        except:
            pass

    # KDJ (panel 2)
    if show_kdj:
        k, d, j = _kdj(df)
        addplots += [
            mpf.make_addplot(k, panel=2, color=None, ylabel="KDJ"),
            mpf.make_addplot(d, panel=2, color=None),
            mpf.make_addplot(j, panel=2, color=None),
        ]
        title += " | KDJ(9,3,3)"

    # MACD (panel 3)
    if show_macd:
        dif, dea, hist = _macd(df["close"])
        addplots += [
            mpf.make_addplot(dif, panel=3, color=None, ylabel="MACD"),
            mpf.make_addplot(dea, panel=3, color=None),
            mpf.make_addplot(hist, panel=3, type="bar", alpha=0.5),
        ]
        title += " | MACD(12,26,9)"

    buf = BytesIO()
    mpf.plot(df, type="candle", style="charles", volume=True,
             addplot=addplots, title=title,
             figsize=(width/100, height/100), tight_layout=True,
             savefig=dict(fname=buf, dpi=110, bbox_inches="tight"))
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/indicators")
def indicators(symbol: str = "XRP/USDT", interval: str = "15m", limit: int = 200):
    """回傳最新一根K + BB/KDJ/MACD（全部四位小數）"""
    ex = _ex_binance()
    raw = ex.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    df = _to_df(raw)

    last = df.iloc[-1]
    o,h,l,c,v = _r4(last.open), _r4(last.high), _r4(last.low), _r4(last.close), float(last.volume)

    # BB(20,2)
    n, s = 20, 2
    mid = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std(ddof=0)
    bb_u = _r4((mid + s*std).iloc[-1]); bb_m = _r4(mid.iloc[-1]); bb_l = _r4((mid - s*std).iloc[-1])

    # KDJ
    k,d,j = _kdj(df)
    k, d, j = _r4(k.iloc[-1]), _r4(d.iloc[-1]), _r4(j.iloc[-1])

    # MACD
    dif, dea, hist = _macd(df["close"])
    dif, dea, hist = _r4(dif.iloc[-1]), _r4(dea.iloc[-1]), _r4(hist.iloc[-1])

    return {
        "exchange":"binance","symbol":symbol,"interval":interval,
        "last":{"ts": int(df.index[-1].value//10**6), "open":o,"high":h,"low":l,"close":c,"volume":v},
        "bbands":{"upper":bb_u,"middle":bb_m,"lower":bb_l,"n":n,"sigma":s},
        "kdj":{"k":k,"d":d,"j":j,"n":9,"m1":3,"m2":3},
        "macd":{"dif":dif,"dea":dea,"hist":hist,"fast":12,"slow":26,"signal":9}
    }
# ===== end =====
