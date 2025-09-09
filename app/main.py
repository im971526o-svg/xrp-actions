# -*- coding: utf-8 -*-
"""
XRP Pattern Scanner & Chart API (FastAPI on Render)

功能總覽
- /                  : 健康檢查
- /price             : 現價（precision、ticker/orderbook，time_local=台北時）
- /klines            : OHLCV 5m/15m/30m/1h/4h/1d
- /indicators        : BB(20,2)、KDJ(9,3,3)、MACD(12,26,9)
- /chart             : 圖（實心蠟燭+影線+成交量+KDJ+MACD），圖片壓到 ~150KB
- /scan              : 《蠟燭圖精解》型態偵測（hammer/射擊之星/十字/吞噬/三白兵/三黑鴉）
- /experience/*      : 經驗紀錄（DB 持久化）
- /chat/*            : 聊天紀錄（DB 持久化）
- /analysis/run      : 一次跑多 timeframe 的分析 → 自動比對前次 → 自動寫入 DB
- /analysis/recent   : 查最近分析
- /auto/analyze      : 排程入口（預設 15m；建議配 Render Cron Job），可驗證 token

資料庫
- 預設 SQLite（檔名 data.db，服務重啟仍在；部署更新會清空）
- 若設 DATABASE_URL（Postgres），則用 Postgres 永久保存（跨部署不掉）
"""

import io
import os
import json
import time
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.responses import JSONResponse, Response

# 交易所
import ccxt

# DB: SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, BigInteger, String, Text, Float, DateTime, JSON as SA_JSON, select, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

APP_NAME   = "XRP Pattern Scanner Actions"
DEFAULT_TZ = "Asia/Taipei"
CRON_SECRET = os.getenv("CRON_SECRET")  # 建議設定，給 /auto/analyze 驗證用

# -------------------- DB 初始化 --------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db")
engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatLog(Base):
    __tablename__ = "chat_logs"
    id      = Column(Integer, primary_key=True, autoincrement=True)
    ts_ms   = Column(BigInteger, index=True)
    role    = Column(String(16))
    content = Column(Text)
    symbol  = Column(String(32), nullable=True)
    interval= Column(String(16), nullable=True)
    tags    = Column(Text, nullable=True)  # JSON 字串

class ExperienceLog(Base):
    __tablename__ = "experience_logs"
    id       = Column(Integer, primary_key=True, autoincrement=True)
    ts_ms    = Column(BigInteger, index=True)
    symbol   = Column(String(32))
    interval = Column(String(16))
    pattern  = Column(String(64))
    outcome  = Column(String(16))
    notes    = Column(Text)

class AnalysisLog(Base):
    __tablename__ = "analysis_logs"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    ts_ms      = Column(BigInteger, index=True)
    symbol     = Column(String(32))
    interval   = Column(String(16))
    result_json= Column(Text)   # JSON 字串（指標、型態、重點）
    delta_json = Column(Text)   # 與前一筆比對的變化（JSON 字串）
    summary    = Column(Text)   # 簡述（便於列表瀏覽）

Base.metadata.create_all(bind=engine)

# -------------------- 工具函式 --------------------
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo  # type: ignore

def local_ts_str(ts_ms: int, tz: str = DEFAULT_TZ) -> str:
    dt = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(ZoneInfo(tz))  # type: ignore
    hour = int(dt.strftime("%I"))
    minute = int(dt.strftime("%M"))
    ampm = "上午" if dt.strftime("%p") == "AM" else "下午"
    return f"{ampm} {hour} 點 {minute} 分"

def round_price(x: float, precision: int = 4) -> float:
    q = Decimal(str(x)).quantize(Decimal("1." + "0" * precision), rounding=ROUND_HALF_UP)
    return float(q)

def make_exchange() -> ccxt.binance:
    ex = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 15000,
    })
    ex.options["adjustForTimeDifference"] = False
    return ex

def timeframe_ok(interval: str) -> str:
    ok = {"5m", "15m", "30m", "1h", "4h", "1d"}
    if interval not in ok:
        raise HTTPException(400, f"interval must be one of {sorted(ok)}")
    return interval

def fetch_ohlcv_df(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    ex = make_exchange()
    tf = timeframe_ok(interval)
    data = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    if not data:
        raise HTTPException(502, "Empty OHLCV from exchange.")
    arr = np.array(data, dtype=float)
    df = pd.DataFrame(arr, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = df["ts"].astype(np.int64)
    return df

# 指標
def sma(series: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(series); return s.rolling(n, min_periods=1).mean().to_numpy()

def ema(series: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(series); return s.ewm(span=n, adjust=False).mean().to_numpy()

def calc_bbands(close: np.ndarray, n: int = 20, sigma: float = 2.0):
    mid = sma(close, n)
    std = pd.Series(close).rolling(n, min_periods=1).std(ddof=0).to_numpy()
    upper = mid + sigma * std; lower = mid - sigma * std
    return upper, mid, lower

def calc_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 9, m1: int = 3, m2: int = 3):
    hh = pd.Series(high).rolling(n, min_periods=1).max().to_numpy()
    ll = pd.Series(low).rolling(n, min_periods=1).min().to_numpy()
    rsv = np.where(hh - ll == 0, 0, (close - ll) / (hh - ll) * 100.0)
    k = pd.Series(rsv).ewm(alpha=1 / m1, adjust=False).mean().to_numpy()
    d = pd.Series(k).ewm(alpha=1 / m2, adjust=False).mean().to_numpy()
    j = 3 * k - 2 * d
    return k, d, j

def calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast); ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().to_numpy()
    hist = dif - dea
    return dif, dea, hist

# 型態規則（《蠟燭圖精解》精神）
def _body(o, c):  return abs(c - o)
def _upper(o, c, h): return h - max(o, c)
def _lower(o, c, l): return min(o, c) - l
def _is_bull(o, c): return c >= o
def _is_bear(o, c): return c <  o

def _atr_like(h, l, n=14):
    rng = (h - l)
    s = pd.Series(rng).rolling(n, min_periods=1).mean().to_numpy()
    s[s == 0] = 1e-9
    return s

def is_hammer(o, h, l, c, atr, strict=False):
    b = _body(o, c); up = _upper(o, c, h); dn = _lower(o, c, l)
    cond = (dn >= (2.5 if strict else 2.0)*b) and (up <= 0.3*b) and (b <= 0.6*atr)
    score = (dn/b if b>0 else 4.0) + (0.3*b/max(atr,1e-9))
    return cond, float(min(score/5.0, 1.0))

def is_shooting_star(o, h, l, c, atr, strict=False):
    b = _body(o, c); up = _upper(o, c, h); dn = _lower(o, c, l)
    cond = (up >= (2.5 if strict else 2.0)*b) and (dn <= 0.3*b) and (b <= 0.6*atr)
    score = (up/b if b>0 else 4.0)
    return cond, float(min(score/5.0, 1.0))

def is_doji(o, h, l, c, atr, strict=False):
    b = _body(o, c); rng = h - l
    cond = (b <= (0.08 if strict else 0.12)*rng)
    score = float(max(0.2, 1.0 - b/max(rng,1e-9)))
    return cond, score

def is_bullish_engulfing(prev, cur, atr, strict=False):
    o1,h1,l1,c1 = prev; o2,h2,l2,c2 = cur
    cond = _is_bear(o1,c1) and _is_bull(o2,c2) and (o2 <= c1) and (c2 >= o1)
    if strict: cond = cond and (_body(o2,c2) >= max(_body(o1,c1), 0.5*atr))
    score = float(min(_body(o2,c2)/max(_body(o1,c1),1e-9), 2.0)/2.0)
    return cond, score

def is_bearish_engulfing(prev, cur, atr, strict=False):
    o1,h1,l1,c1 = prev; o2,h2,l2,c2 = cur
    cond = _is_bull(o1,c1) and _is_bear(o2,c2) and (o2 >= c1) and (c2 <= o1)
    if strict: cond = cond and (_body(o2,c2) >= max(_body(o1,c1), 0.5*atr))
    score = float(min(_body(o2,c2)/max(_body(o1,c1),1e-9), 2.0)/2.0)
    return cond, score

def is_three_white_soldiers(trio, atr, strict=False):
    if len(trio) < 3: return False, 0.0
    ok = True; bonus = 0.0
    for i in range(3):
        o,h,l,c = trio[i]
        if not _is_bull(o,c): ok=False; break
        if _upper(o,c,h) > (0.4 if strict else 0.6)*_body(o,c): ok=False; break
        bonus += _body(o,c)/max(atr,1e-9)
        if i>0:
            o0,h0,l0,c0 = trio[i-1]
            if not (o >= o0 and c >= c0): ok=False; break
    return ok, float(min(bonus/3.0, 1.0))

def is_three_black_crows(trio, atr, strict=False):
    if len(trio) < 3: return False, 0.0
    ok = True; bonus = 0.0
    for i in range(3):
        o,h,l,c = trio[i]
        if not _is_bear(o,c): ok=False; break
        if _lower(o,c,l) > (0.4 if strict else 0.6)*_body(o,c): ok=False; break
        bonus += _body(o,c)/max(atr,1e-9)
        if i>0:
            o0,h0,l0,c0 = trio[i-1]
            if not (o <= o0 and c <= c0): ok=False; break
    return ok, float(min(bonus/3.0, 1.0))

def scan_patterns_df(df: pd.DataFrame, strict=False):
    out=[]
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    o = df["open"].to_numpy(float); c = df["close"].to_numpy(float)
    atr = _atr_like(h, l, n=14)
    n = len(df)
    for i in range(n):
        cur = (o[i], h[i], l[i], c[i])
        ok, sc = is_hammer(*cur, atr[i], strict);         if ok: out.append({"idx": i, "type": "hammer", "score": sc})
        ok, sc = is_shooting_star(*cur, atr[i], strict);  if ok: out.append({"idx": i, "type": "shooting_star", "score": sc})
        ok, sc = is_doji(*cur, atr[i], strict);           if ok: out.append({"idx": i, "type": "doji", "score": sc})
        if i>0:
            prev = (o[i-1], h[i-1], l[i-1], c[i-1])
            ok, sc = is_bullish_engulfing(prev, cur, atr[i], strict);  if ok: out.append({"idx": i, "type": "bullish_engulfing", "score": sc})
            ok, sc = is_bearish_engulfing(prev, cur, atr[i], strict);  if ok: out.append({"idx": i, "type": "bearish_engulfing", "score": sc})
        if i>=2:
            trio = [(o[j],h[j],l[j],c[j]) for j in range(i-2, i+1)]
            ok, sc = is_three_white_soldiers(trio, atr[i], strict); if ok: out.append({"idx": i, "type": "three_white_soldiers", "score": sc})
            ok, sc = is_three_black_crows(trio, atr[i], strict);   if ok: out.append({"idx": i, "type": "three_black_crows", "score": sc})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# 蠟燭繪圖
def draw_candles(ax, df, up="#e74c3c", down="#2ecc71", width=0.7):
    x = np.arange(len(df))
    opens  = df["open"].to_numpy(float)
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    for i in range(len(df)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color = up if c >= o else down
        ax.vlines(i, l, h, color=color, linewidth=1.0, zorder=2)
        lower  = min(o, c)
        height = max(o, c) - lower
        if height == 0: height = max((h - l) * 0.03, 1e-9)
        rect = plt.Rectangle((i - width/2, lower), width, height,
                             facecolor=color, edgecolor=color, linewidth=0.8, zorder=3)
        ax.add_patch(rect)
    ax.set_xlim(0, len(df) - 1)
    ax.set_ylim(df["low"].min()*0.995, df["high"].max()*1.005)

def annotate_patterns(ax_price, detections, df):
    y = df["close"].to_numpy(float)
    for det in detections:
        i   = int(det["idx"]); t = det["type"]; sc = det["score"]
        m   = "^" if ("bull" in t or "white" in t or t=="hammer") else "v"
        col = "#e74c3c" if m == "^" else "#2ecc71"
        ax_price.scatter(i, y[i], marker=m, s=60, color=col, zorder=4)
        ax_price.text(i, y[i], f"{t}\n{sc:.2f}", fontsize=8, ha="center", va="bottom", color=col)

def _ensure_size(buf_png: bytes, target_kb: int = 150, to_format: str = "auto") -> Tuple[bytes, str]:
    kb = len(buf_png) / 1024
    if kb <= target_kb and to_format not in ("jpg", "jpeg"):
        return buf_png, "png"
    im = Image.open(io.BytesIO(buf_png)).convert("RGB")
    lo, hi, best = 35, 95, None
    while lo <= hi:
        q = (lo + hi) // 2
        tmp = io.BytesIO()
        im.save(tmp, format="JPEG", quality=int(q), optimize=True)
        size_kb = tmp.tell() / 1024
        if size_kb <= target_kb:
            best = tmp.getvalue(); lo = q + 1
        else:
            hi = q - 1
    if best is None:
        tmp = io.BytesIO(); im.save(tmp, format="JPEG", quality=35, optimize=True)
        best = tmp.getvalue()
    return best, "jpeg"

# -------------------- FastAPI --------------------
app = FastAPI(title=APP_NAME)

@app.get("/")
def root():
    return {"ok": True, "app": APP_NAME,
            "routes": ["/price","/klines","/indicators","/chart","/scan",
                       "/experience/log (POST)","/experience/recent (GET)",
                       "/chat/log (POST)","/chat/recent (GET)",
                       "/analysis/run (POST)","/analysis/recent (GET)",
                       "/auto/analyze (GET)"]}

# 現價
@app.get("/price")
def get_price(
    symbol: str = Query(..., example="XRP/USDT"),
    precision: int = Query(4, ge=0, le=10),
    source: Literal["ticker","orderbook"] = Query("ticker"),
    adjust: Optional[bool] = Query(None),
    tz: str = Query(DEFAULT_TZ)
):
    ex = make_exchange()
    if adjust is not None:
        ex.options["adjustForTimeDifference"] = bool(adjust)
    if source == "orderbook":
        ob = ex.fetch_order_book(symbol, limit=5)
        best_bid = ob["bids"][0][0] if ob["bids"] else None
        best_ask = ob["asks"][0][0] if ob["asks"] else None
        if best_bid is None or best_ask is None:
            raise HTTPException(502, "Empty orderbook.")
        price = (best_bid + best_ask) / 2.0
        ts_ms = int(time.time() * 1000)
    else:
        t = ex.fetch_ticker(symbol); price = t["last"]; ts_ms = int(t["timestamp"] or time.time()*1000)
    return {"exchange":"binance","symbol":symbol,"price":round_price(float(price),precision),
            "ts":ts_ms,"time_local":local_ts_str(ts_ms, tz)}

# K 線
@app.get("/klines")
def get_klines(symbol: str = Query(...), interval: str = Query("15m"), limit: int = Query(200, ge=50, le=1000)):
    df = fetch_ohlcv_df(symbol, interval, limit)
    rows = [{"ts":int(r.ts),"open":float(r.open),"high":float(r.high),"low":float(r.low),
             "close":float(r.close),"volume":float(r.volume)} for r in df.itertuples(index=False)]
    return {"exchange":"binance","symbol":symbol,"interval":interval,"rows":rows}

# 指標
@app.get("/indicators")
def get_indicators(
    symbol: str = Query(...), interval: str = Query("15m"), limit: int = Query(200, ge=50, le=1000),
    bbands: str = Query("20,2"), show_kdj: bool = Query(True), show_macd: bool = Query(True)
):
    df = fetch_ohlcv_df(symbol, interval, limit)
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    try:
        n_s,s_s = bbands.split(","); n, s = int(n_s), float(s_s)
    except Exception:
        raise HTTPException(400, "bbands must be 'period,sigma', e.g., 20,2")
    up, mid, lo = calc_bbands(close, n=n, sigma=s)
    out: Dict[str,Any] = {
        "exchange":"binance","symbol":symbol,"interval":interval,
        "last":{"open":float(df["open"].iloc[-1]),"high":float(df["high"].iloc[-1]),
                "low":float(df["low"].iloc[-1]),"close":float(df["close"].iloc[-1]),
                "volume":float(df["volume"].iloc[-1])},
        "bbands":{"upper":float(up[-1]),"middle":float(mid[-1]),"lower":float(lo[-1]),"sigma":s}
    }
    if show_kdj:
        k,d,j = calc_kdj(high, low, close); out["kdj"]={"k":float(k[-1]),"d":float(d[-1]),"j":float(j[-1])}
    if show_macd:
        dif,dea,hist = calc_macd(close); out["macd"]={"dif":float(dif[-1]),"dea":float(dea[-1]),"hist":float(hist[-1]),
                                                      "fast":12,"slow":26,"signal":9}
    return out

# 圖（含圖片壓縮到 ~150KB）
@app.get("/chart")
def get_chart(
    symbol: str = Query(...), interval: str = Query("15m"), limit: int = Query(300, ge=100, le=1000),
    sma20: Optional[int] = Query(20), bbands: Optional[int] = Query(20), bb_sigma: float = Query(2.0),
    show_kdj: bool = Query(True), show_macd: bool = Query(True), volume: bool = Query(True),
    annotate: bool = Query(True), strict: bool = Query(False),
    max_kb: int = Query(150, ge=60, le=600), img_format: Literal["auto","png","jpg","jpeg"] = Query("auto"),
    width: float = Query(10.0), height: float = Query(5.0), dpi: int = Query(120, ge=70, le=200)
):
    df = fetch_ohlcv_df(symbol, interval, limit)
    # 佈局比例
    h_price = 0.62; h_vol = 0.12 if volume else 0.0; h_kdj = 0.13 if show_kdj else 0.0; h_macd = 0.13 if show_macd else 0.0
    n_axes = 1 + int(volume) + int(show_kdj) + int(show_macd)
    ratios = [h_price] + ([h_vol] if volume else []) + ([h_kdj] if show_kdj else []) + ([h_macd] if show_macd else [])
    fig, axes = plt.subplots(nrows=n_axes, ncols=1, figsize=(width, height),
                             gridspec_kw={"height_ratios": ratios if ratios else [1.0]})
    if not isinstance(axes, np.ndarray): axes = np.array([axes])
    idx=0; ax_price = axes[idx]; idx+=1
    # 蠟燭
    draw_candles(ax_price, df); ax_price.set_title(f"{symbol}  {interval}")
    # SMA/BB
    if sma20:
        ax_price.plot(sma(df["close"].to_numpy(float), sma20), linewidth=1.05, color="#34495e", label=f"SMA{sma20}")
    if bbands:
        up, mid, lo = calc_bbands(df["close"].to_numpy(float), n=bbands, sigma=bb_sigma)
        ax_price.plot(up,linewidth=0.9,color="#8e44ad"); ax_price.plot(mid,linewidth=0.9,color="#8e44ad",linestyle="--"); ax_price.plot(lo,linewidth=0.9,color="#8e44ad")
    # 成交量
    if volume:
        ax_vol = axes[idx]; idx+=1
        opens = df["open"].to_numpy(float); closes=df["close"].to_numpy(float)
        colors = np.where(closes>=opens, "#e74c3c", "#2ecc71")
        ax_vol.bar(np.arange(len(df)), df["volume"].to_numpy(float), color=colors, width=0.7)
        ax_vol.set_xlim(0,len(df)-1); ax_vol.set_ylabel("Vol")
    # KDJ
    if show_kdj:
        ax_kdj=axes[idx]; idx+=1
        k,d,j = calc_kdj(df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float))
        ax_kdj.plot(k,label="K"); ax_kdj.plot(d,label="D"); ax_kdj.plot(j,label="J")
        ax_kdj.axhline(80,color="#7f8c8d",linewidth=0.8,linestyle="--"); ax_kdj.axhline(20,color="#7f8c8d",linewidth=0.8,linestyle="--")
        ax_kdj.legend(loc="upper left",fontsize=8)
    # MACD
    if show_macd:
        ax_macd=axes[idx]; idx+=1
        dif,dea,hist=calc_macd(df["close"].to_numpy(float))
        ax_macd.plot(dif,label="DIF"); ax_macd.plot(dea,label="DEA")
        colors=np.where(hist>=0,"#e74c3c","#2ecc71")
        ax_macd.bar(np.arange(len(hist)), hist, color=colors, width=0.7, alpha=0.6, label="HIST")
        ax_macd.legend(loc="upper left",fontsize=8)
    # 標註型態
    if annotate:
        dets = scan_patterns_df(df, strict=strict)
        annotate_patterns(ax_price, dets, df)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi); plt.close(fig)
    png_bytes = buf.getvalue()
    if img_format=="png" and len(png_bytes)/1024 <= max_kb:
        return Response(content=png_bytes, media_type="image/png")
    out_bytes, sub = _ensure_size(png_bytes, target_kb=max_kb, to_format=img_format)
    return Response(content=out_bytes, media_type=f"image/{sub}")

# 型態掃描（JSON）
@app.get("/scan")
def scan_endpoint(symbol: str = Query(...), interval: str = Query("15m"), limit: int = Query(200, ge=50, le=1000), strict: bool = Query(False)):
    df = fetch_ohlcv_df(symbol, interval, limit)
    dets = scan_patterns_df(df, strict=strict)
    return {"exchange":"binance","symbol":symbol,"interval":interval,"detections":dets}

# 經驗紀錄（DB）
@app.post("/experience/log")
def post_experience_log(
    symbol: str = Body(...), interval: str = Body(...), pattern: str = Body(...), outcome: str = Body(...), notes: str = Body("")
, db: Session = Depends(get_db)):
    ts_ms = int(time.time()*1000)
    row = ExperienceLog(ts_ms=ts_ms, symbol=symbol, interval=interval, pattern=pattern, outcome=outcome, notes=notes)
    db.add(row); db.commit()
    return {"ok": True, "saved": {"id": row.id, "ts": ts_ms}}

@app.get("/experience/recent")
def get_experience_recent(limit: int = Query(20, ge=1, le=200), symbol: Optional[str] = Query(None), interval: Optional[str] = Query(None), db: Session = Depends(get_db)):
    q = db.query(ExperienceLog).order_by(desc(ExperienceLog.id))
    if symbol:  q = q.filter(ExperienceLog.symbol==symbol)
    if interval:q = q.filter(ExperienceLog.interval==interval)
    rows = q.limit(limit).all()
    return {"items":[{"id":r.id,"ts":r.ts_ms,"symbol":r.symbol,"interval":r.interval,"pattern":r.pattern,"outcome":r.outcome,"notes":r.notes} for r in rows]}

# 聊天紀錄（DB）
@app.post("/chat/log")
def post_chat_log(
    role: Literal["user","assistant","system"] = Body("user"),
    content: str = Body(...),
    symbol: Optional[str] = Body(None),
    interval: Optional[str] = Body(None),
    tags: Optional[List[str]] = Body(default=None),
    db: Session = Depends(get_db)
):
    ts_ms = int(time.time()*1000)
    row = ChatLog(ts_ms=ts_ms, role=role, content=content, symbol=symbol, interval=interval, tags=json.dumps(tags or []))
    db.add(row); db.commit()
    return {"ok": True, "saved": {"id": row.id, "ts": ts_ms}}

@app.get("/chat/recent")
def get_chat_recent(limit: int = Query(50, ge=1, le=500), symbol: Optional[str] = Query(None), interval: Optional[str] = Query(None), role: Optional[Literal["user","assistant","system"]] = Query(None), db: Session = Depends(get_db)):
    q = db.query(ChatLog).order_by(desc(ChatLog.id))
    if symbol:   q = q.filter(ChatLog.symbol==symbol)
    if interval: q = q.filter(ChatLog.interval==interval)
    if role:     q = q.filter(ChatLog.role==role)
    rows = q.limit(limit).all()
    return {"items":[{"id":r.id,"ts":r.ts_ms,"role":r.role,"content":r.content,"symbol":r.symbol,"interval":r.interval,"tags":json.loads(r.tags or "[]")} for r in rows]}

# ---- 分析與自動存檔 ----
def _analyze_one(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    k,d,j = calc_kdj(high, low, close); dif,dea,hist = calc_macd(close)
    up, mid, lo = calc_bbands(close, n=20, sigma=2.0)
    # 位置判斷
    pos = "below_lower" if close[-1] < lo[-1] else ("above_upper" if close[-1] > up[-1] else "in_band")
    # 交叉
    kdj_cross = "golden" if (k[-2] <= d[-2] and k[-1] > d[-1]) else ("dead" if (k[-2] >= d[-2] and k[-1] < d[-1]) else "none")
    macd_cross = "golden" if (dif[-2] <= dea[-2] and dif[-1] > dea[-1]) else ("dead" if (dif[-2] >= dea[-2] and dif[-1] < dea[-1]) else "none")
    # 型態（近 3 根）
    dets = scan_patterns_df(df.tail(60).reset_index(drop=True), strict=False)
    return {
        "last":{"open":float(df["open"].iloc[-1]),"high":float(df["high"].iloc[-1]),"low":float(df["low"].iloc[-1]),"close":float(close[-1]),"volume":float(df["volume"].iloc[-1])},
        "bbands":{"upper":float(up[-1]),"middle":float(mid[-1]),"lower":float(lo[-1]),"pos":pos},
        "kdj":{"k":float(k[-1]),"d":float(d[-1]),"j":float(j[-1]),"cross":kdj_cross},
        "macd":{"dif":float(dif[-1]),"dea":float(dea[-1]),"hist":float(hist[-1]),"cross":macd_cross},
        "patterns": dets[:10]
    }

def _delta(prev: Dict[str,Any], cur: Dict[str,Any]) -> Dict[str,Any]:
    out={}
    try:
        out["close_change"] = round(float(cur["last"]["close"] - prev["last"]["close"]), 6)
        out["kdj_cross_change"] = f'{prev["kdj"]["cross"]} -> {cur["kdj"]["cross"]}'
        out["macd_cross_change"]= f'{prev["macd"]["cross"]} -> {cur["macd"]["cross"]}'
        out["bb_pos_change"]    = f'{prev["bbands"]["pos"]} -> {cur["bbands"]["pos"]}'
    except Exception:
        pass
    return out

@app.post("/analysis/run")
def analysis_run(
    symbol: str = Body("XRP/USDT"),
    intervals: List[str] = Body(["15m","30m","1h","4h","1d"]),
    limit: int = Body(200),
    save: bool = Body(True),
    compare_prev: bool = Body(True),
    db: Session = Depends(get_db)
):
    results=[]; ts_ms = int(time.time()*1000)
    for interval in intervals:
        df = fetch_ohlcv_df(symbol, interval, limit)
        cur = _analyze_one(df)

        delta = {}
        if compare_prev:
            prev_row = db.query(AnalysisLog).filter(AnalysisLog.symbol==symbol, AnalysisLog.interval==interval).order_by(desc(AnalysisLog.id)).first()
            if prev_row:
                prev = json.loads(prev_row.result_json)
                delta = _delta(prev, cur)

        summary = f"{symbol} {interval}: close={cur['last']['close']:.4f}, BB={cur['bbands']['pos']}, KDJ={cur['kdj']['cross']}, MACD={cur['macd']['cross']}"
        results.append({"symbol":symbol,"interval":interval,"ts":ts_ms,"result":cur,"delta":delta,"summary":summary})

        if save:
            row = AnalysisLog(ts_ms=ts_ms, symbol=symbol, interval=interval,
                              result_json=json.dumps(cur, ensure_ascii=False),
                              delta_json=json.dumps(delta, ensure_ascii=False),
                              summary=summary)
            db.add(row)
    if save:
        db.commit()
    return {"ok": True, "items": results}

@app.get("/analysis/recent")
def analysis_recent(symbol: str = Query("XRP/USDT"), interval: Optional[str] = Query(None), limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)):
    q = db.query(AnalysisLog).filter(AnalysisLog.symbol==symbol)
    if interval: q = q.filter(AnalysisLog.interval==interval)
    rows = q.order_by(desc(AnalysisLog.id)).limit(limit).all()
    return {"items":[{"id":r.id,"ts":r.ts_ms,"symbol":r.symbol,"interval":r.interval,"summary":r.summary,"result":json.loads(r.result_json),"delta":json.loads(r.delta_json or "{}")} for r in rows]}

@app.get("/auto/analyze")
def auto_analyze(symbol: str = Query("XRP/USDT"), interval: str = Query("15m"), token: Optional[str] = Query(None), db: Session = Depends(get_db)):
    if CRON_SECRET and token != CRON_SECRET:
        raise HTTPException(401, "Invalid token")
    # 跑單框架，儲存並比對
    payload = {"symbol":symbol, "intervals":[interval], "limit":200, "save":True, "compare_prev":True}
    # 直接呼叫內部函式
    return analysis_run(symbol=symbol, intervals=[interval], limit=200, save=True, compare_prev=True, db=db)

# 本機啟動
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
