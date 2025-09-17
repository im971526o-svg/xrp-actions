# app/main.py
from fastapi import FastAPI, HTTPException, Header, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, time
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # py>=3.9
    TAIPEI = ZoneInfo("Asia/Taipei")
except Exception:
    TAIPEI = timezone.utc

import ccxt
import pandas as pd

APP_TITLE = "XRP Pattern Scanner Actions"
app = FastAPI(title=APP_TITLE, version="0.2.1", description="Market APIs + settings. Chart endpoints disabled.")

# ------------------------- helpers -------------------------
AUTO_SECRET = os.getenv("AUTO_SECRET", "")
RUNTIME_SETTINGS_PATH = os.getenv("RUNTIME_SETTINGS_PATH", "runtime_settings.json")

DEFAULT_SETTINGS = {
    "trade_mode": "both",          # both | long_only | short_only
    "min_winrate": 0.7,            # 0.0 ~ 1.0
    "auto_enabled": False          # optional: whether auto-trader is enabled
}

def _read_settings() -> dict:
    if not os.path.exists(RUNTIME_SETTINGS_PATH):
        _write_settings(DEFAULT_SETTINGS.copy())
    try:
        with open(RUNTIME_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = DEFAULT_SETTINGS.copy()
    # 補齊缺欄
    for k, v in DEFAULT_SETTINGS.items():
        data.setdefault(k, v)
    return data

def _write_settings(obj: dict):
    with open(RUNTIME_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _require_secret(x_secret: Optional[str]):
    if not AUTO_SECRET:
        return  # 未設定就當作無保護
    if not x_secret or x_secret != AUTO_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")

def _fix_symbol(s: Optional[str]) -> str:
    s = (s or "XRP/USDT").upper().strip()
    if s.endswith("USTD"):
        s = s[:-4] + "USDT"
    return s

def _to_iso_taipei(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(TAIPEI)
    return dt.isoformat()

EXCHANGE = ccxt.binance({"enableRateLimit": True})

# ------------------------- models -------------------------
class AutoEnableRequest(BaseModel):
    enabled: bool

class ExperienceLogRequest(BaseModel):
    symbol: str
    timeframe: str
    pattern: Optional[str] = None
    note: str
    price: Optional[float] = None
    ts_local_taipei: Optional[str] = None

class RuntimeSettingsUpdate(BaseModel):
    trade_mode: Optional[str] = None  # both | long_only | short_only
    min_winrate: Optional[float] = None

# ------------------------- routes: market -------------------------
@app.get("/price", tags=["market"])
def get_price(symbol: Optional[str] = None):
    sym = _fix_symbol(symbol)
    t = EXCHANGE.fetch_ticker(sym)
    return {
        "symbol": sym,
        "price": float(t["last"]),
        "ts_exchange": datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc).isoformat(),
        "ts_local_taipei": _to_iso_taipei(t["timestamp"]),
    }

_ALLOWED = ["5m", "15m", "30m", "1h", "4h", "1d"]

@app.get("/klines", tags=["market"])
def get_klines(interval: str, symbol: Optional[str] = None, limit: int = 200):
    if interval not in _ALLOWED:
        raise HTTPException(400, f"interval must be one of {_ALLOWED}")
    sym = _fix_symbol(symbol)
    limit = max(1, min(int(limit), 1000))
    ohlcv = EXCHANGE.fetch_ohlcv(sym, timeframe=interval, limit=limit)
    candles = []
    for t, o, h, l, c, v in ohlcv:
        candles.append({
            "t": _to_iso_taipei(t),
            "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v)
        })
    return {"symbol": sym, "interval": interval, "candles": candles}

@app.get("/indicators", tags=["market"])
def indicators(interval: str, symbol: Optional[str] = None):
    sym = _fix_symbol(symbol)
    if interval not in _ALLOWED:
        raise HTTPException(400, f"interval must be one of {_ALLOWED}")
    # 取多一點做計算
    ohlcv = EXCHANGE.fetch_ohlcv(sym, timeframe=interval, limit=200)
    df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
    close = df["c"].astype(float)

    # BB(20,2)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    upper = mid + 2*std
    lower = mid - 2*std

    # KDJ(9,3,3)
    low9 = df["l"].rolling(9).min()
    high9 = df["h"].rolling(9).max()
    rsv = (close - low9) / (high9 - low9 + 1e-9) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3*k - 2*d

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    return {
        "symbol": sym,
        "interval": interval,
        "bb": {
            "mid": round(mid.iloc[-1], 4) if not pd.isna(mid.iloc[-1]) else None,
            "upper": round(upper.iloc[-1], 4) if not pd.isna(upper.iloc[-1]) else None,
            "lower": round(lower.iloc[-1], 4) if not pd.isna(lower.iloc[-1]) else None,
        },
        "kdj": {
            "k": round(k.iloc[-1], 4) if not pd.isna(k.iloc[-1]) else None,
            "d": round(d.iloc[-1], 4) if not pd.isna(d.iloc[-1]) else None,
            "j": round(j.iloc[-1], 4) if not pd.isna(j.iloc[-1]) else None,
        },
        "macd": {
            "dif": round(dif.iloc[-1], 4) if not pd.isna(dif.iloc[-1]) else None,
            "dea": round(dea.iloc[-1], 4) if not pd.isna(dea.iloc[-1]) else None,
            "hist": round(hist.iloc[-1], 4) if not pd.isna(hist.iloc[-1]) else None,
        }
    }

# ------------------------- routes: experience -------------------------
@app.post("/experience/log", tags=["experience"])
def experience_log(payload: ExperienceLogRequest):
    # 簡單寫到檔案（避免你沒裝 DB 也能通過）
    rec = payload.dict()
    rec.setdefault("ts_local_taipei", datetime.now(TAIPEI).isoformat())
    with open("experience.log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}

# ------------------------- routes: auto enable -------------------------
@app.post("/auto/enable", tags=["auto"])
def auto_enable(payload: AutoEnableRequest, x_auto_secret: Optional[str] = Header(default=None)):
    _require_secret(x_auto_secret)
    st = _read_settings()
    st["auto_enabled"] = bool(payload.enabled)
    _write_settings(st)
    return {"enabled": st["auto_enabled"]}

# ------------------------- routes: settings -------------------------
@app.get("/settings", tags=["auto"])
def read_settings():
    return _read_settings()

@app.post("/settings", tags=["auto"])
def update_settings(payload: RuntimeSettingsUpdate, x_auto_secret: Optional[str] = Header(default=None)):
    _require_secret(x_auto_secret)
    st = _read_settings()
    if payload.trade_mode:
        if payload.trade_mode not in ("both","long_only","short_only"):
            raise HTTPException(400, "trade_mode must be both|long_only|short_only")
        st["trade_mode"] = payload.trade_mode
    if payload.min_winrate is not None:
        v = float(payload.min_winrate)
        if not (0.0 <= v <= 1.0):
            raise HTTPException(400, "min_winrate must be 0.0~1.0")
        st["min_winrate"] = v
    _write_settings(st)
    return st

# ------------------------- routes: LINE command relay -------------------------
@app.post("/line/command", tags=["line"])
def line_command(body: Dict[str, Any] = Body(...), x_auto_secret: Optional[str] = Header(default=None)):
    _require_secret(x_auto_secret)
    text = (body.get("text") or "").strip().lower()
    if not text:
        return {"ok": False, "message": "empty text"}

    st = _read_settings()
    msg = ""

    if text.startswith("mode"):
        if "long" in text:
            st["trade_mode"] = "long_only"
        elif "short" in text:
            st["trade_mode"] = "short_only"
        else:
            st["trade_mode"] = "both"
        msg = f"updated trade_mode={st['trade_mode']}"
    elif text.startswith("winrate"):
        # 支援 winrate 70 / 0.7 / 80%
        val = text.split(" ",1)[-1].replace("%","").strip()
        try:
            v = float(val)
            if v > 1: v = v/100.0
            v = max(0.0, min(1.0, v))
            st["min_winrate"] = v
            msg = f"updated min_winrate={v}"
        except Exception:
            raise HTTPException(400, "bad winrate value")
    else:
        msg = "noop: supported commands: mode long|short|both; winrate <num|percent>"

    _write_settings(st)
    return {"ok": True, "message": msg}

# ------------------------- routes: chart disabled -------------------------
@app.get("/chart")
@app.get("/chart/quick")
def chart_disabled():
    raise HTTPException(status_code=501, detail="chart endpoints are disabled by design")
