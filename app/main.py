from fastapi import FastAPI, Query, HTTPException
from typing import List, Dict, Any
from .models import ScanResponse, ExplainResponse, Detection, Interval
from .models import ScanRequest
from .datasource import fetch_ohlcv
from .patterns import run_all_patterns
from . import config
import hashlib, time, math

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
