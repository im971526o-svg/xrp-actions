from typing import List, Tuple, Dict
import math
from .models import Candle

def body(c: Candle) -> float:
    return abs(c.close - c.open)

def upper_wick(c: Candle) -> float:
    return c.high - max(c.open, c.close)

def lower_wick(c: Candle) -> float:
    return min(c.open, c.close) - c.low

def is_bull(c: Candle) -> bool:
    return c.close > c.open

def is_bear(c: Candle) -> bool:
    return c.open > c.close

def pct(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b

def detect_engulfing(candles: List[Candle]) -> List[Tuple[int, str, float, Dict]]:
    """Return list of tuples (index, direction, confidence, extra)"""
    out = []
    for i in range(1, len(candles)):
        prev, cur = candles[i-1], candles[i]
        prev_body = body(prev)
        cur_body = body(cur)
        total_range = max(prev.high, cur.high) - min(prev.low, cur.low)
        if prev_body == 0 or cur_body == 0 or total_range == 0:
            continue
        # Bullish engulfing: prev bear, current bull; current body engulfs prev body
        if is_bear(prev) and is_bull(cur) and (min(cur.open, cur.close) <= min(prev.open, prev.close)) and (max(cur.open, cur.close) >= max(prev.open, prev.close)):
            conf = 0.5 + 0.5 * pct(cur_body, total_range)
            out.append((i, "bullish", conf, {"type":"engulfing"}))
        # Bearish engulfing
        if is_bull(prev) and is_bear(cur) and (min(cur.open, cur.close) <= min(prev.open, prev.close)) and (max(cur.open, cur.close) >= max(prev.open, prev.close)):
            conf = 0.5 + 0.5 * pct(cur_body, total_range)
            out.append((i, "bearish", conf, {"type":"engulfing"}))
    return out

def detect_hammer_shootingstar(candles: List[Candle]) -> List[Tuple[int, str, float, Dict]]:
    out = []
    for i in range(len(candles)):
        c = candles[i]
        rng = c.high - c.low
        b = body(c)
        uw, lw = upper_wick(c), lower_wick(c)
        if rng == 0:
            continue
        # Hammer: long lower wick >= 2x body, small upper wick
        if lw >= 2*b and uw <= b*0.5:
            direction = "bullish"
            conf = min(1.0, 0.4 + 0.6 * (lw / (b+1e-9)))
            out.append((i, direction, conf, {"type":"hammer"}))
        # Shooting star: long upper wick >= 2x body, small lower wick
        if uw >= 2*b and lw <= b*0.5:
            direction = "bearish"
            conf = min(1.0, 0.4 + 0.6 * (uw / (b+1e-9)))
            out.append((i, direction, conf, {"type":"shooting_star"}))
    return out

def detect_inside_outside(candles: List[Candle]) -> List[Tuple[int, str, float, Dict]]:
    out = []
    for i in range(1, len(candles)):
        a, b = candles[i-1], candles[i]
        # Inside bar: b within a's high-low
        if b.high <= a.high and b.low >= a.low:
            out.append((i, "neutral", 0.55, {"type":"inside"}))
        # Outside bar: b high> a.high and b.low < a.low
        if b.high >= a.high and b.low <= a.low:
            out.append((i, "neutral", 0.55, {"type":"outside"}))
    return out

def detect_three_soldiers_crows(candles: List[Candle]) -> List[Tuple[int, str, float, Dict]]:
    out = []
    for i in range(2, len(candles)):
        a,b,c = candles[i-2], candles[i-1], candles[i]
        if is_bull(a) and is_bull(b) and is_bull(c) and a.close < b.close < c.close:
            out.append((i, "bullish", 0.7, {"type":"three_white_soldiers"}))
        if is_bear(a) and is_bear(b) and is_bear(c) and a.close > b.close > c.close:
            out.append((i, "bearish", 0.7, {"type":"three_black_crows"}))
    return out

PATTERN_FUNCS = {
    "engulfing": detect_engulfing,
    "hammer_star": detect_hammer_shootingstar,
    "inside_outside": detect_inside_outside,
    "soldiers_crows": detect_three_soldiers_crows,
}

def run_all_patterns(candles: List[Candle]) -> List[Tuple[int, str, float, Dict]]:
    all_hits = []
    for name, fn in PATTERN_FUNCS.items():
        hits = fn(candles)
        for (idx, direction, conf, extra) in hits:
            extra = {"rule": name, **extra}
            all_hits.append((idx, direction, conf, extra))
    return all_hits
