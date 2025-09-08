import ccxt
from typing import List, Tuple
from .models import Candle

# Map our intervals to exchange timeframe strings
TIMEFRAMES = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

def fetch_ohlcv(symbol: str, interval: str, lookback: int) -> List[Candle]:
    ex = ccxt.binance({"enableRateLimit": True})
    tf = TIMEFRAMES[interval]
    raw = ex.fetch_ohlcv(symbol, timeframe=tf, limit=lookback)  # [ts, open, high, low, close, volume]
    candles = [Candle(t_open=r[0], open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5]) for r in raw]
    return candles
