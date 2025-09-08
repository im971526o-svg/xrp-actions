from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

Interval = Literal["15m", "1h", "4h", "1d"]

class Candle(BaseModel):
    t_open: int  # ms
    open: float
    high: float
    low: float
    close: float
    volume: float

class ScanRequest(BaseModel):
    symbol: str = Field("XRP/USDT", description="CCXT symbol")
    intervals: List[Interval] = Field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    lookback: int = Field(200, ge=10, le=1000, description="How many candles to fetch per interval")

class Detection(BaseModel):
    id: str
    symbol: str
    interval: Interval
    pattern_name: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    t_start: int
    t_end: int
    index_range: List[int]
    extra: Dict[str, Any] = {}

class ScanResponse(BaseModel):
    detections: List[Detection]
    meta: Dict[str, Any] = {}
    
class ExplainResponse(BaseModel):
    detection: Detection
    evidence: Dict[str, Any]
    rationale: str
    references: List[Dict[str, str]] = []
