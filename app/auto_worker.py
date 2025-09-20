# app/auto_worker.py  (patched)
import os
import time
import math
import json
import ccxt
import pandas as pd
import urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta

# ---- 解析時間字串成秒數（支援 1m/30s/0.5h/數字秒） ----
def _parse_seconds(text: str) -> int:
    if text is None:
        return 900
    t = str(text).strip().lower()
    try:
        if t.endswith("ms"):  # 毫秒
            return max(1, int(float(t[:-2]) / 1000))
        if t.endswith("s"):
            return int(float(t[:-1]))
        if t.endswith("m"):
            return int(float(t[:-1]) * 60)
        if t.endswith("h"):
            return int(float(t[:-1]) * 3600)
        return int(float(t))
    except Exception:
        return 900  # 預設 15 分鐘

# ---------- 環境變數 ----------
# 支援 SYMBOLS 或舊的單一 SYMBOL
_symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "XRP/USDT")
SYMBOLS = [s.strip() for s in _symbols_env.split(",") if s.strip()]

INTERVAL = os.getenv("INTERVAL", "15m")
INTERVAL_SEC = _parse_seconds(INTERVAL)

# 優先讀 LOOP_SLEEP_SEC，其次相容舊變數 LOOP_SEC；都沒設就落回 INTERVAL 秒數
_sleep_env = os.getenv("LOOP_SLEEP_SEC") or os.getenv("LOOP_SEC")
LOOP_SEC = _parse_seconds(_sleep_env) if _sleep_env else INTERVAL_SEC

DRY_RUN   = os.getenv("DRY_RUN", "true").lower() == "true"
TESTNET   = os.getenv("TESTNET", "true").lower() == "true"
EXCHANGE  = os.getenv("EXCHANGE", "binance")
EXCHANGE_LOWER = EXCHANGE.lower()

# 單筆以 quote 幣的金額（USDT）下單；正式上線前務必調整成交易所允許的最小單位
QUOTE_SIZE = float(os.getenv("TRADE_SIZE_USDT", "10"))

BINANCE_KEY    = os.getenv("BINANCE_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

# 勝率門檻：環境變數 MIN_WINRATE 優先；否則讀設定檔；預設 0.6
MIN_WINRATE_ENV = os.getenv("MIN_WINRATE")

# 設定檔
SETTINGS_PATH = os.getenv("RUNTIME_SETTINGS_PATH", "runtime_settings.json")

# LINE Notify
LINE_TOKEN = os.getenv("LINE_NOTIFY_TOKEN", "")
LINE_NOTIFY_ON = os.getenv("LINE_NOTIFY_ON", "orders").lower()  # orders | all | off

TZ = timezone(timedelta(hours=8))  # Asia/Taipei for logs

def log(msg, **kw):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", (kw if kw else ""))

def notify(msg: str):
    if not LINE_TOKEN or LINE_NOTIFY_ON == "off":
        return
    data = urllib.parse.urlencode({"message": msg}).encode("utf-8")
    req = urllib.request.Request(
        "https://notify-api.line.me/api/notify",
        data=data,
        headers={
            "Authorization": f"Bearer {LINE_TOKEN}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
    except Exception as e:
        log(f"LINE notify error: {repr(e)}")

# ---------- 交易所連線 ----------
def build_exchange():
    if EXCHANGE_LOWER != "binance":
        raise RuntimeError(f"Unsupported EXCHANGE: {EXCHANGE}")

    ex = ccxt.binance({
        "apiKey": BINANCE_KEY,
        "secret": BINANCE_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
    })

    # 測試網開關（ccxt sandbox）
    ex.set_sandbox_mode(TESTNET)
    return ex

# ---------- 指標 ----------
def indicators(df: pd.DataFrame):
    # 需要欄位: open high low close volume ts(毫秒)
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    # SMA(20)
    sma = close.rolling(20).mean()

    # BB(20, 2)
    std = close.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    # KDJ(9,3,3)
    low_n  = low.rolling(9).min()
    high_n = high.rolling(9).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(alpha=1/3).mean()
    d = k.ewm(alpha=1/3).mean()
    j = 3 * k - 2 * d

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    return pd.DataFrame(
        {
            "sma20": sma,
            "bb_upper": upper,
            "bb_lower": lower,
            "k": k, "d": d, "j": j,
            "macd_dif": dif, "macd_dea": dea, "macd_hist": hist,
        },
        index=df.index,
    )

# ---------- 取 K 線 ----------
def fetch_klines(ex, symbol: str, timeframe: str, limit: int = 200):
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])

# ---------- 設定讀取 ----------
def _load_min_winrate(default_val=0.6):
    try:
        if MIN_WINRATE_ENV is not None:
            return float(MIN_WINRATE_ENV)
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            return float(d.get("min_winrate", default_val))
    except Exception:
        pass
    return default_val

def _load_trade_mode(default_val="both"):
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            mode = str(d.get("trade_mode", default_val)).lower()
            if mode in ("both","long_only","short_only"):
                return mode
    except Exception:
        pass
    return default_val

# ---------- 很保守的示範訊號 ----------
# MACD 由負翻正 & K > D & 收在 SMA20 上 → buy
# MACD 由正翻負 & K < D & 收在 SMA20 下 → sell
def simple_signal(df: pd.DataFrame, ind: pd.DataFrame):
    """
    Return (side, conf, info)
    side: "buy" | "sell" | None
    conf: 0.0 ~ 1.0, as ratio of bullish/bearish conditions satisfied (3 conditions)
    info: dict with values and booleans
    """
    if len(df) < 30:
        return None, 0.0, {}

    c  = float(df["close"].iloc[-1])
    sma  = float(ind["sma20"].iloc[-1])
    dif0 = float(ind["macd_dif"].iloc[-1])
    dea0 = float(ind["macd_dea"].iloc[-1])
    dif1 = float(ind["macd_dif"].iloc[-2])
    dea1 = float(ind["macd_dea"].iloc[-2])
    k0 = float(ind["k"].iloc[-1])
    d0 = float(ind["d"].iloc[-1])

    bull = {
        "cross_up":     (dif1 <= dea1) and (dif0 >  dea0),
        "k_gt_d":       (k0 > d0),
        "close_gt_sma": (c > sma),
    }
    bear = {
        "cross_down":   (dif1 >= dea1) and (dif0 <  dea0),
        "k_lt_d":       (k0 < d0),
        "close_lt_sma": (c < sma),
    }
    bull_score = sum(1 for v in bull.values() if v)
    bear_score = sum(1 for v in bear.values() if v)

    if bull_score == 0 and bear_score == 0:
        return None, 0.0, {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0}

    if bull_score >= bear_score:
        conf = bull_score / 3.0
        return "buy", conf, {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0, **bull}
    else:
        conf = bear_score / 3.0
        return "sell", conf, {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0, **bear}

# ---------- 下單（示範；正式上線前請驗證最小單位/精度） ----------
def place_order(ex, symbol: str, side: str, quote_amount: float):
    ticker = ex.fetch_ticker(symbol)
    price = float(ticker["last"] or ticker["close"])
    amount = quote_amount / price

    # 依精度把數量往下切
    m = ex.load_markets()[symbol]
    prec_amt = m["precision"].get("amount")
    if prec_amt is not None:
        step = 10 ** (-prec_amt)
        amount = int(amount / step) * step

    log(f"PLACE ORDER (demo) {side} {symbol} amount={amount:.8f} (≈{quote_amount} quote)")
    if DRY_RUN:
        return {"dry_run": True, "side": side, "symbol": symbol, "amount": amount, "price": price}

    return ex.create_market_buy_order(symbol, amount) if side == "buy" else ex.create_market_sell_order(symbol, amount)

def loop():
    ex = build_exchange()
    log(f"Worker started | EXCHANGE={EXCHANGE} TESTNET={TESTNET} INTERVAL={INTERVAL} DRY_RUN={DRY_RUN} SYMBOLS={SYMBOLS}")
    notify(f"Worker started | INTERVAL={INTERVAL} DRY_RUN={DRY_RUN} LOOP_SEC={LOOP_SEC}")

    while True:
        try:
            trade_mode = _load_trade_mode("both")
            min_wr = _load_min_winrate(0.6)

            for symbol in SYMBOLS:
                try:
                    df = fetch_klines(ex, symbol, INTERVAL, limit=200)
                    ind = indicators(df)
                    side, conf, ctx = simple_signal(df, ind)
                    last_ts = int(df["ts"].iloc[-1])
                    t_local = datetime.fromtimestamp(last_ts/1000, TZ).strftime("%H:%M")

                    if not side:
                        log(f"[{symbol}] no-signal at {t_local} | close={float(df['close'].iloc[-1]):.6f}")
                        if LINE_NOTIFY_ON == "all":
                            notify(f"【{symbol}】NO SIGNAL {t_local} (interval={INTERVAL})")
                        continue

                    # trade mode gating
                    if trade_mode == "long_only" and side == "sell":
                        log(f"[{symbol}] SIGNAL={side} conf={conf:.2f} at {t_local} blocked by mode=long_only", **ctx)
                        continue
                    if trade_mode == "short_only" and side == "buy":
                        log(f"[{symbol}] SIGNAL={side} conf={conf:.2f} at {t_local} blocked by mode=short_only", **ctx)
                        continue

                    # confidence / winrate gating
                    if conf < min_wr:
                        log(f"[{symbol}] SIGNAL={side} conf={conf:.2f} < min_winrate={min_wr:.2f}, skip at {t_local}", **ctx)
                        continue

                    log(f"[{symbol}] EXECUTE SIGNAL={side} conf={conf:.2f} at {t_local}", **ctx)
                    res = place_order(ex, symbol, side, QUOTE_SIZE)
                    log(f"[{symbol}] RESULT: {json.dumps(res, ensure_ascii=False)}")
                    try:
                        notify(f"【{symbol}】{side.upper()} {t_local} conf={conf:.2f}\nsize={QUOTE_SIZE} USDT\ninterval={INTERVAL}")
                    except Exception:
                        pass
                except Exception as e:
                    log(f"[{symbol}] error: {repr(e)}")
                    notify(f"[{symbol}] error: {repr(e)}")
            log(f"sleep {LOOP_SEC}s ...")
            time.sleep(LOOP_SEC)
        except KeyboardInterrupt:
            log("worker stopped by KeyboardInterrupt")
            notify("Worker stopped by KeyboardInterrupt")
            break
        except Exception as e:
            log(f"loop error: {repr(e)}")
            notify(f"Worker loop error: {repr(e)}")
            time.sleep(10)

if __name__ == "__main__":
    loop()
