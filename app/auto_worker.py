# app/auto_worker.py
import os
import time
import math
import json
import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta

# ---------- 環境變數 ----------
# 支援 SYMBOLS 或舊的單一 SYMBOL
_symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "XRP/USDT")
SYMBOLS = [s.strip() for s in _symbols_env.split(",") if s.strip()]

INTERVAL  = os.getenv("INTERVAL", "15m")
LOOP_SEC  = int(os.getenv("LOOP_SEC", "900"))        # 15 分鐘輪巡
DRY_RUN   = os.getenv("DRY_RUN", "true").lower() == "true"
TESTNET   = os.getenv("TESTNET", "true").lower() == "true"
EXCHANGE  = os.getenv("EXCHANGE", "binance")
EXCHANGE_LOWER = EXCHANGE.lower()

# 單筆以 quote 幣的金額（USDT）下單；正式上線前務必調整成交易所允許的最小單位
QUOTE_SIZE = float(os.getenv("TRADE_SIZE_USDT", "10"))

BINANCE_KEY    = os.getenv("BINANCE_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

TZ = timezone(timedelta(hours=8))  # Asia/Taipei for logs

def log(msg, **kw):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", (kw if kw else ""))

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

    # 關鍵：僅用 ccxt 的 sandbox 切測試網；不要再覆寫任何 ex.urls 或 BASE_URL
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

# ---------- 很保守的示範訊號 ----------
# MACD 由負翻正 & K > D & 收在 SMA20 上 → buy
# MACD 由正翻負 & K < D & 收在 SMA20 下 → sell
def simple_signal(df: pd.DataFrame, ind: pd.DataFrame):
    if len(df) < 30:
        return None, {}

    c  = float(df["close"].iloc[-1])
    sma  = float(ind["sma20"].iloc[-1])
    dif0 = float(ind["macd_dif"].iloc[-1])
    dea0 = float(ind["macd_dea"].iloc[-1])
    dif1 = float(ind["macd_dif"].iloc[-2])
    dea1 = float(ind["macd_dea"].iloc[-2])
    k0 = float(ind["k"].iloc[-1])
    d0 = float(ind["d"].iloc[-1])

    cross_up   = (dif1 <= dea1) and (dif0 >  dea0)
    cross_down = (dif1 >= dea1) and (dif0 <  dea0)

    if cross_up and (k0 > d0) and (c > sma):
        return "buy",  {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0}
    if cross_down and (k0 < d0) and (c < sma):
        return "sell", {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0}
    return None, {}

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

    while True:
        try:
            for symbol in SYMBOLS:
                try:
                    df = fetch_klines(ex, symbol, INTERVAL, limit=200)
                    ind = indicators(df)
                    sig, ctx = simple_signal(df, ind)
                    last_ts = int(df["ts"].iloc[-1])
                    t_local = datetime.fromtimestamp(last_ts/1000, TZ).strftime("%H:%M")

                    if sig:
                        log(f"[{symbol}] SIGNAL={sig} at {t_local}", **ctx)
                        res = place_order(ex, symbol, sig, QUOTE_SIZE)
                        log(f"[{symbol}] RESULT: {json.dumps(res, ensure_ascii=False)}")
                    else:
                        log(f"[{symbol}] no-signal at {t_local} | close={float(df['close'].iloc[-1]):.6f}")
                except Exception as e:
                    log(f"[{symbol}] error: {repr(e)}")
            log(f"sleep {LOOP_SEC}s ...")
            time.sleep(LOOP_SEC)
        except KeyboardInterrupt:
            log("worker stopped by KeyboardInterrupt")
            break
        except Exception as e:
            log(f"loop error: {repr(e)}")
            time.sleep(10)

if __name__ == "__main__":
    loop()
