# app/auto_worker.py
import os
import time
import math
import json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ---------- 環境變數 ----------
SYMBOLS = os.getenv("SYMBOLS", "XRP/USDT").split(",")
SYMBOLS = [s.strip() for s in SYMBOLS if s.strip()]
INTERVAL  = os.getenv("INTERVAL", "15m")
LOOP_SEC  = int(os.getenv("LOOP_SEC", "900"))        # 15 分鐘
DRY_RUN   = os.getenv("DRY_RUN", "true").lower() == "true"
TESTNET   = os.getenv("TESTNET", "true").lower() == "true"
BASE_URL  = os.getenv("BASE_URL", "")                # 可留空
EXCHANGE  = os.getenv("EXCHANGE", "binance")

# 交易大小（以 quote 幣，例如 USDT 金額）；實單前務必調整成符合交易所最小單位
QUOTE_SIZE = float(os.getenv("TRADE_SIZE_USDT", "10"))

BINANCE_KEY    = os.getenv("BINANCE_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

TZ = timezone(timedelta(hours=8))  # Asia/Taipei for logs

def log(msg, **kw):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", (kw if kw else ""))
    
# ---------- 交易所連線 ----------
def build_exchange():
    if EXCHANGE.lower() == "binance":
        params = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True
            }
        }
        if BINANCE_KEY and BINANCE_SECRET:
            params["apiKey"] = BINANCE_KEY
            params["secret"] = BINANCE_SECRET
        ex = ccxt.binance(params)

        # 測試網處理
        if TESTNET:
            ex.set_sandbox_mode(True)
            # 對 ccxt binance sandbox，使用 testnet 網域
            ex.urls["api"] = {
                "public": "https://testnet.binance.vision/api",
                "private": "https://testnet.binance.vision/api",
            }
        if BASE_URL:
            # 若你想強制覆寫 API 網域（通常不用）
            ex.urls["api"]["public"] = BASE_URL
            ex.urls["api"]["private"] = BASE_URL

        return ex
    else:
        raise RuntimeError(f"Unsupported EXCHANGE: {EXCHANGE}")

# ---------- 指標 ----------
def indicators(df: pd.DataFrame):
    # df 欄: open high low close volume ts (毫秒)
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

    out = pd.DataFrame({
        "sma20": sma,
        "bb_upper": upper,
        "bb_lower": lower,
        "k": k,
        "d": d,
        "j": j,
        "macd_dif": dif,
        "macd_dea": dea,
        "macd_hist": hist
    }, index=df.index)

    return out

# ---------- 取 K 線 ----------
def fetch_klines(ex, symbol: str, timeframe: str, limit: int = 200):
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    return df

# ---------- 非常保守策略（範例） ----------
# 僅示意：MACD 由負翻正 & K > D & 收在 SMA20 上 → "buy"
#       MACD 由正翻負 & K < D & 收在 SMA20 下 → "sell"
def simple_signal(df: pd.DataFrame, ind: pd.DataFrame):
    if len(df) < 30:
        return None, {}

    c  = float(df["close"].iloc[-1])
    c1 = float(df["close"].iloc[-2])

    sma  = float(ind["sma20"].iloc[-1])
    dif0 = float(ind["macd_dif"].iloc[-1])
    dea0 = float(ind["macd_dea"].iloc[-1])
    dif1 = float(ind["macd_dif"].iloc[-2])
    dea1 = float(ind["macd_dea"].iloc[-2])

    k0 = float(ind["k"].iloc[-1]); d0 = float(ind["d"].iloc[-1])

    # 金叉由下往上
    macd_cross_up   = (dif1 <= dea1) and (dif0 > dea0)
    macd_cross_down = (dif1 >= dea1) and (dif0 < dea0)

    if macd_cross_up and (k0 > d0) and (c > sma):
        return "buy", {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0}
    if macd_cross_down and (k0 < d0) and (c < sma):
        return "sell", {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0}
    return None, {}

# ---------- 下單（僅示意；正式前請務必驗證最小單位/精度） ----------
def place_order(ex, symbol: str, side: str, quote_amount: float):
    # 以 quote 金額換算數量（市價單）
    ticker = ex.fetch_ticker(symbol)
    price = float(ticker["last"] or ticker["close"])
    amount = quote_amount / price
    # 取精度
    markets = ex.load_markets()
    m = markets[symbol]
    precision_amount = m["precision"].get("amount", None)
    if precision_amount is not None:
        step = 10 ** (-precision_amount)
        amount = math.floor(amount * step) / step

    log(f"PLACE ORDER (demo) {side} {symbol} amount={amount:.6f} (≈{quote_amount} quote)")
    if DRY_RUN:
        return {"dry_run": True, "side": side, "symbol": symbol, "amount": amount, "price": price}

    # 市價單
    if side == "buy":
        return ex.create_market_buy_order(symbol, amount)
    else:
        return ex.create_market_sell_order(symbol, amount)

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
