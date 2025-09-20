# app/auto_worker.py (Futures + TP/SL + ATR)
import os, time, json, math
import ccxt
import pandas as pd
import urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta

# ---------- 小工具 ----------
def _parse_seconds(text: str) -> int:
    if not text:
        return 900
    t = str(text).strip().lower()
    try:
        if t.endswith("ms"): return max(1, int(float(t[:-2]) / 1000))
        if t.endswith("s"):  return int(float(t[:-1]))
        if t.endswith("m"):  return int(float(t[:-1]) * 60)
        if t.endswith("h"):  return int(float(t[:-1]) * 3600)
        return int(float(t))
    except Exception:
        return 900

TZ = timezone(timedelta(hours=8))
def log(msg, **kw):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", (kw if kw else ""))

# ---------- 環境變數 ----------
_symbols_env = os.getenv("SYMBOLS") or os.getenv("SYMBOL", "XRP/USDT")
SYMBOLS = [s.strip() for s in _symbols_env.split(",") if s.strip()]

INTERVAL = os.getenv("INTERVAL", "15m")
_sleep_env = os.getenv("LOOP_SLEEP_SEC") or os.getenv("LOOP_SEC")
LOOP_SEC = _parse_seconds(_sleep_env) if _sleep_env else _parse_seconds(INTERVAL)

DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1","true","yes")
TESTNET = os.getenv("TESTNET", "true").lower() in ("1","true","yes")
EXCHANGE = os.getenv("EXCHANGE", "binance").lower()

MARKET_TYPE = os.getenv("MARKET_TYPE", "spot").lower()  # spot | future
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
POSITION_MODE = os.getenv("POSITION_MODE", "isolated").lower()  # isolated | cross
HEDGE_MODE = os.getenv("HEDGE_MODE", "oneway").lower()          # oneway | hedge

# 風控/出場
MIN_WINRATE_ENV = os.getenv("MIN_WINRATE")              # e.g. 0.6
SETTINGS_PATH = os.getenv("RUNTIME_SETTINGS_PATH", "runtime_settings.json")
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0"))   # 0: 用固定 USDT 下單；>0: 以資金風險反推數量
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "10"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_RR       = float(os.getenv("TP_RR", "1.8"))
TRAILING_ATR= float(os.getenv("TRAILING_ATR", "0"))  # 0 表示不啟用移動停損
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))

# LINE Notify
LINE_TOKEN = os.getenv("LINE_NOTIFY_TOKEN", "")
LINE_NOTIFY_ON = os.getenv("LINE_NOTIFY_ON", "orders").lower()  # orders | all | off
def notify(msg: str):
    if not LINE_TOKEN or LINE_NOTIFY_ON == "off":
        return
    data = urllib.parse.urlencode({"message": msg}).encode("utf-8")
    req = urllib.request.Request(
        "https://notify-api.line.me/api/notify",
        data=data,
        headers={"Authorization": f"Bearer {LINE_TOKEN}",
                 "Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
    except Exception as e:
        log(f"LINE notify error: {repr(e)}")

# ---------- 交易所初始化 ----------
def build_exchange():
    if EXCHANGE != "binance":
        raise RuntimeError(f"Unsupported EXCHANGE: {EXCHANGE}")

    options = {"adjustForTimeDifference": True}
    if MARKET_TYPE == "future":
        options["defaultType"] = "future"
    else:
        options["defaultType"] = "spot"

    ex = ccxt.binance({
        "apiKey": os.getenv("BINANCE_KEY", ""),
        "secret": os.getenv("BINANCE_SECRET", ""),
        "enableRateLimit": True,
        "options": options,
    })
    # 測試網
    ex.set_sandbox_mode(TESTNET)
    markets = ex.load_markets()  # 取得精度/最小單位
    # 合約：設定槓桿/保證金/持倉模式
    if MARKET_TYPE == "future":
        # 持倉模式（hedge/oneway）
        try:
            ex.set_position_mode(HEDGE_MODE == "hedge")
        except Exception as e:
            log(f"set_position_mode warn: {repr(e)}")
        for sym in SYMBOLS:
            try:
                ex.set_leverage(LEVERAGE, sym)
            except Exception as e:
                log(f"set_leverage warn {sym}: {repr(e)}")
            try:
                ex.set_margin_mode(POSITION_MODE, sym)
            except Exception as e:
                log(f"set_margin_mode warn {sym}: {repr(e)}")
    return ex

# ---------- 技術指標 ----------
def indicators(df: pd.DataFrame):
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    sma20 = close.rolling(20).mean()
    std   = close.rolling(20).std()
    bb_upper = sma20 + 2 * std
    bb_lower = sma20 - 2 * std

    low_n  = low.rolling(9).min()
    high_n = high.rolling(9).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(alpha=1/3).mean()
    d = k.ewm(alpha=1/3).mean()
    j = 3 * k - 2 * d

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea

    return pd.DataFrame({
        "sma20": sma20,
        "bb_upper": bb_upper, "bb_lower": bb_lower,
        "k": k, "d": d, "j": j,
        "macd_dif": dif, "macd_dea": dea, "macd_hist": hist,
    }, index=df.index)

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1])

def fetch_klines(ex, symbol: str, timeframe: str, limit: int = 200):
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])

# ---------- 訊號 ----------
def simple_signal(df: pd.DataFrame, ind: pd.DataFrame):
    if len(df) < 30:
        return None, 0.0, {}
    c  = float(df["close"].iloc[-1])
    sma = float(ind["sma20"].iloc[-1])
    dif0 = float(ind["macd_dif"].iloc[-1]);  dea0 = float(ind["macd_dea"].iloc[-1])
    dif1 = float(ind["macd_dif"].iloc[-2]);  dea1 = float(ind["macd_dea"].iloc[-2])
    k0 = float(ind["k"].iloc[-1]); d0 = float(ind["d"].iloc[-1])

    bull = {"cross_up": (dif1 <= dea1) and (dif0 > dea0),
            "k_gt_d": (k0 > d0),
            "close_gt_sma": (c > sma)}
    bear = {"cross_down": (dif1 >= dea1) and (dif0 < dea0),
            "k_lt_d": (k0 < d0),
            "close_lt_sma": (c < sma)}

    b = sum(1 for v in bull.values() if v)
    s = sum(1 for v in bear.values() if v)

    if b == 0 and s == 0:
        return None, 0.0, {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0}

    if b >= s:
        return "buy", b/3.0, {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0, **bull}
    else:
        return "sell", s/3.0, {"close": c, "sma20": sma, "k": k0, "d": d0, "dif": dif0, "dea": dea0, **bear}

# ---------- 設定讀取 ----------
def _load_trade_mode(default_val="both"):
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            v = str(d.get("trade_mode", default_val)).lower()
            if v in ("both","long_only","short_only"):
                return v
    except Exception:
        pass
    return default_val

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

# ---------- 風險/倉位計算 ----------
def qty_from_quote(ex, symbol: str, quote_amount: float, price: float):
    amount = quote_amount / price
    m = ex.markets.get(symbol) or {}
    prec = (m.get("precision") or {}).get("amount")
    if prec is not None:
        step = 10 ** (-prec)
        amount = math.floor(amount / step) * step
    return max(amount, 0.0)

def bracket_prices(side: str, entry: float, atr: float, sl_mult=1.5, rr=1.8):
    if side == "buy":
        sl = entry - atr * sl_mult
        tp = entry + (entry - sl) * rr
    else:
        sl = entry + atr * sl_mult
        tp = entry - (sl - entry) * rr
    return (tp, sl)

def count_open_positions(ex, symbol: str):
    try:
        poss = ex.fetch_positions([symbol])
        n = 0
        for p in poss:
            contracts = float(p.get("contracts") or 0)
            if contracts > 0:
                n += 1
        return n
    except Exception:
        return 0

# ---------- 下單（現貨） ----------
def place_spot(ex, symbol: str, side: str, quote_amount: float):
    ticker = ex.fetch_ticker(symbol)
    price = float(ticker["last"] or ticker["close"])
    amount = qty_from_quote(ex, symbol, quote_amount, price)
    log(f"[{symbol}] SPOT place {side} amount={amount:.8f} (~{quote_amount} quote)")
    if DRY_RUN:
        return {"dry_run": True, "side": side, "symbol": symbol, "amount": amount, "price": price}
    return ex.create_order(symbol, "market", side, amount)

# ---------- 下單（合約：市價 + reduceOnly 的 TP/SL） ----------
def place_futures_with_bracket(ex, symbol: str, side: str, quote_amount: float, atr: float):
    # 取得入場價
    ticker = ex.fetch_ticker(symbol)
    entry = float(ticker["last"] or ticker["close"])
    amount = qty_from_quote(ex, symbol, quote_amount, entry)
    if amount <= 0:
        raise RuntimeError("amount <= 0 after precision adjust")

    tp, sl = bracket_prices(side, entry, atr, SL_ATR_MULT, TP_RR)

    # 方向
    close_side = "sell" if side == "buy" else "buy"

    if DRY_RUN:
        log(f"[{symbol}] FUTURES DRY-RUN {side} amount={amount:.6f} entry≈{entry} tp≈{tp} sl≈{sl}")
        return {"dry_run": True, "entry": entry, "tp": tp, "sl": sl, "amount": amount}

    # 1) 市價開倉
    order = ex.create_order(symbol, "market", side, amount)
    log(f"[{symbol}] opened {side} amount={amount} entry≈{entry}")

    # 2) 止損（STOP_MARKET reduceOnly）
    sl_params = {
        "reduceOnly": True,
        "stopPrice": sl,
        "timeInForce": "GTC",
        "workingType": "MARK_PRICE",
    }
    if HEDGE_MODE == "hedge":
        sl_params["positionSide"] = "LONG" if side == "buy" else "SHORT"
    ex.create_order(symbol, "STOP_MARKET", close_side, amount, None, sl_params)
    log(f"[{symbol}] SL placed at {sl}")

    # 3) 止盈（TAKE_PROFIT_MARKET reduceOnly）
    tp_params = {
        "reduceOnly": True,
        "stopPrice": tp,
        "timeInForce": "GTC",
        "workingType": "MARK_PRICE",
    }
    if HEDGE_MODE == "hedge":
        tp_params["positionSide"] = "LONG" if side == "buy" else "SHORT"
    ex.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, amount, None, tp_params)
    log(f"[{symbol}] TP placed at {tp}")

    return {"ok": True, "entry": entry, "tp": tp, "sl": sl, "amount": amount}

# ---------- 主迴圈 ----------
def loop():
    ex = build_exchange()
    log(f"Worker started | MARKET_TYPE={MARKET_TYPE} TESTNET={TESTNET} INTERVAL={INTERVAL} DRY_RUN={DRY_RUN} SYMBOLS={SYMBOLS} LOOP_SEC={LOOP_SEC}")
    notify(f"Worker started | {MARKET_TYPE.upper()} TESTNET={TESTNET} LOOP={LOOP_SEC}s")

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
                    close_px = float(df["close"].iloc[-1])

                    if not side:
                        log(f"[{symbol}] no-signal {t_local} | close={close_px:.6f}")
                        if LINE_NOTIFY_ON == "all":
                            notify(f"【{symbol}】NO SIGNAL {t_local}")
                        continue

                    if trade_mode == "long_only" and side == "sell":
                        log(f"[{symbol}] signal={side} conf={conf:.2f} blocked by mode=long_only", **ctx); continue
                    if trade_mode == "short_only" and side == "buy":
                        log(f"[{symbol}] signal={side} conf={conf:.2f} blocked by mode=short_only", **ctx); continue

                    if conf < min_wr:
                        log(f"[{symbol}] signal={side} conf={conf:.2f} < min_wr={min_wr:.2f}, skip", **ctx); continue

                    # 合約同時檢查「同時持倉上限」
                    if MARKET_TYPE == "future" and MAX_CONCURRENT > 0:
                        open_n = count_open_positions(ex, symbol)
                        if open_n >= MAX_CONCURRENT:
                            log(f"[{symbol}] skip: open positions {open_n} >= MAX_CONCURRENT {MAX_CONCURRENT}")
                            continue

                    log(f"[{symbol}] EXECUTE signal={side} conf={conf:.2f} at {t_local}", **ctx)
                    if MARKET_TYPE == "future":
                        atr = compute_atr(df, 14)
                        res = place_futures_with_bracket(ex, symbol, side, TRADE_SIZE_USDT, atr)
                    else:
                        res = place_spot(ex, symbol, side, TRADE_SIZE_USDT)

                    log(f"[{symbol}] RESULT: {json.dumps(res, ensure_ascii=False)}")
                    try:
                        msg = f"【{symbol}】{MARKET_TYPE.upper()} {side.upper()} {t_local}\nconf={conf:.2f} min_wr={min_wr:.2f}\nsize≈{TRADE_SIZE_USDT} USDT"
                        if MARKET_TYPE == "future" and isinstance(res, dict):
                            msg += f"\nentry≈{res.get('entry'):.6f} tp≈{res.get('tp'):.6f} sl≈{res.get('sl'):.6f}"
                        notify(msg)
                    except Exception:
                        pass

                except Exception as e:
                    log(f"[{symbol}] error: {repr(e)}")
                    notify(f"[{symbol}] error: {repr(e)}")

            log(f"sleep {LOOP_SEC}s ...")
            time.sleep(LOOP_SEC)
        except KeyboardInterrupt:
            log("worker stopped by KeyboardInterrupt"); notify("Worker stopped"); break
        except Exception as e:
            log(f"loop error: {repr(e)}"); notify(f"Worker loop error: {repr(e)}"); time.sleep(10)

if __name__ == "__main__":
    loop()
