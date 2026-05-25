import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


# ============================================================
# CELL 1 — Imports, MT5 init, refresh timer
# ============================================================
@app.cell
def _():
    import importlib
    import MetaTrader5 as mt5
    import polars as pl
    import numpy as np
    import marimo as mo
    from datetime import datetime, timezone
    from concurrent.futures import ThreadPoolExecutor

    import modules.config as cfg
    import modules.features as ft
    import modules.data_engine as de
    import modules.model_handler as mh
    import modules.journal_engine as je

    for _m in [cfg, ft, de, mh, je]:
        importlib.reload(_m)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    acc     = mt5.account_info()
    balance = acc.balance if acc else 0.0

    refresh_timer = mo.ui.refresh(
        options=["1m", "3m", "5m", "10m"],
        default_interval="3m"
    )

    # Cooldown state — only set when a position is actually opened
    get_cooldown, set_cooldown = mo.state({})

    refresh_timer
    return (
        ThreadPoolExecutor,
        acc,
        balance,
        cfg,
        datetime,
        de,
        ft,
        get_cooldown,
        je,
        mh,
        mo,
        mt5,
        np,
        pl,
        refresh_timer,
        set_cooldown,
        timezone,
    )


# ============================================================
# CELL 2 — Boot: ADR + D1 SMA + model training (once)
# ============================================================
@app.cell
def _(ThreadPoolExecutor, cfg, ft, je, mh, mo, mt5, np, pl):
    """
    Runs once at startup — no refresh_timer dependency.

    Fetches D1 bars for ADR sizing and D1 SMA direction.
    Trains model_long / model_short — shown as ML prob (informational).
    """
    with mo.status.spinner("Booting FinMetrix..."):
        from modules.data_engine import get_mt5_df

        # ── D1 fetch: ADR + D1 SMA ────────────────────────────────────────────
        _D1_BARS = max(cfg.ADR_LOOKBACK_DAYS, cfg.D1_SMA_SLOW) + 5

        def _fetch_d1(symbol):
            _rates = mt5.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_D1, 0, _D1_BARS
            )
            if _rates is None or len(_rates) == 0:
                return symbol, None, None, None
            _info = mt5.symbol_info(symbol)
            if _info is None:
                return symbol, None, None, None

            _pip      = _info.point * (10.0 if _info.digits in [3, 5] else 1.0)
            _df       = pl.DataFrame(_rates)
            _closes   = _df["close"].to_numpy().astype(float)

            # ADR
            _adr_price = float(
                (_df["high"] - _df["low"]).tail(cfg.ADR_LOOKBACK_DAYS).mean()
            )
            _adr_pips  = round(_adr_price / _pip, 1)

            # D1 SMA cross: fast SMA > slow SMA = bullish daily bias
            _d1_bull = None
            if len(_closes) >= cfg.D1_SMA_SLOW:
                _fast = float(np.mean(_closes[-cfg.D1_SMA_FAST:]))
                _slow = float(np.mean(_closes[-cfg.D1_SMA_SLOW:]))
                _d1_bull = _fast > _slow

            return symbol, _adr_price, _adr_pips, _d1_bull

        with ThreadPoolExecutor(max_workers=8) as _ex:
            _d1_res = list(_ex.map(_fetch_d1, cfg.SYMBOLS))

        adr_map      = {}
        adr_pips_map = {}
        d1_bull_map  = {}

        for _sym, _ap, _apips, _d1b in _d1_res:
            if _ap    is not None: adr_map[_sym]      = _ap
            if _apips is not None: adr_pips_map[_sym] = _apips
            if _d1b   is not None: d1_bull_map[_sym]  = _d1b

        # ── Model training ────────────────────────────────────────────────────
        _long_rows  = []
        _short_rows = []

        for _symbol in cfg.SYMBOLS:
            _df = get_mt5_df(_symbol, cfg.TIMEFRAME, cfg.BARS)
            if _df is None:
                continue
            _df = ft.engineer_features(_df)
            _df = ft.triple_barrier(
                _df, tp_mult=cfg.TP1_MULT,
                sl_mult=cfg.SL_MULT, max_hold=cfg.MAX_HOLD,
            )
            _df_long = _df.filter(
                (pl.col("ema144_slope") > 0) &
                (pl.col("vol_ratio")    > 1.0)
            ).select([*cfg.FEATURES, "target_long"]).drop_nulls()

            _df_short = _df.filter(
                (pl.col("ema144_slope") < 0) &
                (pl.col("vol_ratio")    > 1.0)
            ).select([*cfg.FEATURES, "target_short"]).drop_nulls()

            if _df_long.height  > 0:
                _long_rows.append(_df_long.cast(pl.Float64).to_numpy())
            if _df_short.height > 0:
                _short_rows.append(_df_short.cast(pl.Float64).to_numpy())

        model_long  = None
        _n_long     = 0
        if _long_rows:
            _data_l    = np.vstack(_long_rows)
            _n_long    = len(_data_l)
            model_long = mh.train_model(_data_l[:, :-1], _data_l[:, -1])

        model_short = None
        _n_short    = 0
        if _short_rows:
            _data_s     = np.vstack(_short_rows)
            _n_short    = len(_data_s)
            model_short = mh.train_model(_data_s[:, :-1], _data_s[:, -1])

    kelly_risk_pct = je.calculate_adaptive_risk()
    _kelly_src     = "adaptive" if kelly_risk_pct > cfg.KELLY_MIN_FLOOR else "floor"

    mo.md(f"""
✅ **FinMetrix ready**

| | |
| :--- | :--- |
| **Signal logic** | H1 SMA({cfg.H1_SMA_FAST}/{cfg.H1_SMA_SLOW}) · D1 SMA({cfg.D1_SMA_FAST}/{cfg.D1_SMA_SLOW}) · RSI({cfg.RSI_PERIOD_SIG}) MA({cfg.RSI_MA_FAST}/{cfg.RSI_MA_SLOW}) · Rel ATR ≥ {cfg.REL_ATR_MIN} · Vol > avg |
| **ADR / D1 loaded** | {len(adr_map)} / {len(d1_bull_map)} of {len(cfg.SYMBOLS)} symbols |
| **Long / Short rows** | {_n_long:,} / {_n_short:,} |
| **Kelly risk** | {kelly_risk_pct*100:.2f}% ({_kelly_src}) |
| **SL / TP1 / TP2** | {cfg.SL_ADR_FRACTION*100:.0f}% / {cfg.TP1_ADR_FRACTION*100:.0f}% / {cfg.TP2_ADR_FRACTION*100:.0f}% of {cfg.ADR_LOOKBACK_DAYS}-day ADR |
| **Auto-close (TME)** | Age > {cfg.TME_MAX_BARS}h AND R < {cfg.TME_MIN_R} |
    """)
    return (
        adr_map, adr_pips_map, d1_bull_map,
        kelly_risk_pct, model_long, model_short,
    )


# ============================================================
# CELL 3 — Scanner (every refresh tick)
# ============================================================
@app.cell
def _(
    ThreadPoolExecutor,
    adr_map,
    adr_pips_map,
    balance,
    cfg,
    d1_bull_map,
    datetime,
    de,
    ft,
    get_cooldown,
    kelly_risk_pct,
    model_long,
    model_short,
    mt5,
    np,
    pl,
    refresh_timer,
    set_cooldown,
    timezone,
):
    refresh_timer.value

    # ── SMA helper ────────────────────────────────────────────────────────────
    def _sma(arr, period):
        if len(arr) < period:
            return None
        return float(np.mean(arr[-period:]))

    # ── Wilder RSI from scratch ───────────────────────────────────────────────
    def _calc_rsi(closes, period):
        n      = len(closes)
        rsi    = np.full(n, np.nan)
        if n <= period:
            return rsi
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = float(np.mean(gains[:period]))
        avg_l  = float(np.mean(losses[:period]))
        for i in range(period, n - 1):
            avg_g = (avg_g * (period - 1) + gains[i])  / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
            rs    = avg_g / avg_l if avg_l > 0 else 0.0
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # ── Fetch H1 raw bars (for signal calculation) ────────────────────────────
    # 300 bars is enough for all SMAs and RSI MA periods.
    def _fetch_h1(symbol):
        _r = mt5.copy_rates_from_pos(symbol, cfg.TIMEFRAME, 0, 300)
        if _r is None or len(_r) == 0:
            return None
        return symbol, _r

    with ThreadPoolExecutor(max_workers=8) as _ex:
        _h1_raw = list(_ex.map(_fetch_h1, cfg.SYMBOLS))
    _h1_map = {s: r for s, r in _h1_raw if r is not None}

    # ── Fetch engineered features (for ML prob only) ──────────────────────────
    def _fetch_feat(symbol):
        _df = de.get_mt5_df(symbol, cfg.TIMEFRAME, cfg.BARS)
        if _df is None:
            return None
        return symbol, ft.engineer_features_fast(_df)

    with ThreadPoolExecutor(max_workers=8) as _ex2:
        _feat_raw = list(_ex2.map(_fetch_feat, cfg.SYMBOLS))
    _feat_map = {s: df for s, df in _feat_raw if df is not None}

    # ── Candle-close guard ────────────────────────────────────────────────────
    scan_market = True
    if _h1_map:
        _ref_sym  = "EURUSD" if "EURUSD" in _h1_map else next(iter(_h1_map))
        _last_str = str(int(_h1_map[_ref_sym]["time"][-1]))
        if cfg.LAST_CANDLE_FILE.exists():
            if _last_str == cfg.LAST_CANDLE_FILE.read_text().strip():
                scan_market = False
        if scan_market:
            cfg.LAST_CANDLE_FILE.write_text(_last_str)

    # ── Cooldown: only decrement on new candle ────────────────────────────────
    _cd = get_cooldown()
    if scan_market:
        _cd = {s: c - 1 for s, c in _cd.items() if c > 1}
        set_cooldown(_cd)

    # ── Open positions for exposure filter ────────────────────────────────────
    _positions    = mt5.positions_get() or []
    _open_symbols = {p.symbol for p in _positions}
    _exposed_curr = {c for sym in _open_symbols for c in (sym[:3], sym[3:6])}

    def _is_blocked(symbol):
        if not _open_symbols:
            return False
        return symbol[:3] in _exposed_curr or symbol[3:6] in _exposed_curr

    # ── Score every pair ──────────────────────────────────────────────────────
    signal_rows = []
    today_count = de.signals_today()
    _now_ts     = datetime.now(timezone.utc)

    for symbol in cfg.SYMBOLS:
        if symbol not in _h1_map:
            continue

        _h1_closes = _h1_map[symbol]["close"].astype(float)
        _h1_vols   = _h1_map[symbol]["tick_volume"].astype(float)
        _price     = float(_h1_closes[-1])

        # ── Gate 1: H1 SMA cross ─────────────────────────────────────────────
        _h1_fast = _sma(_h1_closes, cfg.H1_SMA_FAST)
        _h1_slow = _sma(_h1_closes, cfg.H1_SMA_SLOW)
        if _h1_fast is None or _h1_slow is None:
            continue
        _h1_bull = _h1_fast > _h1_slow

        # ── Gate 2: D1 SMA cross (fetched at boot) ───────────────────────────
        _d1_bull = d1_bull_map.get(symbol)
        # None means D1 data unavailable — don't block, show ? in table

        # ── Gate 3: RSI MA cross ─────────────────────────────────────────────
        _rsi_arr     = _calc_rsi(_h1_closes, cfg.RSI_PERIOD_SIG)
        _rsi_valid   = _rsi_arr[~np.isnan(_rsi_arr)]
        _rsi_ma_fast = _sma(_rsi_valid, cfg.RSI_MA_FAST)
        _rsi_ma_slow = _sma(_rsi_valid, cfg.RSI_MA_SLOW)
        if _rsi_ma_fast is None or _rsi_ma_slow is None:
            continue
        _rsi_bull = _rsi_ma_fast > _rsi_ma_slow

        # ── Gate 4: Relative ATR ─────────────────────────────────────────────
        # Computed from raw H1 bars (high - low as ATR proxy for simplicity;
        # for a proper ATR use the feature map if available)
        _rel_atr = 1.0
        if symbol in _feat_map:
            _atr_col = _feat_map[symbol]["atr"].drop_nulls().to_numpy()
            if len(_atr_col) >= 20:
                _rel_atr = float(_atr_col[-1]) / float(np.mean(_atr_col[-20:]))
        _atr_ok = _rel_atr >= cfg.REL_ATR_MIN

        # ── Gate 5: Volume above average ─────────────────────────────────────
        _vol_avg = float(np.mean(_h1_vols[-cfg.VOL_AVG_BARS:])) if len(_h1_vols) >= cfg.VOL_AVG_BARS else 0.0
        _vol_ok  = float(_h1_vols[-1]) > _vol_avg if _vol_avg > 0 else False

        # ── All five gates must agree ─────────────────────────────────────────
        _d1_gate_l = (_d1_bull is True) or (_d1_bull is None)
        _d1_gate_s = (_d1_bull is False) or (_d1_bull is None)

        _is_long  = _h1_bull  and _d1_gate_l and _rsi_bull  and _atr_ok and _vol_ok
        _is_short = (not _h1_bull) and _d1_gate_s and (not _rsi_bull) and _atr_ok and _vol_ok

        # ── Cooldown only blocks if position already open on this symbol ──────
        # (not just because a signal appeared — user decides whether to trade)
        _sym_in_market = symbol in _open_symbols
        _corr_blocked  = _is_blocked(symbol) and not _sym_in_market
        _in_cooldown   = (symbol in _cd) and _sym_in_market

        signal = "Neutral"
        if _is_long  and not _in_cooldown and not _corr_blocked:
            signal = "BUY"
        elif _is_short and not _in_cooldown and not _corr_blocked:
            signal = "SELL"

        # ── ML prob (informational) ───────────────────────────────────────────
        _prob = 0.0
        if symbol in _feat_map:
            _now = _feat_map[symbol].tail(1).to_dicts()[0]
            _X   = np.array([[_now.get(f, 0.0) for f in cfg.FEATURES]])
            if signal == "BUY"  and model_long:
                _prob = float(model_long.predict_proba(_X)[0][1])
            elif signal == "SELL" and model_short:
                _prob = float(model_short.predict_proba(_X)[0][1])

        # ── Sizing ────────────────────────────────────────────────────────────
        _adr   = adr_map.get(symbol)
        _adr_p = adr_pips_map.get(symbol, 0.0)

        if _adr and _adr > 0:
            _sl_dist  = _adr * cfg.SL_ADR_FRACTION
            _tp1_dist = _adr * cfg.TP1_ADR_FRACTION
            _tp2_dist = _adr * cfg.TP2_ADR_FRACTION
        else:
            _atr_fb   = float(_feat_map[symbol]["atr"][-1]) if symbol in _feat_map else 0.001
            _sl_dist  = _atr_fb * cfg.SL_MULT
            _tp1_dist = _atr_fb * cfg.TP1_MULT
            _tp2_dist = _atr_fb * cfg.TP2_MULT
            _adr_p    = 0.0

        _dir = "BUY" if _h1_bull else "SELL"

        if _dir == "BUY":
            _sl  = round(_price - _sl_dist,  5)
            _tp1 = round(_price + _tp1_dist, 5)
            _tp2 = round(_price + _tp2_dist, 5)
        else:
            _sl  = round(_price + _sl_dist,  5)
            _tp1 = round(_price - _tp1_dist, 5)
            _tp2 = round(_price - _tp2_dist, 5)

        _lot_size = 0.0
        _risk_gbp = 0.0
        _sym_info = mt5.symbol_info(symbol)
        if _sym_info and _sl_dist > 0:
            _sl_ticks = _sl_dist / _sym_info.trade_tick_size
            _lot_size = (balance * kelly_risk_pct) / (_sl_ticks * _sym_info.trade_tick_value)
            _lot_size = max(_sym_info.volume_min,
                            round(_lot_size / _sym_info.volume_step) * _sym_info.volume_step)
            _risk_gbp = round(_lot_size * _sl_ticks * _sym_info.trade_tick_value, 2)

        # ── Log signal on new candle only ─────────────────────────────────────
        # Cooldown is set here — only when you've actually taken the trade.
        # The log_signal call records it; cooldown blocks duplicate signals
        # while a position is open on that symbol.
        if scan_market and signal != "Neutral":
            today_count += 1
            de.log_signal({
                "time":        _now_ts,
                "candle_time": datetime.fromtimestamp(
                    int(_h1_map[symbol]["time"][-1]), tz=timezone.utc
                ),
                "symbol":      symbol,
                "signal":      signal,
                "entry":       _price,
                "sl":          _sl,
                "tp1":         _tp1,
                "tp2":         _tp2,
                "adr_pips":    _adr_p,
                "rel_atr":     round(_rel_atr, 3),
                "lot_size":    _lot_size,
                "prob":        _prob,
            })

        # Status tag
        _tag = ""
        if _corr_blocked:
            _tag = " [exp]"
        elif _in_cooldown:
            _tag = f" [cd{_cd.get(symbol, 0)}]"

        # D1 display
        _d1_str = "bull" if _d1_bull is True else ("bear" if _d1_bull is False else "?")

        signal_rows.append({
            "Symbol":   symbol,
            "Signal":   signal + _tag,
            "ADR pips": _adr_p,
            "Rel ATR":  round(_rel_atr, 2),
            "H1 MA":    "bull" if _h1_bull else "bear",
            "D1 MA":    _d1_str,
            "RSI MA":   "bull" if _rsi_bull else "bear",
            "Vol":      "✓" if _vol_ok else "·",
            "Price":    round(_price, 5),
            "Lot":      round(_lot_size, 2),
            "Risk GBP": _risk_gbp,
            "SL":       _sl,
            "TP1":      _tp1,
            "TP2":      _tp2,
            "ML prob":  f"{_prob*100:.0f}%",
        })

    signal_rows.sort(key=lambda x: (
        x["Signal"].split()[0] not in ("BUY", "SELL"),
        -float(x["ML prob"][:-1]),
    ))

    return signal_rows, today_count


# ============================================================
# CELL 4 — Scanner dashboard
# ============================================================
@app.cell
def _(acc, balance, cfg, datetime, kelly_risk_pct, mo, signal_rows, today_count):
    mo.vstack([
        mo.Html("<style>td, th { font-size: 0.9rem !important; }</style>"),
        mo.md(
            f"### {acc.login} · {acc.currency} {balance:,.2f} · "
            f"{datetime.now().strftime('%H:%M:%S')} · "
            f"Signals today: {today_count} · "
            f"Kelly: {kelly_risk_pct*100:.2f}%"
        ),
        mo.md(
            f"Signal = H1 SMA({cfg.H1_SMA_FAST}/{cfg.H1_SMA_SLOW}) "
            f"+ D1 SMA({cfg.D1_SMA_FAST}/{cfg.D1_SMA_SLOW}) "
            f"+ RSI({cfg.RSI_PERIOD_SIG}) MA({cfg.RSI_MA_FAST}/{cfg.RSI_MA_SLOW}) "
            f"+ Rel ATR ≥ {cfg.REL_ATR_MIN} + Vol > avg  |  "
            f"[exp] = currency exposure · [cdN] = cooldown (position open)"
        ),
        mo.ui.table(signal_rows, pagination=False),
    ])
    return


# ============================================================
# CELL 5 — Position manager with exit logic
# ============================================================
@app.cell
def _(
    adr_map, cfg, d1_bull_map, datetime,
    mo, mt5, np, refresh_timer, set_cooldown, timezone,
):
    refresh_timer.value

    # ── SMA / RSI helpers (same as scanner) ──────────────────────────────────
    def _sma_pm(arr, period):
        if len(arr) < period:
            return None
        return float(np.mean(arr[-period:]))

    def _calc_rsi_pm(closes, period):
        n = len(closes)
        rsi = np.full(n, np.nan)
        if n <= period:
            return rsi
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = float(np.mean(gains[:period]))
        avg_l  = float(np.mean(losses[:period]))
        for i in range(period, n - 1):
            avg_g = (avg_g * (period - 1) + gains[i])  / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
            rs    = avg_g / avg_l if avg_l > 0 else 0.0
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    _positions = mt5.positions_get() or []
    _now_ts    = datetime.now(timezone.utc)
    _auto_log  = []

    if not _positions:
        _mgmt = mo.md("*No open positions.*")
    else:
        _rows = []

        for _pos in _positions:
            _sym    = _pos.symbol
            _side   = "BUY" if _pos.type == 0 else "SELL"
            _entry  = _pos.price_open
            _sl     = _pos.sl
            _lots   = _pos.volume
            _pnl    = _pos.profit
            _ticket = _pos.ticket

            _tick   = mt5.symbol_info_tick(_sym)
            _info   = mt5.symbol_info(_sym)
            _price  = (_tick.bid if _side == "BUY" else _tick.ask) if _tick else _entry
            _pip    = (_info.point * (10.0 if _info.digits in [3, 5] else 1.0)) if _info else 0.0001

            # ── SL distances from ADR ─────────────────────────────────────────
            _adr = adr_map.get(_sym)
            if _adr and _adr > 0:
                _sl_d  = _adr * cfg.SL_ADR_FRACTION
                _tp1_d = _adr * cfg.TP1_ADR_FRACTION
                _tp2_d = _adr * cfg.TP2_ADR_FRACTION
                _tr_d  = _adr * cfg.TRAIL_ADR_FRACTION
            else:
                _sl_d  = abs(_entry - _sl) if _sl and _sl != 0 else 0.001
                _tp1_d = _sl_d
                _tp2_d = _sl_d * 2
                _tr_d  = _sl_d * 0.8

            if _sl_d <= 0:
                _rows.append({
                    "Symbol": _sym, "Side": _side, "Lots": _lots,
                    "Entry": round(_entry, 5), "Now": round(_price, 5),
                    "Age(h)": "?", "R": "—", "SL pips": "—",
                    "SL": "SET SL FIRST", "TP1(50%)": "—", "TP2(25%)": "—",
                    "Trail": "—", "TP1": "—", "TP2": "—",
                    "P&L": round(_pnl, 2), "Flags": "—", "Action": "SET SL FIRST",
                })
                continue

            # ── Position R and TP targets ─────────────────────────────────────
            if _side == "BUY":
                _tp1_p   = round(_entry + _tp1_d, 5)
                _tp2_p   = round(_entry + _tp2_d, 5)
                _trail   = round(max(_price - _tr_d, _entry), 5)
                _r       = round((_price - _entry) / _sl_d, 2)
                _tp1_hit = _price >= _tp1_p
                _tp2_hit = _price >= _tp2_p
            else:
                _tp1_p   = round(_entry - _tp1_d, 5)
                _tp2_p   = round(_entry - _tp2_d, 5)
                _trail   = round(min(_price + _tr_d, _entry), 5)
                _r       = round((_entry - _price) / _sl_d, 2)
                _tp1_hit = _price <= _tp1_p
                _tp2_hit = _price <= _tp2_p

            _sl_pips = round(_sl_d / _pip, 1)

            # ── Position age in H1 bars ───────────────────────────────────────
            _open_time = datetime.fromtimestamp(_pos.time, tz=timezone.utc)
            _age_secs  = (_now_ts - _open_time).total_seconds()
            _age_bars  = int(_age_secs / 3600)   # H1 = 3600 seconds

            # ── EXIT FLAGS ────────────────────────────────────────────────────
            _flags = []

            # 1. RSI flag: H1 MA and RSI MA disagree (after >= 2 bars open)
            _rsi_flag = False
            if _age_bars >= 2:
                _h1_rates = mt5.copy_rates_from_pos(_sym, cfg.TIMEFRAME, 0, 100)
                if _h1_rates is not None and len(_h1_rates) >= cfg.H1_SMA_SLOW:
                    _h1_c     = _h1_rates["close"].astype(float)
                    _h1_fast  = _sma_pm(_h1_c, cfg.H1_SMA_FAST)
                    _h1_slow  = _sma_pm(_h1_c, cfg.H1_SMA_SLOW)
                    _rsi_v    = _calc_rsi_pm(_h1_c, cfg.RSI_PERIOD_SIG)
                    _rsi_ok   = _rsi_v[~np.isnan(_rsi_v)]
                    _rsi_f    = _sma_pm(_rsi_ok, cfg.RSI_MA_FAST)
                    _rsi_s    = _sma_pm(_rsi_ok, cfg.RSI_MA_SLOW)
                    if _h1_fast and _h1_slow and _rsi_f and _rsi_s:
                        _h1_bull_now  = _h1_fast > _h1_slow
                        _rsi_bull_now = _rsi_f   > _rsi_s
                        _diverge = (
                            (_side == "BUY"  and (not _h1_bull_now or not _rsi_bull_now)) or
                            (_side == "SELL" and (_h1_bull_now or _rsi_bull_now))
                        )
                        if _diverge:
                            _flags.append("RSI")
                            _rsi_flag = True

            # 2. ATR flag: market compressing while in trade
            _atr_flag = False
            _h1_atr_rates = mt5.copy_rates_from_pos(_sym, cfg.TIMEFRAME, 0, 30)
            if _h1_atr_rates is not None and len(_h1_atr_rates) >= 20:
                _highs   = _h1_atr_rates["high"].astype(float)
                _lows    = _h1_atr_rates["low"].astype(float)
                _ranges  = _highs - _lows
                _cur_atr = float(_ranges[-1])
                _avg_atr = float(np.mean(_ranges[-20:]))
                _rel_atr = _cur_atr / _avg_atr if _avg_atr > 0 else 1.0
                if _rel_atr < cfg.REL_ATR_COMPRESS:
                    _flags.append("ATR")
                    _atr_flag = True

            # 3. ADR flag: > ADR_USED_WARN_PCT of daily range used today
            _adr_flag = False
            _d1_today = mt5.copy_rates_from_pos(_sym, mt5.TIMEFRAME_D1, 0, 2)
            if _d1_today is not None and len(_d1_today) >= 1 and _adr and _adr > 0:
                _today_high = float(_d1_today["high"][-1])
                _today_low  = float(_d1_today["low"][-1])
                _used_range = _today_high - _today_low
                _used_pct   = (_used_range / _adr) * 100.0
                if _used_pct >= cfg.ADR_USED_WARN_PCT:
                    _flags.append("ADR")
                    _adr_flag = True

            _flags_str = " ".join(_flags) if _flags else "—"

            # ── AUTO-CLOSE ACTIONS ────────────────────────────────────────────
            _action    = "Hold"
            _auto_done = False

            # TME: auto-close if age > max bars AND R < min R
            if (not _auto_done
                    and _age_bars >= cfg.TME_MAX_BARS
                    and _r < cfg.TME_MIN_R
                    and _info):
                _res = mt5.order_send({
                    "action":   mt5.TRADE_ACTION_DEAL,
                    "position": _ticket,
                    "symbol":   _sym,
                    "volume":   _lots,
                    "type":     mt5.ORDER_TYPE_SELL if _side == "BUY" else mt5.ORDER_TYPE_BUY,
                    "price":    _price,
                    "deviation": 20,
                    "comment":  "FM_TME_autoclose",
                })
                if _res and _res.retcode == mt5.TRADE_RETCODE_DONE:
                    _action    = f"✅ TME closed (age={_age_bars}h R={_r})"
                    _auto_done = True
                    _auto_log.append(f"{_sym} TME→closed")
                    # Remove from cooldown since position is gone
                    _cd_now = {k: v for k, v in {}.items()}
                    set_cooldown(_cd_now)
                else:
                    _action = f"⚠ TME close failed (rc={_res.retcode if _res else '?'})"

            # Trail SL at TP2
            if not _auto_done and _tp2_hit and _info:
                _need_trail = (
                    (_side == "BUY"  and (_sl is None or _sl == 0 or _trail > _sl)) or
                    (_side == "SELL" and (_sl is None or _sl == 0 or _trail < _sl))
                )
                if _need_trail:
                    _res = mt5.order_send({
                        "action":   mt5.TRADE_ACTION_SLTP,
                        "position": _ticket,
                        "symbol":   _sym,
                        "sl":       _trail,
                        "tp":       _pos.tp,
                    })
                    if _res and _res.retcode == mt5.TRADE_RETCODE_DONE:
                        _action    = f"✅ Trailed → {_trail}"
                        _auto_done = True
                        _auto_log.append(f"{_sym} trail→{_trail}")
                    else:
                        _action = f"⚠ Trail failed (rc={_res.retcode if _res else '?'})"

            # Move SL to BE at TP1
            if not _auto_done and _tp1_hit and _info:
                _need_be = (
                    (_side == "BUY"  and (_sl is None or _sl == 0 or _sl < _entry)) or
                    (_side == "SELL" and (_sl is None or _sl == 0 or _sl > _entry))
                )
                if _need_be:
                    _res = mt5.order_send({
                        "action":   mt5.TRADE_ACTION_SLTP,
                        "position": _ticket,
                        "symbol":   _sym,
                        "sl":       _entry,
                        "tp":       _pos.tp,
                    })
                    if _res and _res.retcode == mt5.TRADE_RETCODE_DONE:
                        _action    = f"✅ BE → {_entry}"
                        _auto_done = True
                        _auto_log.append(f"{_sym} BE→{_entry}")
                    else:
                        _action = f"⚠ BE failed (rc={_res.retcode if _res else '?'})"

            # Human action suggestions (no auto-close)
            if not _auto_done:
                if _tp2_hit:
                    _action = "Trail final 25% (manual)"
                elif _tp1_hit:
                    _action = "Close 25% + BE (manual)"
                elif _flags:
                    _action = f"Review: {_flags_str}"
                elif _r >= 0.8:
                    _action = "Approaching TP1"
                elif _r < -0.5:
                    _action = "Under pressure"

            _rows.append({
                "Symbol":   _sym,      "Side":      _side,
                "Lots":     _lots,     "Entry":     round(_entry, 5),
                "Now":      round(_price, 5),
                "Age(h)":   _age_bars, "R":         _r,
                "SL pips":  _sl_pips,
                "SL":       round(_sl, 5) if _sl else "—",
                "TP1(50%)": _tp1_p,   "TP2(25%)":  _tp2_p,
                "Trail":    _trail,
                "TP1":      "✅" if _tp1_hit else "—",
                "TP2":      "✅" if _tp2_hit else "—",
                "P&L":      round(_pnl, 2),
                "Flags":    _flags_str,
                "Action":   _action,
            })

        _log  = " · ".join(_auto_log) if _auto_log else "No auto-actions."
        _mgmt = mo.vstack([
            mo.md(
                f"### Open Positions · {_now_ts.strftime('%H:%M:%S')} UTC · "
                f"Auto: {_log}"
            ),
            mo.md(
                f"SL={cfg.SL_ADR_FRACTION*100:.0f}%ADR · "
                f"TP1={cfg.TP1_ADR_FRACTION*100:.0f}%ADR (close 50% manually) · "
                f"TP2={cfg.TP2_ADR_FRACTION*100:.0f}%ADR (close 25% manually) · "
                f"Trail={cfg.TRAIL_ADR_FRACTION*100:.0f}%ADR auto  |  "
                f"**Flags:** RSI=momentum diverging · ATR=compression · ADR=range exhausted  |  "
                f"**TME auto-close:** age > {cfg.TME_MAX_BARS}h AND R < {cfg.TME_MIN_R}"
            ),
            mo.ui.table(_rows, pagination=False),
        ])

    _mgmt
    return


if __name__ == "__main__":
    app.run()
