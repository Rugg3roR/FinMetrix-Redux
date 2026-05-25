import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


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


@app.cell
def _(ThreadPoolExecutor, cfg, ft, je, mh, mo, mt5, np, pl):
    """
    Runs once. No refresh_timer dependency.

    Fetches D1 bars (used for ADR sizing).
    Trains model_long and model_short — shown as ML prob in the
    dashboard for reference. Not gating signals.
    """
    with mo.status.spinner("Booting FinMetrix..."):
        from modules.data_engine import get_mt5_df

        # ── ADR — 14-day average daily range ─────────────────────────────────
        def _fetch_adr(symbol):
            _rates = mt5.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_D1, 0, cfg.ADR_LOOKBACK_DAYS
            )
            if _rates is None or len(_rates) == 0:
                return symbol, None, None
            _info = mt5.symbol_info(symbol)
            if _info is None:
                return symbol, None, None
            _pip      = _info.point * (10.0 if _info.digits in [3, 5] else 1.0)
            _df       = pl.DataFrame(_rates)
            _adr_price = float((_df["high"] - _df["low"]).mean())
            _adr_pips  = round(_adr_price / _pip, 1)
            return symbol, _adr_price, _adr_pips

        with ThreadPoolExecutor(max_workers=8) as _ex:
            _adr_res = list(_ex.map(_fetch_adr, cfg.SYMBOLS))

        adr_map      = {}
        adr_pips_map = {}
        for _sym, _ap, _apips in _adr_res:
            if _ap is not None:
                adr_map[_sym]      = _ap
                adr_pips_map[_sym] = _apips

        # ── Model training ────────────────────────────────────────────────────
        _long_rows  = []
        _short_rows = []

        for _symbol in cfg.SYMBOLS:
            _df = get_mt5_df(_symbol, cfg.TIMEFRAME, cfg.BARS)
            if _df is None:
                continue
            _df = ft.engineer_features(_df)
            _df = ft.triple_barrier(
                _df,
                tp_mult=cfg.TP1_MULT,
                sl_mult=cfg.SL_MULT,
                max_hold=cfg.MAX_HOLD,
            )
            _df_long = _df.filter(
                (pl.col("ema144_slope") > 0) &
                (pl.col("adx")          > cfg.ADX_MIN)
            ).select([*cfg.FEATURES, "target_long"]).drop_nulls()

            _df_short = _df.filter(
                (pl.col("ema144_slope") < 0) &
                (pl.col("adx")          > cfg.ADX_MIN)
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
    | **Signal logic** | H1 SMA cross · D1 SMA cross · RSI MA cross — all three must agree |
    | **ADR symbols** | {len(adr_map)} / {len(cfg.SYMBOLS)} |
    | **Long training rows** | {_n_long:,} |
    | **Short training rows** | {_n_short:,} |
    | **Kelly risk** | {kelly_risk_pct*100:.2f}% ({_kelly_src}) |
    | **SL / TP1 / TP2** | {cfg.SL_ADR_FRACTION*100:.0f}% / {cfg.TP1_ADR_FRACTION*100:.0f}% / {cfg.TP2_ADR_FRACTION*100:.0f}% of {cfg.ADR_LOOKBACK_DAYS}-day ADR |
    | **Cooldown** | {cfg.COOLDOWN_BARS}h after signal fires |
    """)
    return adr_map, adr_pips_map, kelly_risk_pct, model_long, model_short


@app.cell
def _(
    ThreadPoolExecutor,
    adr_map,
    adr_pips_map,
    balance,
    cfg,
    datetime,
    de,
    ft,
    get_cooldown,
    kelly_risk_pct,
    model_long,
    model_short,
    mt5,
    np,
    refresh_timer,
    set_cooldown,
    timezone,
):
    refresh_timer.value

    # ══════════════════════════════════════════════════════════════
    # SIGNAL PARAMETERS
    # Periods match the overlay exactly.
    # ══════════════════════════════════════════════════════════════
    _H1_SMA_FAST  =  5    # fast SMA on H1 close prices
    _H1_SMA_SLOW  = 25    # slow SMA on H1 close prices
    _D1_SMA_FAST  =  5    # fast SMA on D1 close prices
    _D1_SMA_SLOW  = 25    # slow SMA on D1 close prices
    _RSI_PERIOD   =  9    # RSI calculation period
    _RSI_MA_FAST  =  4    # fast MA of RSI
    _RSI_MA_SLOW  = 21    # slow MA of RSI

    # ══════════════════════════════════════════════════════════════
    # FETCH H1 BARS (raw OHLCV — no feature engineering needed)
    # We compute our own SMAs from close prices directly.
    # ══════════════════════════════════════════════════════════════
    def _fetch_h1(symbol):
        _rates = mt5.copy_rates_from_pos(symbol, cfg.TIMEFRAME, 0, 300)
        if _rates is None or len(_rates) == 0:
            return None
        return symbol, _rates

    # ══════════════════════════════════════════════════════════════
    # FETCH D1 BARS (raw OHLCV — only need close prices for SMAs)
    # ══════════════════════════════════════════════════════════════
    def _fetch_d1(symbol):
        _rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 60)
        if _rates is None or len(_rates) == 0:
            return None
        return symbol, _rates

    with ThreadPoolExecutor(max_workers=8) as _ex:
        _h1_res = list(_ex.map(_fetch_h1, cfg.SYMBOLS))
        _d1_res = list(_ex.map(_fetch_d1, cfg.SYMBOLS))

    _h1_map = {sym: r for sym, r in _h1_res if r is not None} if _h1_res else {}
    _d1_map = {sym: r for sym, r in _d1_res if r is not None} if _d1_res else {}

    # ── Candle-close guard ────────────────────────────────────────
    scan_market = True
    if _h1_map:
        _ref_sym   = "EURUSD" if "EURUSD" in _h1_map else next(iter(_h1_map))
        _last_time = int(_h1_map[_ref_sym]["time"][-1])
        _last_str  = str(_last_time)
        if cfg.LAST_CANDLE_FILE.exists():
            if _last_str == cfg.LAST_CANDLE_FILE.read_text().strip():
                scan_market = False
        if scan_market:
            cfg.LAST_CANDLE_FILE.write_text(_last_str)

    # ── Cooldown decay on new candle ──────────────────────────────
    _cd = get_cooldown()
    if scan_market:
        _cd = {s: c - 1 for s, c in _cd.items() if c > 1}
        set_cooldown(_cd)

    # ── Currency exposure filter ──────────────────────────────────
    _positions    = mt5.positions_get() or []
    _open_symbols = {p.symbol for p in _positions}
    _exposed_curr = {c for sym in _open_symbols for c in (sym[:3], sym[3:6])}

    def _is_blocked(symbol):
        if not _open_symbols:
            return False
        return symbol[:3] in _exposed_curr or symbol[3:6] in _exposed_curr

    # ══════════════════════════════════════════════════════════════
    # SMA HELPER
    # Returns the SMA of the last `period` values in arr.
    # ══════════════════════════════════════════════════════════════
    def _sma(arr, period):
        if len(arr) < period:
            return None
        return float(np.mean(arr[-period:]))

    # ══════════════════════════════════════════════════════════════
    # RSI HELPER
    # Wilder smoothed RSI — calculated from scratch on close prices.
    # Returns an array of RSI values aligned with the input closes.
    # ══════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════
    # Also need engineer_features_fast for ML prob column only
    # ══════════════════════════════════════════════════════════════
    def _fetch_features(symbol):
        _df = de.get_mt5_df(symbol, cfg.TIMEFRAME, cfg.BARS)
        if _df is None:
            return None
        return symbol, ft.engineer_features_fast(_df)

    with ThreadPoolExecutor(max_workers=8) as _ex2:
        _feat_res = list(_ex2.map(_fetch_features, cfg.SYMBOLS))
    _feat_map = {sym: df for sym, df in _feat_res if df is not None}

    # ══════════════════════════════════════════════════════════════
    # SCORE EVERY PAIR
    # ══════════════════════════════════════════════════════════════
    signal_rows = []
    today_count = de.signals_today()
    _now_ts     = datetime.now(timezone.utc)

    for symbol in cfg.SYMBOLS:
        if symbol not in _h1_map or symbol not in _d1_map:
            continue

        _h1_closes = _h1_map[symbol]["close"].astype(float)
        _d1_closes = _d1_map[symbol]["close"].astype(float)
        _price     = float(_h1_closes[-1])
        _adr       = adr_map.get(symbol)
        _adr_p     = adr_pips_map.get(symbol, 0.0)

        # ── H1 SMA cross ─────────────────────────────────────────
        _h1_fast = _sma(_h1_closes, _H1_SMA_FAST)
        _h1_slow = _sma(_h1_closes, _H1_SMA_SLOW)
        if _h1_fast is None or _h1_slow is None:
            continue
        _h1_bull = _h1_fast > _h1_slow   # True = bullish, False = bearish

        # ── D1 SMA cross ─────────────────────────────────────────
        _d1_fast = _sma(_d1_closes, _D1_SMA_FAST)
        _d1_slow = _sma(_d1_closes, _D1_SMA_SLOW)
        if _d1_fast is None or _d1_slow is None:
            continue
        _d1_bull = _d1_fast > _d1_slow

        # ── RSI MA cross ─────────────────────────────────────────
        # Calculate RSI from H1 closes, then take fast/slow MA of RSI
        _rsi_arr      = _calc_rsi(_h1_closes, _RSI_PERIOD)
        _rsi_valid    = _rsi_arr[~np.isnan(_rsi_arr)]
        _rsi_ma_fast  = _sma(_rsi_valid, _RSI_MA_FAST)
        _rsi_ma_slow  = _sma(_rsi_valid, _RSI_MA_SLOW)
        if _rsi_ma_fast is None or _rsi_ma_slow is None:
            continue
        _rsi_bull = _rsi_ma_fast > _rsi_ma_slow

        # ── All three must agree ──────────────────────────────────
        _is_long  = _h1_bull  and _d1_bull  and _rsi_bull
        _is_short = (not _h1_bull) and (not _d1_bull) and (not _rsi_bull)

        # ── Cooldown + exposure check ─────────────────────────────
        _corr_blocked = _is_blocked(symbol)
        _in_cooldown  = symbol in _cd
        signal        = "Neutral"

        if _is_long and not _in_cooldown and not _corr_blocked:
            signal = "BUY"
        elif _is_short and not _in_cooldown and not _corr_blocked:
            signal = "SELL"

        # ── ML prob (informational) ───────────────────────────────
        _prob = 0.0
        if symbol in _feat_map:
            _now = _feat_map[symbol].tail(1).to_dicts()[0]
            _X   = np.array([[_now.get(f, 0.0) for f in cfg.FEATURES]])
            if signal == "BUY" and model_long:
                _prob = float(model_long.predict_proba(_X)[0][1])
            elif signal == "SELL" and model_short:
                _prob = float(model_short.predict_proba(_X)[0][1])

        # ── Sizing ────────────────────────────────────────────────
        if _adr and _adr > 0:
            _sl_dist  = _adr * cfg.SL_ADR_FRACTION
            _tp1_dist = _adr * cfg.TP1_ADR_FRACTION
            _tp2_dist = _adr * cfg.TP2_ADR_FRACTION
        else:
            # fallback: use ATR from feature map if available
            _atr_fb   = _feat_map[symbol].tail(1)["atr"][0] if symbol in _feat_map else 0.001
            _sl_dist  = float(_atr_fb) * cfg.SL_MULT
            _tp1_dist = float(_atr_fb) * cfg.TP1_MULT
            _tp2_dist = float(_atr_fb) * cfg.TP2_MULT
            _adr_p    = 0.0

        if signal == "BUY" or (signal == "Neutral" and _h1_bull):
            _dir = "BUY"
        elif signal == "SELL" or (signal == "Neutral" and not _h1_bull):
            _dir = "SELL"
        else:
            _dir = "—"

        if _dir == "BUY":
            _sl  = round(_price - _sl_dist,  5)
            _tp1 = round(_price + _tp1_dist, 5)
            _tp2 = round(_price + _tp2_dist, 5)
        elif _dir == "SELL":
            _sl  = round(_price + _sl_dist,  5)
            _tp1 = round(_price - _tp1_dist, 5)
            _tp2 = round(_price - _tp2_dist, 5)
        else:
            _sl = _tp1 = _tp2 = "—"

        _lot_size = 0.0
        _risk_gbp = 0.0
        _sym_info = mt5.symbol_info(symbol)
        if _sym_info and _sl_dist > 0:
            _sl_ticks = _sl_dist / _sym_info.trade_tick_size
            _lot_size = (balance * kelly_risk_pct) / (_sl_ticks * _sym_info.trade_tick_value)
            _lot_size = max(_sym_info.volume_min,
                            round(_lot_size / _sym_info.volume_step) * _sym_info.volume_step)
            _risk_gbp = round(_lot_size * _sl_ticks * _sym_info.trade_tick_value, 2)

        # ── Log on new candle ─────────────────────────────────────
        if scan_market and signal != "Neutral":
            today_count += 1
            _cd[symbol] = cfg.COOLDOWN_BARS
            set_cooldown(_cd)
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
                "lot_size":    _lot_size,
                "prob":        _prob,
            })

        # ── Status tag ────────────────────────────────────────────
        if _corr_blocked:
            _tag = " [exp]"
        elif _in_cooldown:
            _tag = f" [cd{_cd.get(symbol, 0)}]"
        else:
            _tag = ""

        signal_rows.append({
            "Symbol":   symbol,
            "Signal":   signal + _tag,
            "ADR pips": _adr_p,
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


@app.cell
def _(acc, balance, datetime, kelly_risk_pct, mo, signal_rows, today_count):
    mo.vstack([
        mo.Html("<style>td, th { font-size: 0.9rem !important; }</style>"),
        mo.md(
            f"### {acc.login} · {acc.currency} {balance:,.2f} · "
            f"{datetime.now().strftime('%H:%M:%S')} · "
            f"Signals today: {today_count} · "
            f"Kelly: {kelly_risk_pct*100:.2f}%"
        ),
        mo.md(
            f"H1 SMA({5}/{25}) · D1 SMA({5}/{25}) · RSI({9}) MA({4}/{21}) — all three must agree"
        ),
        mo.ui.table(signal_rows, pagination=False),
    ])
    return


@app.cell
def _(adr_map, cfg, datetime, mo, mt5, refresh_timer, timezone):
    refresh_timer.value

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

            _tick  = mt5.symbol_info_tick(_sym)
            _info  = mt5.symbol_info(_sym)
            _price = (_tick.bid if _side == "BUY" else _tick.ask) if _tick else _entry
            _pip   = (_info.point * (10.0 if _info.digits in [3, 5] else 1.0)) if _info else 0.0001

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

            if _sl_d > 0:
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

                _sl_pips   = round(_sl_d / _pip, 1)
                _action    = "Hold"
                _auto_done = False

                if _tp2_hit and _info:
                    _need = (
                        (_side == "BUY"  and (_sl is None or _sl == 0 or _trail > _sl)) or
                        (_side == "SELL" and (_sl is None or _sl == 0 or _trail < _sl))
                    )
                    if _need:
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

                elif _tp1_hit and not _auto_done and _info:
                    _need = (
                        (_side == "BUY"  and (_sl is None or _sl == 0 or _sl < _entry)) or
                        (_side == "SELL" and (_sl is None or _sl == 0 or _sl > _entry))
                    )
                    if _need:
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

                if not _auto_done:
                    if _tp2_hit:
                        _action = "Trail final 25% (manual)"
                    elif _tp1_hit:
                        _action = "Close 25% + move SL to BE (manual)"
                    elif _r >= 0.8:
                        _action = "Approaching TP1"
                    elif _r < -0.5:
                        _action = "Under pressure"

                _rows.append({
                    "Symbol":   _sym,    "Side":      _side,
                    "Lots":     _lots,   "Entry":     round(_entry, 5),
                    "Now":      round(_price, 5),
                    "R":        _r,      "SL pips":   _sl_pips,
                    "SL":       round(_sl, 5) if _sl else "—",
                    "BE":       round(_entry, 5),
                    "TP1(50%)": _tp1_p,  "TP2(25%)":  _tp2_p,
                    "Trail":    _trail,
                    "TP1 hit":  "✅" if _tp1_hit else "—",
                    "TP2 hit":  "✅" if _tp2_hit else "—",
                    "P&L":      round(_pnl, 2),
                    "Action":   _action,
                })
            else:
                _rows.append({
                    "Symbol":   _sym,  "Side":      _side,
                    "Lots":     _lots, "Entry":     round(_entry, 5),
                    "Now":      round(_price, 5),
                    "R":        "—",   "SL pips":   "—",
                    "SL":       "SET SL FIRST",
                    "BE":       "—",   "TP1(50%)":  "—",
                    "TP2(25%)": "—",   "Trail":     "—",
                    "TP1 hit":  "—",   "TP2 hit":   "—",
                    "P&L":      round(_pnl, 2),
                    "Action":   "SET SL FIRST",
                })

        _log  = " · ".join(_auto_log) if _auto_log else "No auto-actions this tick."
        _mgmt = mo.vstack([
            mo.md(
                f"### Open Positions · {_now_ts.strftime('%H:%M:%S')} UTC · "
                f"Auto: {_log}"
            ),
            mo.md(
                f"SL={cfg.SL_ADR_FRACTION*100:.0f}%ADR · "
                f"TP1={cfg.TP1_ADR_FRACTION*100:.0f}%ADR (close 50% manually) · "
                f"TP2={cfg.TP2_ADR_FRACTION*100:.0f}%ADR (close 25% manually) · "
                f"Trail={cfg.TRAIL_ADR_FRACTION*100:.0f}%ADR auto-managed"
            ),
            mo.ui.table(_rows, pagination=False),
        ])

    _mgmt
    return


if __name__ == "__main__":
    app.run()
