import marimo

__generated_with = "0.23.3"
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

    importlib.reload(cfg)
    importlib.reload(ft)
    importlib.reload(de)
    importlib.reload(mh)
    importlib.reload(je)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    acc     = mt5.account_info()
    balance = acc.balance if acc else 0.0

    refresh_timer = mo.ui.refresh(
        options=["5m", "10m", "15m", "30m"],
        default_interval="5m"
    )
    refresh_timer
    return (
        ThreadPoolExecutor,
        acc,
        balance,
        cfg,
        datetime,
        de,
        ft,
        je,
        mh,
        mo,
        mt5,
        np,
        pl,
        refresh_timer,
        timezone,
    )


@app.cell
def _(ThreadPoolExecutor, cfg, ft, je, mh, mo, mt5, np, pl):
    """Boot: fetch ADR, train models. No timer dependency — runs once."""

    with mo.status.spinner("Fetching ADR + training models..."):
        from modules.data_engine import get_mt5_df

        # ── ADR fetch ─────────────────────────────────────────────────────────
        def _fetch_adr(symbol):
            rates = mt5.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_D1, 0, cfg.ADR_LOOKBACK_DAYS
            )
            if rates is None or len(rates) == 0:
                return symbol, None, None
            info = mt5.symbol_info(symbol)
            if info is None:
                return symbol, None, None
            pip_size  = info.point * (10.0 if info.digits in [3, 5] else 1.0)
            df        = pl.DataFrame(rates)
            adr_price = float((df["high"] - df["low"]).mean())
            adr_pips  = round(adr_price / pip_size, 1)
            return symbol, adr_price, adr_pips

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
        _rsi_l_min  = cfg.RSI_MID + cfg.RSI_BUFFER
        _rsi_s_max  = cfg.RSI_MID - cfg.RSI_BUFFER

        for _symbol in cfg.SYMBOLS:
            _df = get_mt5_df(_symbol, cfg.TIMEFRAME, cfg.BARS)
            if _df is None:
                continue
            _df = ft.engineer_features(_df)      # full — Hurst computed
            _df = ft.triple_barrier(
                _df,
                tp_mult=cfg.TP1_MULT,
                sl_mult=cfg.SL_MULT,
                max_hold=cfg.MAX_HOLD,
            )

            # Training filter — identical to scanner signal logic
            # Gate 1: trend confirmation
            # Gate 2: predictive alignment
            # Gate 3: RSI MA
            # Gate 4: volume
            _df_long = _df.filter(
                (pl.col("ema144_slope")     >  cfg.EMA144_SLOPE_MIN) &
                (pl.col("pred_alignment")   >= cfg.PRED_ALIGN_LONG) &
                (pl.col("pred_align_delta") >= cfg.PRED_ALIGN_DELTA_MIN) &
                (pl.col("rsi_ma")           >  _rsi_l_min) &
                (pl.col("vol_ratio")        >  1.0)
            ).select([*cfg.FEATURES, "target_long"]).drop_nulls()

            _df_short = _df.filter(
                (pl.col("ema144_slope")     <  -cfg.EMA144_SLOPE_MIN) &
                (pl.col("pred_alignment")   <= cfg.PRED_ALIGN_SHORT) &
                (pl.col("pred_align_delta") <= -cfg.PRED_ALIGN_DELTA_MIN) &
                (pl.col("rsi_ma")           <  _rsi_s_max) &
                (pl.col("vol_ratio")        >  1.0)
            ).select([*cfg.FEATURES, "target_short"]).drop_nulls()

            if _df_long.height > 0:
                _long_rows.append(_df_long.cast(pl.Float64).to_numpy())
            if _df_short.height > 0:
                _short_rows.append(_df_short.cast(pl.Float64).to_numpy())

        model_long  = None
        _data_l     = np.zeros((0, len(cfg.FEATURES) + 1))
        if _long_rows:
            _data_l    = np.vstack(_long_rows)
            model_long = mh.train_model(_data_l[:, :-1], _data_l[:, -1])

        model_short = None
        _data_s     = np.zeros((0, len(cfg.FEATURES) + 1))
        if _short_rows:
            _data_s     = np.vstack(_short_rows)
            model_short = mh.train_model(_data_s[:, :-1], _data_s[:, -1])

    kelly_risk_pct = je.calculate_adaptive_risk()
    _kelly_src     = "adaptive" if kelly_risk_pct > cfg.KELLY_MIN_FLOOR else "floor"

    _adr_rows = sorted([
        {
            "Symbol":    s,
            "ADR pips":  adr_pips_map[s],
            "SL pips":   round(adr_pips_map[s] * cfg.SL_ADR_FRACTION,  1),
            "TP1 pips":  round(adr_pips_map[s] * cfg.TP1_ADR_FRACTION, 1),
            "TP2 pips":  round(adr_pips_map[s] * cfg.TP2_ADR_FRACTION, 1),
        }
        for s in adr_pips_map
    ], key=lambda x: x["ADR pips"], reverse=True)

    mo.vstack([
        mo.md(f"""
    ✅ **Boot complete**
    | | |
    | :--- | :--- |
    | **Long training rows** | {len(_data_l):,} |
    | **Short training rows** | {len(_data_s):,} |
    | **ADR symbols loaded** | {len(adr_map)} / {len(cfg.SYMBOLS)} |
    | **Kelly risk** | {kelly_risk_pct*100:.2f}% ({_kelly_src}) |
    | **Trend gate** | ema144 slope > {cfg.EMA144_SLOPE_MIN} (BUY) / < -{cfg.EMA144_SLOPE_MIN} (SELL) |
    | **Features** | {', '.join(cfg.FEATURES)} |
    | **SL / TP1 / TP2** | {cfg.SL_ADR_FRACTION*100:.0f}% / {cfg.TP1_ADR_FRACTION*100:.0f}% / {cfg.TP2_ADR_FRACTION*100:.0f}% of ADR |
        """),
        mo.md("**Per-symbol SL/TP levels:**"),
        mo.ui.table(_adr_rows, pagination=False),
    ])
    return adr_map, adr_pips_map, kelly_risk_pct, model_long, model_short


@app.cell
def _(
    ThreadPoolExecutor,
    adr_map,
    balance,
    cfg,
    datetime,
    de,
    ft,
    kelly_risk_pct,
    model_long,
    model_short,
    mt5,
    np,
    pl,
    refresh_timer,
    timezone,
):
    refresh_timer.value

    def _fetch(symbol):
        df = de.get_mt5_df(symbol, cfg.TIMEFRAME, cfg.BARS)
        if df is None:
            return None
        df = ft.engineer_features_fast(df)
        return symbol, df

    with ThreadPoolExecutor(max_workers=8) as _ex:
        _raw = list(_ex.map(_fetch, cfg.SYMBOLS))
    results = [r for r in _raw if r is not None]

    # ── Candle-close guard ────────────────────────────────────────────────────
    scan_market = True
    if results:
        _ref = next((df for s, df in results if s == "EURUSD"), results[0][1])
        _last_candle = _ref.tail(1)["time"][0]
        if cfg.LAST_CANDLE_FILE.exists():
            if str(_last_candle) == cfg.LAST_CANDLE_FILE.read_text().strip():
                scan_market = False
        if scan_market:
            cfg.LAST_CANDLE_FILE.write_text(str(_last_candle))

    # ── Cooldown tracker ──────────────────────────────────────────────────────
    try:
        _cooldown_state
    except NameError:
        _cooldown_state = {}
    if scan_market:
        _cooldown_state = {s: c - 1 for s, c in _cooldown_state.items() if c > 1}

    # ── Correlation filter: block signals correlated with open positions ───────
    # Compute correlation matrix from current results (M30 already in memory)
    # then check each candidate against all open positions.
    _open_positions = mt5.positions_get() or []
    _open_symbols   = {p.symbol for p in _open_positions}

    _corr_map = {}   # {symbol: {other_symbol: r}}
    if _open_symbols and len(results) > 1:
        _price_dfs = [
            _df.tail(14 * 48).select([
                pl.col("time"), pl.col("close").alias(_sym)
            ])
            for _sym, _df in results
        ]
        _combined = _price_dfs[0]
        for _nxt in _price_dfs[1:]:
            _combined = _combined.join(_nxt, on="time", how="inner")
        _corr_syms = [c for c in _combined.columns if c != "time"]
        _rets      = _combined.select([
            pl.col(s).pct_change() for s in _corr_syms
        ]).drop_nulls()
        _corr_np = _rets.corr().to_numpy()
        for _i, _s in enumerate(_corr_syms):
            _corr_map[_s] = {
                _corr_syms[_j]: _corr_np[_i, _j]
                for _j in range(len(_corr_syms)) if _j != _i
            }

    def _is_corr_blocked(symbol):
        """True if |r| > threshold with any currently open position."""
        if not _open_symbols:
            return False
        sym_corrs = _corr_map.get(symbol, {})
        for open_sym in _open_symbols:
            r = sym_corrs.get(open_sym, 0.0)
            if abs(r) > cfg.CORR_BLOCK_THRESHOLD:
                return True
        return False

    # ── Thresholds ────────────────────────────────────────────────────────────
    _rsi_l_min = cfg.RSI_MID + cfg.RSI_BUFFER
    _rsi_s_max = cfg.RSI_MID - cfg.RSI_BUFFER

    def _signal_age(df, direction):
        tail = df.tail(cfg.MAX_SIGNAL_AGE + 2)
        n    = tail.height
        if direction == "long":
            cond = (
                (tail["ema144_slope"].to_numpy()     >  cfg.EMA144_SLOPE_MIN) &
                (tail["pred_alignment"].to_numpy()   >= cfg.PRED_ALIGN_LONG) &
                (tail["pred_align_delta"].to_numpy() >= cfg.PRED_ALIGN_DELTA_MIN) &
                (tail["rsi_ma"].to_numpy()           >  _rsi_l_min) &
                (tail["vol_ratio"].to_numpy()        >  1.0)
            )
        else:
            cond = (
                (tail["ema144_slope"].to_numpy()     <  -cfg.EMA144_SLOPE_MIN) &
                (tail["pred_alignment"].to_numpy()   <= cfg.PRED_ALIGN_SHORT) &
                (tail["pred_align_delta"].to_numpy() <= -cfg.PRED_ALIGN_DELTA_MIN) &
                (tail["rsi_ma"].to_numpy()           <  _rsi_s_max) &
                (tail["vol_ratio"].to_numpy()        >  1.0)
            )
        if not cond[-1]:
            return -1
        run_start = n - 1
        for i in range(n - 2, -1, -1):
            if cond[i]:
                run_start = i
            else:
                break
        return (n - 1) - run_start

    # ── Score every pair ──────────────────────────────────────────────────────
    candidates  = []
    today_count = de.signals_today()

    for symbol, df in results:
        now  = df.tail(1).to_dicts()[0]
        X    = np.array([[now.get(f, 0.0) for f in cfg.FEATURES]])

        prob_l = float(model_long.predict_proba(X)[0][1])  if model_long  else 0.0
        prob_s = float(model_short.predict_proba(X)[0][1]) if model_short else 0.0

        ema144_slope = now.get("ema144_slope", 0.0)
        pred_align   = now.get("pred_alignment",   2.0)
        pred_delta   = now.get("pred_align_delta", 0.0)
        rsi_ma       = now.get("rsi_ma",   50.0)
        vol_ratio    = now.get("vol_ratio", 1.0)

        # Individual condition flags
        c_trend_l = ema144_slope >  cfg.EMA144_SLOPE_MIN
        c_trend_s = ema144_slope < -cfg.EMA144_SLOPE_MIN
        c_align_l = pred_align   >= cfg.PRED_ALIGN_LONG
        c_align_s = pred_align   <= cfg.PRED_ALIGN_SHORT
        c_delta_l = pred_delta   >= cfg.PRED_ALIGN_DELTA_MIN
        c_delta_s = pred_delta   <= -cfg.PRED_ALIGN_DELTA_MIN
        c_rsi_l   = rsi_ma       >  _rsi_l_min
        c_rsi_s   = rsi_ma       <  _rsi_s_max
        c_vol     = vol_ratio    >  1.0

        is_long  = c_trend_l and c_align_l and c_delta_l and c_rsi_l and c_vol
        is_short = c_trend_s and c_align_s and c_delta_s and c_rsi_s and c_vol

        long_score  = sum([c_trend_l, c_align_l, c_delta_l, c_rsi_l, c_vol])
        short_score = sum([c_trend_s, c_align_s, c_delta_s, c_rsi_s, c_vol])

        corr_blocked = _is_corr_blocked(symbol)
        in_cooldown  = symbol in _cooldown_state

        signal = "Neutral"
        prob   = max(prob_l, prob_s)
        age    = -1

        if is_long and not in_cooldown and not corr_blocked:
            age = _signal_age(df, "long")
            if 0 <= age <= cfg.MAX_SIGNAL_AGE and prob_l >= cfg.MIN_PROB:
                signal = "BUY"
                prob   = prob_l
        elif is_short and not in_cooldown and not corr_blocked:
            age = _signal_age(df, "short")
            if 0 <= age <= cfg.MAX_SIGNAL_AGE and prob_s >= cfg.MIN_PROB:
                signal = "SELL"
                prob   = prob_s

        if age == -1:
            age = _signal_age(df, "long" if is_long else "short") if (is_long or is_short) else -1

        candidates.append({
            "symbol": symbol, "signal": signal,
            "prob_l": prob_l, "prob_s": prob_s, "prob": prob, "age": age,
            "long_score": long_score, "short_score": short_score,
            "c_trend_l": c_trend_l, "c_trend_s": c_trend_s,
            "c_align_l": c_align_l, "c_align_s": c_align_s,
            "c_delta_l": c_delta_l, "c_delta_s": c_delta_s,
            "c_rsi_l": c_rsi_l, "c_rsi_s": c_rsi_s, "c_vol": c_vol,
            "corr_blocked": corr_blocked, "in_cooldown": in_cooldown,
            "ema144_slope": ema144_slope, "now": now,
        })

    # ── Build display rows + log signals ─────────────────────────────────────
    signal_stats = []
    _now_ts      = datetime.now(timezone.utc)
    _log_rows    = []

    for c in candidates:
        symbol       = c["symbol"]
        now          = c["now"]
        signal       = c["signal"]
        prob         = c["prob"]
        age          = c["age"]
        pred_align   = now.get("pred_alignment",   2.0)
        pred_delta   = now.get("pred_align_delta", 0.0)
        rsi_ma       = now.get("rsi_ma",   50.0)
        vol_ratio    = now.get("vol_ratio", 1.0)
        ema144_slope = c["ema144_slope"]

        # ADR-based SL/TP
        adr = adr_map.get(symbol)
        if adr and adr > 0:
            sl_dist  = adr * cfg.SL_ADR_FRACTION
            tp1_dist = adr * cfg.TP1_ADR_FRACTION
            tp2_dist = adr * cfg.TP2_ADR_FRACTION
            src      = "ADR"
        else:
            _atr     = now.get("atr", 0.001)
            sl_dist  = _atr * cfg.SL_MULT
            tp1_dist = _atr * cfg.TP1_MULT
            tp2_dist = _atr * cfg.TP2_MULT
            src      = "ATR"

        if signal == "BUY":
            target_sl  = round(now["close"] - sl_dist,  5)
            target_tp1 = round(now["close"] + tp1_dist, 5)
            target_tp2 = round(now["close"] + tp2_dist, 5)
        elif signal == "SELL":
            target_sl  = round(now["close"] + sl_dist,  5)
            target_tp1 = round(now["close"] - tp1_dist, 5)
            target_tp2 = round(now["close"] - tp2_dist, 5)
        else:
            target_sl = target_tp1 = target_tp2 = "-"

        lot_size = 0.0
        risk_gbp = 0.0
        sl_pips  = 0.0

        info = mt5.symbol_info(symbol)
        if info and signal != "Neutral" and sl_dist > 0:
            pip_size    = info.point * (10.0 if info.digits in [3, 5] else 1.0)
            sl_pips     = round(sl_dist / pip_size, 1)
            sl_in_ticks = sl_dist / info.trade_tick_size
            lot_size    = (balance * kelly_risk_pct) / (sl_in_ticks * info.trade_tick_value)
            lot_size    = max(info.volume_min,
                              round(lot_size / info.volume_step) * info.volume_step)
            risk_gbp    = lot_size * sl_in_ticks * info.trade_tick_value

        if scan_market and signal != "Neutral":
            today_count += 1
            _cooldown_state[symbol] = cfg.COOLDOWN_BARS
            de.log_signal({
                "time": _now_ts, "candle_time": now["time"],
                "symbol": symbol, "signal": signal,
                "entry": now["close"], "sl": target_sl,
                "tp1": target_tp1, "tp2": target_tp2,
                "sl_pips": sl_pips, "lot_size": lot_size,
                "prob": prob, "signal_age": age, "sizing_src": src,
                **{f: now.get(f, 0.0) for f in cfg.FEATURES},
            })

        # Status string
        if c["corr_blocked"]:
            status = "corr"
        elif c["in_cooldown"]:
            status = f"cd{_cooldown_state.get(symbol, 0)}"
        else:
            status = ""

        trend_str = ("▲" if ema144_slope > 0 else "▼") + f"{ema144_slope:+.5f}"
        score_str = (
            f"{c['long_score']}/5 L"
            if c["long_score"] >= c["short_score"]
            else f"{c['short_score']}/5 S"
        )
        age_str = "—" if age < 0 else ("new" if age == 0 else f"{age}b")

        # Condition flags — now 5 gates
        flags_l = "".join([
            "T" if c["c_trend_l"] else ".",
            "A" if c["c_align_l"] else ".",
            "D" if c["c_delta_l"] else ".",
            "R" if c["c_rsi_l"]   else ".",
            "V" if c["c_vol"]     else ".",
        ])
        flags_s = "".join([
            "T" if c["c_trend_s"] else ".",
            "A" if c["c_align_s"] else ".",
            "D" if c["c_delta_s"] else ".",
            "R" if c["c_rsi_s"]   else ".",
            "V" if c["c_vol"]     else ".",
        ])

        signal_stats.append({
            "Symbol":    symbol,
            "Signal":    f"{signal} {status}".strip(),
            "Prob":      f"{prob*100:.1f}%",
            "Age":       age_str,
            "Score":     score_str,
            "Trend144":  trend_str,
            "PredAlign": int(pred_align),
            "Δalign":    int(pred_delta),
            "RSI MA":    round(rsi_ma, 2),
            "VolRatio":  round(vol_ratio, 3),
            "Lot":       round(lot_size, 2),
            "Risk GBP":  round(risk_gbp, 2),
            "SL pips":   sl_pips if signal != "Neutral" else "—",
            "Price":     round(now["close"], 5),
            "SL":        target_sl,
            "TP1 (50%)": target_tp1,
            "TP2 (25%)": target_tp2,
            "Gates L":   flags_l,
            "Gates S":   flags_s,
        })

        _log_rows.append({
            "ts": str(_now_ts), "symbol": symbol,
            "signal": signal, "sl_pips": sl_pips,
            "score": score_str, "age": age,
            "trend_slope": round(float(ema144_slope), 6),
            "pa": int(pred_align), "pd": round(float(pred_delta), 2),
            "rsi_ma": round(rsi_ma, 2), "vol_ratio": round(vol_ratio, 3),
            "prob_l": round(c["prob_l"], 4), "prob_s": round(c["prob_s"], 4),
            "corr_blocked": c["corr_blocked"], "cooldown": _cooldown_state.get(symbol, 0),
        })

    _new_batch = pl.DataFrame(_log_rows)
    _log_path  = cfg.DATA_DIR / "indicator_log.csv"
    if _log_path.exists():
        try:
            _existing = pl.read_csv(_log_path, try_parse_dates=False)
            _combined = pl.concat([_existing, _new_batch], how="diagonal")
        except Exception:
            _combined = _new_batch
    else:
        _combined = _new_batch
    _combined.write_csv(_log_path)

    signal_stats.sort(key=lambda x: (
        x["Signal"].split()[0] not in ("BUY", "SELL"),
        -float(x["Prob"][:-1])
    ))
    return results, signal_stats, today_count


@app.cell
def _(adr_map, cfg, datetime, mo, mt5, refresh_timer, timezone):
    """Position manager — refreshes every tick, auto-manages SL."""
    refresh_timer.value   # <-- this is what was missing before

    _positions = mt5.positions_get() or []
    _now_ts    = datetime.now(timezone.utc)
    _auto_log  = []

    if not _positions:
        _mgmt_output = mo.md("*No open positions.*")
    else:
        _rows = []
        for _pos in _positions:
            _sym   = _pos.symbol
            _side  = "BUY" if _pos.type == 0 else "SELL"
            _entry = _pos.price_open
            _sl    = _pos.sl
            _lots  = _pos.volume
            _pnl   = _pos.profit
            _ticket= _pos.ticket

            _tick  = mt5.symbol_info_tick(_sym)
            _info  = mt5.symbol_info(_sym)
            _price = (_tick.bid if _side == "BUY" else _tick.ask) if _tick else _entry
            _pip   = (_info.point * (10.0 if _info.digits in [3, 5] else 1.0)) if _info else 0.0001

            _adr = adr_map.get(_sym)
            if _adr and _adr > 0:
                _sl_dist    = _adr * cfg.SL_ADR_FRACTION
                _tp1_dist   = _adr * cfg.TP1_ADR_FRACTION
                _tp2_dist   = _adr * cfg.TP2_ADR_FRACTION
                _trail_dist = _adr * cfg.TRAIL_ADR_FRACTION
            else:
                _sl_dist    = abs(_entry - _sl) if _sl and _sl != 0 else 0.001
                _tp1_dist   = _sl_dist
                _tp2_dist   = _sl_dist * 2
                _trail_dist = _sl_dist * 0.8

            if _sl_dist > 0:
                if _side == "BUY":
                    _tp1_p    = round(_entry + _tp1_dist, 5)
                    _tp2_p    = round(_entry + _tp2_dist, 5)
                    _be       = _entry
                    _trail_sl = round(max(_price - _trail_dist, _entry), 5)
                    _r_now    = round((_price - _entry) / _sl_dist, 2)
                    _tp1_hit  = _price >= _tp1_p
                    _tp2_hit  = _price >= _tp2_p
                else:
                    _tp1_p    = round(_entry - _tp1_dist, 5)
                    _tp2_p    = round(_entry - _tp2_dist, 5)
                    _be       = _entry
                    _trail_sl = round(min(_price + _trail_dist, _entry), 5)
                    _r_now    = round((_entry - _price) / _sl_dist, 2)
                    _tp1_hit  = _price <= _tp1_p
                    _tp2_hit  = _price <= _tp2_p

                _sl_pips = round(_sl_dist / _pip, 1)

                # ── Auto SL management ────────────────────────────────────────
                # After TP2: trail the stop if trail_sl is better than current SL
                # After TP1: move SL to breakeven if current SL is still below entry
                # Both actions use mt5.order_send() TRADE_ACTION_SLTP
                _action    = "Hold"
                _auto_done = False

                if _tp2_hit and _info:
                    # Trail the final 25% — only move SL in favourable direction
                    _need_trail = (
                        (_side == "BUY"  and (_sl is None or _sl == 0 or _trail_sl > _sl)) or
                        (_side == "SELL" and (_sl is None or _sl == 0 or _trail_sl < _sl))
                    )
                    if _need_trail:
                        _res = mt5.order_send({
                            "action":   mt5.TRADE_ACTION_SLTP,
                            "position": _ticket,
                            "symbol":   _sym,
                            "sl":       _trail_sl,
                            "tp":       _pos.tp,
                        })
                        if _res and _res.retcode == mt5.TRADE_RETCODE_DONE:
                            _action    = f"✅ Trailed SL -> {_trail_sl}"
                            _auto_done = True
                            _auto_log.append(f"{_sym} trail -> {_trail_sl}")
                        else:
                            _action = f"⚠ Trail failed (rc={_res.retcode if _res else '?'})"

                elif _tp1_hit and not _auto_done and _info:
                    # Move SL to breakeven
                    _need_be = (
                        (_side == "BUY"  and (_sl is None or _sl == 0 or _sl < _be)) or
                        (_side == "SELL" and (_sl is None or _sl == 0 or _sl > _be))
                    )
                    if _need_be:
                        _res = mt5.order_send({
                            "action":   mt5.TRADE_ACTION_SLTP,
                            "position": _ticket,
                            "symbol":   _sym,
                            "sl":       _be,
                            "tp":       _pos.tp,
                        })
                        if _res and _res.retcode == mt5.TRADE_RETCODE_DONE:
                            _action    = f"✅ BE set -> {_be}"
                            _auto_done = True
                            _auto_log.append(f"{_sym} BE -> {_be}")
                        else:
                            _action = f"⚠ BE failed (rc={_res.retcode if _res else '?'})"

                if not _auto_done:
                    if _tp2_hit:
                        _action = "Trail final 25% (manual)"
                    elif _tp1_hit:
                        _action = "Close 25% more — move SL to BE (manual)"
                    elif _r_now >= 0.8:
                        _action = "Approaching TP1"
                    elif _r_now < -0.5:
                        _action = "Under pressure"

                _rows.append({
                    "Symbol":     _sym,
                    "Side":       _side,
                    "Lots":       _lots,
                    "Entry":      round(_entry, 5),
                    "Now":        round(_price, 5),
                    "R":          _r_now,
                    "SL pips":    _sl_pips,
                    "SL":         round(_sl, 5) if _sl else "—",
                    "BE":         round(_be, 5),
                    "TP1 (50%)":  _tp1_p,
                    "TP2 (25%)":  _tp2_p,
                    "Trail SL":   _trail_sl,
                    "TP1 hit":    "YES" if _tp1_hit else "—",
                    "TP2 hit":    "YES" if _tp2_hit else "—",
                    "P&L":        round(_pnl, 2),
                    "Action":     _action,
                })
            else:
                _rows.append({
                    "Symbol": _sym, "Side": _side, "Lots": _lots,
                    "Entry": round(_entry, 5), "Now": round(_price, 5),
                    "R": "—", "SL pips": "—", "SL": "SET SL FIRST",
                    "BE": "—", "TP1 (50%)": "—", "TP2 (25%)": "—",
                    "Trail SL": "—", "TP1 hit": "—", "TP2 hit": "—",
                    "P&L": round(_pnl, 2), "Action": "SET SL FIRST",
                })

        _log_str = "  |  ".join(_auto_log) if _auto_log else "No auto-actions this tick."
        _mgmt_output = mo.vstack([
            mo.md(f"### Open Positions — {_now_ts.strftime('%H:%M:%S')} UTC"),
            mo.md(
                f"SL={cfg.SL_ADR_FRACTION*100:.0f}%ADR · "
                f"TP1={cfg.TP1_ADR_FRACTION*100:.0f}%ADR (50%) · "
                f"TP2={cfg.TP2_ADR_FRACTION*100:.0f}%ADR (25%) · "
                f"Trail={cfg.TRAIL_ADR_FRACTION*100:.0f}%ADR  |  "
                f"Auto-actions: {_log_str}"
            ),
            mo.ui.table(_rows, pagination=False),
        ])

    _mgmt_output
    return


@app.cell
def _(
    acc,
    balance,
    cfg,
    datetime,
    kelly_risk_pct,
    mo,
    signal_stats,
    today_count,
):
    mo.vstack([
        mo.Html("<style>td, th { font-size: 0.9rem !important; }</style>"),
        mo.md(
            f"## {acc.login} | {acc.currency} {balance:,.2f} | "
            f"{datetime.now().strftime('%H:%M:%S')} | "
            f"Signals: {today_count} | Kelly: {kelly_risk_pct*100:.2f}%"
        ),
        mo.md(
            f"**Gates (L/S):** "
            f"T=trend(ema144) · A=pred_align · D=delta · R=RSI_MA · V=vol  |  "
            f"**Corr block:** |r|>{cfg.CORR_BLOCK_THRESHOLD} with open positions  |  "
            f"**SL:** {cfg.SL_ADR_FRACTION*100:.0f}%ADR · "
            f"**TP1:** {cfg.TP1_ADR_FRACTION*100:.0f}%ADR · "
            f"**TP2:** {cfg.TP2_ADR_FRACTION*100:.0f}%ADR"
        ),
        mo.ui.table(signal_stats, pagination=False),
    ])
    return


@app.cell
def _(mo):
    lookback_selector = mo.ui.dropdown(
        options=["7", "14", "30", "60"],
        value="7",
        label="Lookback (Days)",
    )
    lookback_selector
    return (lookback_selector,)


@app.cell
def _(
    ThreadPoolExecutor,
    adr_pips_map,
    cfg,
    lookback_selector,
    mo,
    mt5,
    np,
    pl,
    results,
):
    import altair as alt

    _days     = int(lookback_selector.value)
    _m30_bars = _days * 48

    def _get_adr_pips(symbol):
        if _days == cfg.ADR_LOOKBACK_DAYS and symbol in adr_pips_map:
            return {"Symbol": symbol, "ADR_pips": adr_pips_map[symbol]}
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, _days)
        if rates is None or len(rates) == 0:
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        pip_size = info.point * (10.0 if info.digits in [3, 5] else 1.0)
        df       = pl.DataFrame(rates)
        return {"Symbol": symbol, "ADR_pips": round(float((df["high"] - df["low"]).mean() / pip_size), 1)}

    with ThreadPoolExecutor(max_workers=10) as _ex:
        _adr_data = [r for r in _ex.map(_get_adr_pips, cfg.SYMBOLS) if r is not None]

    # Correlation from results (already in memory)
    _price_dfs = [
        _df.tail(_m30_bars).select([pl.col("time"), pl.col("close").alias(_sym)])
        for _sym, _df in results
    ]
    _corr_matrix = None
    _corr_syms   = []
    if len(_price_dfs) > 1:
        _combined = _price_dfs[0]
        for _nxt in _price_dfs[1:]:
            _combined = _combined.join(_nxt, on="time", how="inner")
        _corr_syms = [c for c in _combined.columns if c != "time"]
        _rets      = _combined.select([pl.col(s).pct_change() for s in _corr_syms]).drop_nulls()
        _corr_matrix = _rets.corr()

    _corr_avgs = {}
    if _corr_matrix is not None:
        _cm = _corr_matrix.to_numpy()
        for _i, _s in enumerate(_corr_syms):
            _corr_avgs[_s] = round(float(np.mean([abs(v) for j, v in enumerate(_cm[_i]) if j != _i])), 3)

    if _adr_data:
        _adr_df = pl.DataFrame(_adr_data).sort("ADR_pips", descending=True)
        _adr_df = _adr_df.with_columns(
            pl.Series("Avg|r|",   [_corr_avgs.get(s, 0.0) for s in _adr_df["Symbol"].to_list()]),
            pl.Series("SL pips",  [round(_adr_df["ADR_pips"][i] * cfg.SL_ADR_FRACTION,  1) for i in range(_adr_df.height)]),
            pl.Series("TP1 pips", [round(_adr_df["ADR_pips"][i] * cfg.TP1_ADR_FRACTION, 1) for i in range(_adr_df.height)]),
            pl.Series("TP2 pips", [round(_adr_df["ADR_pips"][i] * cfg.TP2_ADR_FRACTION, 1) for i in range(_adr_df.height)]),
        )
        _adr_chart = (
            alt.Chart(_adr_df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Symbol:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-60)),
                y=alt.Y("ADR_pips:Q", title="ADR (Pips)"),
                color=alt.Color("ADR_pips:Q", scale=alt.Scale(scheme="viridis"), legend=None),
                tooltip=[
                    alt.Tooltip("Symbol:N"),
                    alt.Tooltip("ADR_pips:Q", format=".1f", title="ADR (pips)"),
                    alt.Tooltip("SL pips:Q",  format=".1f", title="SL (pips)"),
                    alt.Tooltip("TP1 pips:Q", format=".1f", title="TP1 (pips)"),
                    alt.Tooltip("TP2 pips:Q", format=".1f", title="TP2 (pips)"),
                    alt.Tooltip("Avg|r|:Q",   format=".3f", title="Avg |corr|"),
                ],
            )
            .properties(width=780, height=280, title=f"ADR — last {_days} days")
        )
        _adr_section = mo.vstack([
            mo.md(f"SL={cfg.SL_ADR_FRACTION*100:.0f}%ADR · TP1={cfg.TP1_ADR_FRACTION*100:.0f}%ADR · TP2={cfg.TP2_ADR_FRACTION*100:.0f}%ADR — hover bars for pip values."),
            _adr_chart,
            mo.ui.table(_adr_df.to_dicts(), pagination=False),
        ])
    else:
        _adr_section = mo.md("No ADR data.")

    if _corr_matrix is not None:
        _corr_long = (
            _corr_matrix
            .with_columns(pl.Series("S1", _corr_syms))
            .unpivot(index="S1", on=_corr_syms, variable_name="S2", value_name="Corr")
        )
        _base = alt.Chart(_corr_long).encode(
            x=alt.X("S1:O", title=None, axis=alt.Axis(labelAngle=-60)),
            y=alt.Y("S2:O", title=None),
            tooltip=["S1", "S2", alt.Tooltip("Corr:Q", format=".2f")],
        )
        _heat = _base.mark_rect().encode(
            color=alt.Color("Corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                            legend=alt.Legend(title="r"))
        )
        _text = _base.mark_text(baseline="middle", fontSize=7).encode(
            text=alt.Text("Corr:Q", format=".2f"),
            color=alt.condition("abs(datum.Corr) > 0.5", alt.value("white"), alt.value("black")),
        )
        _corr_section = mo.vstack([
            mo.md(f"|r| > {cfg.CORR_BLOCK_THRESHOLD} = signal blocked when position open. |r| > 0.7 = effectively the same position."),
            (_heat + _text).properties(width=820, height=820,
                                        title=f"Return Correlations — last {_days}d M30"),
        ])
    else:
        _corr_section = mo.md("Not enough data for correlation.")

    mo.vstack([
        mo.md("### Analytics — Volatility & Correlation"),
        mo.md(f"*ADR: D1 bars. Correlation: M30 already in memory. Lookback: {_days} days.*"),
        mo.md("#### ADR + SL/TP levels"), _adr_section,
        mo.md("#### Pair Correlations"), _corr_section,
    ])
    return


if __name__ == "__main__":
    app.run()
