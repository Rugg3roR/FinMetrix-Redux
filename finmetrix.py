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

    # Persistent cooldown state across refresh cycles
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
    Boot cell — runs once at startup, no refresh dependency.

    1. Fetches D1 bars for ADR calculation (14-day average daily range)
    2. Trains model_long and model_short (kept for logging/future use —
       NOT used to gate signals in this version)
    3. Computes Kelly risk % from trade journal
    """
    with mo.status.spinner("Booting FinMetrix..."):
        from modules.data_engine import get_mt5_df

        # ── ADR fetch ─────────────────────────────────────────────────────────
        def _fetch_adr(symbol):
            rates = mt5.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_D1, 0, cfg.ADR_LOOKBACK_DAYS
            )
            if rates is None or len(rates) == 0:
                return symbol, None, None
            _info = mt5.symbol_info(symbol)
            if _info is None:
                return symbol, None, None
            _pip_size  = _info.point * (10.0 if _info.digits in [3, 5] else 1.0)
            _df        = pl.DataFrame(rates)
            _adr_price = float((_df["high"] - _df["low"]).mean())
            _adr_pips  = round(_adr_price / _pip_size, 1)
            return symbol, _adr_price, _adr_pips

        with ThreadPoolExecutor(max_workers=8) as _ex:
            _adr_res = list(_ex.map(_fetch_adr, cfg.SYMBOLS))

        adr_map      = {}
        adr_pips_map = {}
        for _sym, _ap, _apips in _adr_res:
            if _ap is not None:
                adr_map[_sym]      = _ap
                adr_pips_map[_sym] = _apips

        # ── Model training (informational — not gating signals) ───────────────
        # Models are trained and their probabilities shown in the dashboard
        # so you can observe how they correlate with outcomes over time.
        # They do NOT block or require signals in the current version.
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
            # Train on EMA-aligned + ADX-trending rows (relaxed — just
            # needs to be directionally aligned, no RSI/vol constraint)
            _df_long = _df.filter(
                (pl.col("ema144_slope")   > 0) &
                (pl.col("adx")            > cfg.ADX_MIN) &
                (pl.col("pred_alignment") >= 2)
            ).select([*cfg.FEATURES, "target_long"]).drop_nulls()

            _df_short = _df.filter(
                (pl.col("ema144_slope")   < 0) &
                (pl.col("adx")            > cfg.ADX_MIN) &
                (pl.col("pred_alignment") <= 2)
            ).select([*cfg.FEATURES, "target_short"]).drop_nulls()

            if _df_long.height > 0:
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

    # ── Kelly ─────────────────────────────────────────────────────────────────
    kelly_risk_pct = je.calculate_adaptive_risk()
    _kelly_src     = "adaptive" if kelly_risk_pct > cfg.KELLY_MIN_FLOOR else "floor"

    mo.md(f"""
    ✅ **FinMetrix ready** — H1 candles · 3-gate signal (EMA + RSI MA cross + ADX)

    | | |
    | :--- | :--- |
    | **Long training rows** | {_n_long:,} |
    | **Short training rows** | {_n_short:,} |
    | **ADR symbols** | {len(adr_map)} / {len(cfg.SYMBOLS)} |
    | **Kelly risk** | {kelly_risk_pct*100:.2f}% ({_kelly_src}) |
    | **SL / TP1 / TP2** | {cfg.SL_ADR_FRACTION*100:.0f}% / {cfg.TP1_ADR_FRACTION*100:.0f}% / {cfg.TP2_ADR_FRACTION*100:.0f}% of {cfg.ADR_LOOKBACK_DAYS}-day ADR |
    | **Cooldown** | {cfg.COOLDOWN_BARS}h after signal fires |
    | **Signal gates** | ADX > {cfg.ADX_MIN} · EMA alignment · RSI MA(4) > RSI MA(21) |
    | **ML prob** | Displayed only — not gating signals |
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

    # ── RSI dual MA periods — matches the overlay exactly ────────────────────
    # RSI fast MA (4) crosses above RSI slow MA (21) → bullish momentum
    # RSI fast MA (4) crosses below RSI slow MA (21) → bearish momentum
    _RSI_FAST = 4
    _RSI_SLOW = 21

    # ── Fetch H1 data ─────────────────────────────────────────────────────────
    def _fetch(symbol):
        _df = de.get_mt5_df(symbol, cfg.TIMEFRAME, cfg.BARS)
        if _df is None:
            return None
        return symbol, ft.engineer_features_fast(_df)

    with ThreadPoolExecutor(max_workers=8) as _ex:
        _raw = list(_ex.map(_fetch, cfg.SYMBOLS))
    results = [r for r in _raw if r is not None]

    # ── Candle-close guard ────────────────────────────────────────────────────
    scan_market = True
    if results:
        _ref  = next((df for s, df in results if s == "EURUSD"), results[0][1])
        _last = _ref.tail(1)["time"][0]
        if cfg.LAST_CANDLE_FILE.exists():
            if str(_last) == cfg.LAST_CANDLE_FILE.read_text().strip():
                scan_market = False
        if scan_market:
            cfg.LAST_CANDLE_FILE.write_text(str(_last))

    # ── Cooldown decay on new candle only ─────────────────────────────────────
    _cd = get_cooldown()
    if scan_market:
        _cd = {s: c - 1 for s, c in _cd.items() if c > 1}
        set_cooldown(_cd)

    # ── Currency exposure filter ──────────────────────────────────────────────
    _positions    = mt5.positions_get() or []
    _open_symbols = {p.symbol for p in _positions}
    _exposed_curr = {c for sym in _open_symbols for c in (sym[:3], sym[3:6])}

    def _blocked(symbol):
        if not _open_symbols:
            return False
        return symbol[:3] in _exposed_curr or symbol[3:6] in _exposed_curr

    # ── RSI dual MA helper ────────────────────────────────────────────────────
    # Computes fast and slow moving averages of the RSI series.
    # Returns (rsi_fast_ma, rsi_slow_ma) from the last bar.
    # Uses a simple rolling mean on the rsi column from engineer_features.
    def _rsi_dual_ma(df):
        rsi_arr = df["rsi"].to_numpy()
        n       = len(rsi_arr)
        if n < _RSI_SLOW:
            return 50.0, 50.0
        fast = float(np.mean(rsi_arr[-_RSI_FAST:])) if n >= _RSI_FAST else 50.0
        slow = float(np.mean(rsi_arr[-_RSI_SLOW:]))
        return fast, slow

    # ── ATR regime helper (mirrors overlay) ───────────────────────────────────
    # Returns "EXPANSION", "COMPRESSION", or "NORMAL"
    def _regime(df):
        atr_arr = df["atr"].drop_nulls().to_numpy()
        if len(atr_arr) < 21:
            return "NORMAL"
        current = atr_arr[-1]
        avg     = float(np.mean(atr_arr[-20:]))
        if avg == 0:
            return "NORMAL"
        if current > avg * 1.3:
            return "EXPANSION"
        if current < avg * 0.7:
            return "COMPRESSION"
        return "NORMAL"

    # ── Trade Appeal score (mirrors overlay logic) ────────────────────────────
    # Score 0-100. Informational — you decide whether to act on it.
    # Components match the overlay: trend, RSI MA cross, volume, candle
    # completion approximation, and regime.
    def _appeal(trend_aligned, rsi_fast, rsi_slow, vol_ratio, regime_str):
        score = 50
        if trend_aligned:
            score += 15
        # RSI MA cross in the right direction
        rsi_cross_ok = (rsi_fast > rsi_slow) if trend_aligned else (rsi_fast < rsi_slow)
        if rsi_cross_ok:
            score += 15
        # Volume expanding
        if vol_ratio > 1.0:
            score += 15
        # Regime
        if regime_str == "EXPANSION":
            score += 10
        elif regime_str == "COMPRESSION":
            score -= 10
        return max(0, min(100, score))

    # ── Score every pair ──────────────────────────────────────────────────────
    signal_rows = []
    today_count = de.signals_today()
    _now_ts     = datetime.now(timezone.utc)

    for symbol, df in results:
        now = df.tail(1).to_dicts()[0]

        # ── Core indicator values ─────────────────────────────────────────────
        ema144_slope = now.get("ema144_slope",    0.0)
        adx          = now.get("adx",             0.0)
        pred_align   = now.get("pred_alignment",  2.0)
        rsi_ma_val   = now.get("rsi_ma",          50.0)
        vol_ratio    = now.get("vol_ratio",        1.0)
        price        = now.get("close",            0.0)

        # RSI dual MA cross (4-period vs 21-period of raw RSI)
        rsi_fast, rsi_slow = _rsi_dual_ma(df)
        rsi_bull = rsi_fast > rsi_slow   # bullish RSI momentum
        rsi_bear = rsi_fast < rsi_slow   # bearish RSI momentum

        # Regime and appeal
        regime_str = _regime(df)

        # ── THREE signal gates ────────────────────────────────────────────────
        # Gate 1 — ADX: market is actually trending, not ranging
        # Gate 2 — EMA alignment: pred_alignment direction
        # Gate 3 — RSI MA cross: fast RSI MA aligned with direction
        #
        # These three must ALL pass. No ML gate. No vol gate. No delta gate.
        # You apply your own discretion on top using the Appeal score and
        # the additional columns shown in the dashboard.

        g_adx     = adx > cfg.ADX_MIN           # ADX > 20
        g_align_l = pred_align >= 2              # at least 2 of 4 EMA pairs bullish
        g_align_s = pred_align <= 2              # at least 2 of 4 EMA pairs bearish
        g_trend_l = ema144_slope > 0             # EMA144 pointing up
        g_trend_s = ema144_slope < 0             # EMA144 pointing down
        g_rsi_l   = rsi_bull                     # RSI fast MA > RSI slow MA
        g_rsi_s   = rsi_bear                     # RSI fast MA < RSI slow MA

        is_long  = g_adx and g_trend_l and g_align_l and g_rsi_l
        is_short = g_adx and g_trend_s and g_align_s and g_rsi_s

        # Gate flags for display — 4 chars: ADX | Trend | Align | RSI cross
        flags_l = "".join([
            "X" if g_adx     else ".",
            "T" if g_trend_l else ".",
            "A" if g_align_l else ".",
            "R" if g_rsi_l   else ".",
        ])
        flags_s = "".join([
            "X" if g_adx     else ".",
            "T" if g_trend_s else ".",
            "A" if g_align_s else ".",
            "R" if g_rsi_s   else ".",
        ])

        # Appeal score for display
        _appeal_l = _appeal(g_trend_l and g_align_l, rsi_fast, rsi_slow,
                            vol_ratio, regime_str)
        _appeal_s = _appeal(g_trend_s and g_align_s, rsi_fast, rsi_slow,
                            vol_ratio, regime_str)
        appeal_score = _appeal_l if (is_long or g_trend_l) else _appeal_s

        # ── ML probability (informational only) ───────────────────────────────
        _X     = np.array([[now.get(f, 0.0) for f in cfg.FEATURES]])
        prob_l = float(model_long.predict_proba(_X)[0][1])  if model_long  else 0.0
        prob_s = float(model_short.predict_proba(_X)[0][1]) if model_short else 0.0

        # ── Signal decision ───────────────────────────────────────────────────
        corr_blocked = _blocked(symbol)
        in_cooldown  = symbol in _cd
        signal       = "Neutral"
        prob         = max(prob_l, prob_s)

        if is_long and not in_cooldown and not corr_blocked:
            signal = "BUY"
            prob   = prob_l

        elif is_short and not in_cooldown and not corr_blocked:
            signal = "SELL"
            prob   = prob_s

        # ── SL / TP / Lot sizing ──────────────────────────────────────────────
        adr   = adr_map.get(symbol)
        adr_p = adr_pips_map.get(symbol, 0.0)

        if adr and adr > 0:
            sl_dist  = adr * cfg.SL_ADR_FRACTION
            tp1_dist = adr * cfg.TP1_ADR_FRACTION
            tp2_dist = adr * cfg.TP2_ADR_FRACTION
        else:
            _atr     = now.get("atr", 0.001)
            sl_dist  = _atr * cfg.SL_MULT
            tp1_dist = _atr * cfg.TP1_MULT
            tp2_dist = _atr * cfg.TP2_MULT
            adr_p    = 0.0

        # Always show levels based on likely direction
        if signal == "BUY" or (signal == "Neutral" and g_trend_l and g_align_l):
            _dir = "BUY"
        elif signal == "SELL" or (signal == "Neutral" and g_trend_s and g_align_s):
            _dir = "SELL"
        else:
            _dir = "—"

        if _dir == "BUY":
            sl  = round(price - sl_dist,  5)
            tp1 = round(price + tp1_dist, 5)
            tp2 = round(price + tp2_dist, 5)
        elif _dir == "SELL":
            sl  = round(price + sl_dist,  5)
            tp1 = round(price - tp1_dist, 5)
            tp2 = round(price - tp2_dist, 5)
        else:
            sl = tp1 = tp2 = "—"

        lot_size = 0.0
        risk_gbp = 0.0
        sl_pips  = round(adr_p * cfg.SL_ADR_FRACTION, 1) if adr_p else 0.0

        _info = mt5.symbol_info(symbol)
        if _info and sl_dist > 0:
            _sl_ticks = sl_dist / _info.trade_tick_size
            lot_size  = (balance * kelly_risk_pct) / (_sl_ticks * _info.trade_tick_value)
            lot_size  = max(_info.volume_min,
                            round(lot_size / _info.volume_step) * _info.volume_step)
            risk_gbp  = round(lot_size * _sl_ticks * _info.trade_tick_value, 2)

        # ── Log signal ────────────────────────────────────────────────────────
        if scan_market and signal != "Neutral":
            today_count += 1
            _cd[symbol] = cfg.COOLDOWN_BARS
            set_cooldown(_cd)
            de.log_signal({
                "time":        _now_ts,
                "candle_time": now["time"],
                "symbol":      symbol,
                "signal":      signal,
                "entry":       price,
                "sl":          sl,
                "tp1":         tp1,
                "tp2":         tp2,
                "sl_pips":     sl_pips,
                "lot_size":    lot_size,
                "prob":        prob,
                "appeal":      appeal_score,
                "rsi_fast":    round(rsi_fast, 2),
                "rsi_slow":    round(rsi_slow, 2),
                "regime":      regime_str,
                "adx":         round(adx, 1),
                "pred_align":  pred_align,
                **{f: now.get(f, 0.0) for f in cfg.FEATURES},
            })

        # ── Status tag ────────────────────────────────────────────────────────
        if corr_blocked:
            tag = " [exp]"
        elif in_cooldown:
            tag = f" [cd{_cd.get(symbol, 0)}]"
        else:
            tag = ""

        # Appeal colour hint for display
        _appeal_str = f"{appeal_score}/100"

        signal_rows.append({
            "Symbol":    symbol,
            "Signal":    signal + tag,
            "ADR pips":  adr_p,
            "Appeal":    _appeal_str,
            "Regime":    regime_str,
            "Price":     round(price, 5),
            "Lot":       round(lot_size, 2),
            "Risk GBP":  risk_gbp,
            "SL":        sl,
            "TP1":       tp1,
            "TP2":       tp2,
            "ML prob":   f"{prob*100:.0f}%",
            "Gates L":   flags_l,
            "Gates S":   flags_s
        })

    # Sort: active signals first, then by appeal score descending
    signal_rows.sort(key=lambda x: (
        x["Signal"].split()[0] not in ("BUY", "SELL"),
        -int(x["Appeal"].split("/")[0])
    ))
    return signal_rows, today_count


@app.cell
def _(
    acc,
    balance,
    cfg,
    datetime,
    kelly_risk_pct,
    mo,
    signal_rows,
    today_count,
):
    mo.vstack([
        mo.Html("<style>td, th { font-size: 0.9rem !important; }</style>"),
        mo.md(
            f"### {acc.login} · {acc.currency} {balance:,.2f} · "
            f"{datetime.now().strftime('%H:%M:%S')} · "
            f"Signals today: {today_count} · "
            f"Kelly: {kelly_risk_pct*100:.2f}%"
        ),
        mo.md(
            f"**Gates (all 3 must pass):** "
            f"**X** ADX>{cfg.ADX_MIN} · "
            f"**T** EMA144 slope · "
            f"**A** EMA alignment (pred) · "
            f"**R** RSI MA(4) x RSI MA(21)  |  "
            f"[exp]=currency blocked · [cdN]=cooldown  |  "
            f"**Appeal** 0–100: use for discretion (≥80 = favourable, ≤50 = caution)"
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

            _tick   = mt5.symbol_info_tick(_sym)
            _info   = mt5.symbol_info(_sym)
            _price  = (_tick.bid if _side == "BUY" else _tick.ask) if _tick else _entry
            _pip    = (_info.point * (10.0 if _info.digits in [3, 5] else 1.0)) if _info else 0.0001

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

                _sl_pips = round(_sl_d / _pip, 1)
                _action  = "Hold"
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
                    "Lots":     _lots, "Entry":      round(_entry, 5),
                    "Now":      round(_price, 5),
                    "R":        "—",   "SL pips":    "—",
                    "SL":       "SET SL FIRST",
                    "BE":       "—",   "TP1(50%)":   "—",
                    "TP2(25%)": "—",   "Trail":       "—",
                    "TP1 hit":  "—",   "TP2 hit":     "—",
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
