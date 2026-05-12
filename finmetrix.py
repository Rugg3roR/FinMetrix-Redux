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

    # 3-minute refresh — catches new H1 candles within minutes of close
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
    Boot cell — no refresh_timer dependency so runs exactly once.

    1. Fetches 14-day ADR for all 28 pairs (D1 bars)
    2. Fetches D1 history for HTF persistence features
    3. Trains three XGBoost models on H1 history:
         model_long  — predicts target_long  (prob_win for longs)
         model_short — predicts target_short (prob_win for shorts)
         model_veto  — predicts target_fail_* (prob of fast failure)
                       trained on signal-matching rows from both
                       directions combined; direction-agnostic by design
                       because "bad trade conditions" are not directional
    4. Computes Kelly risk % from trade journal
    """
    with mo.status.spinner("Booting FinMetrix..."):
        from modules.data_engine import get_mt5_df

        # ── D1 fetch helper ───────────────────────────────────────────────────
        # Used for both ADR calculation and D1 persistence features.
        # Fetched once per symbol and shared between both uses.
        def _fetch_d1(symbol):
            rates = mt5.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_D1, 0, cfg.D1_BARS
            )
            if rates is None or len(rates) == 0:
                return symbol, None
            df = (
                pl.DataFrame(rates)
                .with_columns([
                    pl.from_epoch("time").alias("time"),
                    pl.col("open", "high", "low", "close").cast(pl.Float64),
                    pl.col("tick_volume").cast(pl.Float64),
                ])
            )
            return symbol, df

        with ThreadPoolExecutor(max_workers=8) as _ex:
            _d1_res = list(_ex.map(_fetch_d1, cfg.SYMBOLS))

        d1_map = {sym: df for sym, df in _d1_res if df is not None}

        # ── ADR calculation from D1 data ──────────────────────────────────────
        adr_map      = {}
        adr_pips_map = {}
        for _sym, _d1 in d1_map.items():
            _info = mt5.symbol_info(_sym)
            if _info is None or _d1 is None:
                continue
            _pip_size  = _info.point * (10.0 if _info.digits in [3, 5] else 1.0)
            _adr_price = float(
                (_d1["high"] - _d1["low"]).tail(cfg.ADR_LOOKBACK_DAYS).mean()
            )
            adr_map[_sym]      = _adr_price
            adr_pips_map[_sym] = round(_adr_price / _pip_size, 1)

        # ── Model training ────────────────────────────────────────────────────
        _long_rows  = []
        _short_rows = []
        _fail_rows  = []   # combined long + short fail-fast rows for model_veto
        _rsi_l_min  = cfg.RSI_MID + cfg.RSI_BUFFER
        _rsi_s_max  = cfg.RSI_MID - cfg.RSI_BUFFER

        for _symbol in cfg.SYMBOLS:
            _h1 = get_mt5_df(_symbol, cfg.TIMEFRAME, cfg.BARS)
            if _h1 is None:
                continue

            # Pass D1 data for this symbol so engineer_features can compute
            # the D1 persistence feature family (d1_persist_3 etc.)
            _d1 = d1_map.get(_symbol)
            _df = ft.engineer_features(_h1, _d1)
            _df = ft.triple_barrier(
                _df,
                tp_mult=cfg.TP1_MULT,
                sl_mult=cfg.SL_MULT,
                max_hold=cfg.MAX_HOLD,
            )

            # ── Long model training rows ──────────────────────────────────────
            _df_long = _df.filter(
                (pl.col("ema144_slope")     >  cfg.EMA144_SLOPE_MIN) &
                (pl.col("adx")              >  cfg.ADX_MIN) &
                (pl.col("pred_alignment")   >= cfg.PRED_ALIGN_LONG) &
                (pl.col("pred_align_delta") >= cfg.PRED_ALIGN_DELTA_MIN) &
                (pl.col("rsi_ma")           >  _rsi_l_min) &
                (pl.col("vol_ratio")        >  1.0)
            ).select([*cfg.FEATURES, "target_long"]).drop_nulls()

            # ── Short model training rows ─────────────────────────────────────
            _df_short = _df.filter(
                (pl.col("ema144_slope")     <  -cfg.EMA144_SLOPE_MIN) &
                (pl.col("adx")              >  cfg.ADX_MIN) &
                (pl.col("pred_alignment")   <= cfg.PRED_ALIGN_SHORT) &
                (pl.col("pred_align_delta") <= -cfg.PRED_ALIGN_DELTA_MIN) &
                (pl.col("rsi_ma")           <  _rsi_s_max) &
                (pl.col("vol_ratio")        >  1.0)
            ).select([*cfg.FEATURES, "target_short"]).drop_nulls()

            # ── Veto model training rows ──────────────────────────────────────
            # Trained on ALL signal-matching rows (long + short combined).
            # Fail-fast is direction-agnostic — the market conditions that
            # cause violent rejection are symmetric. Combining both directions
            # doubles the training set and improves generalisation.
            # Target column: target_fail_long for long rows, target_fail_short
            # for short rows — renamed to a unified "target_fail" column.
            _df_fail_l = _df.filter(
                (pl.col("ema144_slope")     >  cfg.EMA144_SLOPE_MIN) &
                (pl.col("adx")             >  cfg.ADX_MIN) &
                (pl.col("pred_alignment")  >= cfg.PRED_ALIGN_LONG) &
                (pl.col("pred_align_delta")>= cfg.PRED_ALIGN_DELTA_MIN) &
                (pl.col("rsi_ma")          >  _rsi_l_min) &
                (pl.col("vol_ratio")       >  1.0)
            ).select([*cfg.VETO_FEATURES, "target_fail_long"]).drop_nulls() \
             .rename({"target_fail_long": "target_fail"})

            _df_fail_s = _df.filter(
                (pl.col("ema144_slope")     <  -cfg.EMA144_SLOPE_MIN) &
                (pl.col("adx")             >  cfg.ADX_MIN) &
                (pl.col("pred_alignment")  <= cfg.PRED_ALIGN_SHORT) &
                (pl.col("pred_align_delta")<= -cfg.PRED_ALIGN_DELTA_MIN) &
                (pl.col("rsi_ma")          <  _rsi_s_max) &
                (pl.col("vol_ratio")       >  1.0)
            ).select([*cfg.VETO_FEATURES, "target_fail_short"]).drop_nulls() \
             .rename({"target_fail_short": "target_fail"})

            if _df_long.height > 0:
                _long_rows.append(_df_long.cast(pl.Float64).to_numpy())
            if _df_short.height > 0:
                _short_rows.append(_df_short.cast(pl.Float64).to_numpy())
            if _df_fail_l.height > 0:
                _fail_rows.append(_df_fail_l.cast(pl.Float64).to_numpy())
            if _df_fail_s.height > 0:
                _fail_rows.append(_df_fail_s.cast(pl.Float64).to_numpy())

        # ── Fit model_long ────────────────────────────────────────────────────
        model_long = None
        _n_long    = 0
        if _long_rows:
            _data_l    = np.vstack(_long_rows)
            _n_long    = len(_data_l)
            model_long = mh.train_model(_data_l[:, :-1], _data_l[:, -1])

        # ── Fit model_short ───────────────────────────────────────────────────
        model_short = None
        _n_short    = 0
        if _short_rows:
            _data_s     = np.vstack(_short_rows)
            _n_short    = len(_data_s)
            model_short = mh.train_model(_data_s[:, :-1], _data_s[:, -1])

        # ── Fit model_veto ────────────────────────────────────────────────────
        # Uses the same XGB hyperparameters as long/short models.
        # scale_pos_weight is inherited from config — fail-fast labels are
        # typically minority class (~20-40% of signal rows), so the
        # existing imbalance correction applies naturally.
        model_veto = None
        _n_veto    = 0
        if _fail_rows:
            _data_v    = np.vstack(_fail_rows)
            _n_veto    = len(_data_v)
            model_veto = mh.train_model(_data_v[:, :-1], _data_v[:, -1])

    # ── Kelly ─────────────────────────────────────────────────────────────────
    kelly_risk_pct = je.calculate_adaptive_risk()
    _kelly_src     = "adaptive" if kelly_risk_pct > cfg.KELLY_MIN_FLOOR else "floor"

    # ── Veto class balance diagnostic ─────────────────────────────────────────
    _veto_pos_pct = 0.0
    if _fail_rows and _n_veto > 0:
        _all_labels   = np.vstack(_fail_rows)[:, -1]
        _veto_pos_pct = float(_all_labels.mean() * 100)

    mo.md(f"""
    ✅ **FinMetrix ready** — H1 candles · 6-gate signal · ADX > {cfg.ADX_MIN}

    | | |
    | :--- | :--- |
    | **Long training rows** | {_n_long:,} |
    | **Short training rows** | {_n_short:,} |
    | **Veto training rows** | {_n_veto:,} (fail-fast rate: {_veto_pos_pct:.1f}%) |
    | **D1 symbols loaded** | {len(d1_map)} / {len(cfg.SYMBOLS)} |
    | **ADR symbols** | {len(adr_map)} / {len(cfg.SYMBOLS)} |
    | **Kelly risk** | {kelly_risk_pct*100:.2f}% ({_kelly_src}) |
    | **SL / TP1 / TP2** | {cfg.SL_ADR_FRACTION*100:.0f}% / {cfg.TP1_ADR_FRACTION*100:.0f}% / {cfg.TP2_ADR_FRACTION*100:.0f}% of {cfg.ADR_LOOKBACK_DAYS}-day ADR |
    | **Signal age limit** | {cfg.MAX_SIGNAL_AGE} bars ({cfg.MAX_SIGNAL_AGE}h on H1) |
    | **Cooldown** | {cfg.COOLDOWN_BARS} bars ({cfg.COOLDOWN_BARS}h on H1) |
    | **Min edge score** | {cfg.MIN_EDGE_SCORE} (prob_win − prob_veto) |
    """)
    return (
        adr_map,
        adr_pips_map,
        d1_map,
        kelly_risk_pct,
        model_long,
        model_short,
        model_veto,
    )


@app.cell
def _(
    ThreadPoolExecutor,
    adr_map,
    adr_pips_map,
    balance,
    cfg,
    d1_map,
    datetime,
    de,
    ft,
    get_cooldown,
    kelly_risk_pct,
    model_long,
    model_short,
    model_veto,
    mt5,
    np,
    refresh_timer,
    set_cooldown,
    timezone,
):
    refresh_timer.value

    # ── Fetch H1 data ─────────────────────────────────────────────────────────
    # D1 features for the live scanner use the boot-time d1_map.
    # The D1 data is already shifted by 1 day inside _compute_d1_features,
    # so the previous day's D1 close is always what's reflected in the
    # current H1 bar's d1_* columns. No intraday refresh of D1 is needed.
    def _fetch(symbol):
        df = de.get_mt5_df(symbol, cfg.TIMEFRAME, cfg.BARS)
        if df is None:
            return None
        _d1 = d1_map.get(symbol)
        return symbol, ft.engineer_features_fast(df, _d1)

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

    # ── Cooldown ──────────────────────────────────────────────────────────────
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

    # ── Thresholds ────────────────────────────────────────────────────────────
    _rsi_l_min = cfg.RSI_MID + cfg.RSI_BUFFER
    _rsi_s_max = cfg.RSI_MID - cfg.RSI_BUFFER

    # ── Signal age helper ─────────────────────────────────────────────────────
    def _age(df, direction):
        tail = df.tail(cfg.MAX_SIGNAL_AGE + 2)
        n    = tail.height
        rows = tail.to_dicts()
        if direction == "long":
            cond = [
                r.get("ema144_slope",     0)   >  cfg.EMA144_SLOPE_MIN and
                r.get("adx",              0)   >  cfg.ADX_MIN and
                r.get("pred_alignment",   2.0) >= cfg.PRED_ALIGN_LONG and
                r.get("pred_align_delta", 0)   >= cfg.PRED_ALIGN_DELTA_MIN and
                r.get("rsi_ma",           50)  >  _rsi_l_min and
                r.get("vol_ratio",        0)   >  1.0
                for r in rows
            ]
        else:
            cond = [
                r.get("ema144_slope",     0)   <  -cfg.EMA144_SLOPE_MIN and
                r.get("adx",              0)   >  cfg.ADX_MIN and
                r.get("pred_alignment",   2.0) <= cfg.PRED_ALIGN_SHORT and
                r.get("pred_align_delta", 0)   <= -cfg.PRED_ALIGN_DELTA_MIN and
                r.get("rsi_ma",           50)  <  _rsi_s_max and
                r.get("vol_ratio",        0)   >  1.0
                for r in rows
            ]
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
    signal_rows = []
    today_count = de.signals_today()
    _now_ts     = datetime.now(timezone.utc)

    for symbol, df in results:
        now = df.tail(1).to_dicts()[0]
        X   = np.array([[now.get(f, 0.0) for f in cfg.FEATURES]])

        # Directional win probabilities
        prob_l = float(model_long.predict_proba(X)[0][1])  if model_long  else 0.0
        prob_s = float(model_short.predict_proba(X)[0][1]) if model_short else 0.0

        # Veto probability — probability of fast failure given current context.
        # Uses VETO_FEATURES (same as FEATURES for now; can diverge later).
        X_veto   = np.array([[now.get(f, 0.0) for f in cfg.VETO_FEATURES]])
        prob_veto = float(model_veto.predict_proba(X_veto)[0][1]) if model_veto else 0.0

        # Edge scores — net signal quality after subtracting failure risk
        edge_l = prob_l - prob_veto
        edge_s = prob_s - prob_veto

        ema144_slope = now.get("ema144_slope",    0.0)
        adx          = now.get("adx",             0.0)
        pred_align   = now.get("pred_alignment",  2.0)
        pred_delta   = now.get("pred_align_delta",0.0)
        rsi_ma       = now.get("rsi_ma",          50.0)
        vol_ratio    = now.get("vol_ratio",        1.0)
        price        = now.get("close",            0.0)

        # D1 context values for display
        d1_persist   = now.get("d1_persist_3",      0.0)
        d1_adx       = now.get("d1_adx",            20.0)

        # Gate checks — 6 gates each direction (unchanged)
        g_trend_l = ema144_slope >  cfg.EMA144_SLOPE_MIN
        g_trend_s = ema144_slope < -cfg.EMA144_SLOPE_MIN
        g_adx     = adx          >  cfg.ADX_MIN
        g_align_l = pred_align   >= cfg.PRED_ALIGN_LONG
        g_align_s = pred_align   <= cfg.PRED_ALIGN_SHORT
        g_delta_l = pred_delta   >= cfg.PRED_ALIGN_DELTA_MIN
        g_delta_s = pred_delta   <= -cfg.PRED_ALIGN_DELTA_MIN
        g_rsi_l   = rsi_ma       >  _rsi_l_min
        g_rsi_s   = rsi_ma       <  _rsi_s_max
        g_vol     = vol_ratio    >  1.0

        is_long  = g_trend_l and g_adx and g_align_l and g_delta_l and g_rsi_l and g_vol
        is_short = g_trend_s and g_adx and g_align_s and g_delta_s and g_rsi_s and g_vol

        # Gate flag strings — 6 chars: Trend | ADX | Align | Delta | RSI | Vol
        flags_l = "".join([
            "T" if g_trend_l else ".",
            "X" if g_adx     else ".",
            "A" if g_align_l else ".",
            "D" if g_delta_l else ".",
            "R" if g_rsi_l   else ".",
            "V" if g_vol     else ".",
        ])
        flags_s = "".join([
            "T" if g_trend_s else ".",
            "X" if g_adx     else ".",
            "A" if g_align_s else ".",
            "D" if g_delta_s else ".",
            "R" if g_rsi_s   else ".",
            "V" if g_vol     else ".",
        ])

        corr_blocked = _blocked(symbol)
        in_cooldown  = symbol in _cd
        signal       = "Neutral"
        prob         = max(prob_l, prob_s)
        edge_score   = 0.0
        age          = -1

        if is_long and not in_cooldown and not corr_blocked:
            age = _age(df, "long")
            # Signal fires only if both raw prob AND edge score clear their
            # respective thresholds. MIN_EDGE_SCORE = 0.0 in config so the
            # veto has no blocking effect until it's been calibrated.
            if (0 <= age <= cfg.MAX_SIGNAL_AGE
                    and prob_l >= cfg.MIN_PROB
                    and edge_l >= cfg.MIN_EDGE_SCORE):
                signal     = "BUY"
                prob       = prob_l
                edge_score = edge_l

        elif is_short and not in_cooldown and not corr_blocked:
            age = _age(df, "short")
            if (0 <= age <= cfg.MAX_SIGNAL_AGE
                    and prob_s >= cfg.MIN_PROB
                    and edge_s >= cfg.MIN_EDGE_SCORE):
                signal     = "SELL"
                prob       = prob_s
                edge_score = edge_s

        # ── SL / TP / Lot — always calculated ────────────────────────────────
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

        if signal == "BUY":
            _dir = "BUY"
        elif signal == "SELL":
            _dir = "SELL"
        elif ema144_slope > 0 and pred_align >= 3:
            _dir = "BUY"
        elif ema144_slope < 0 and pred_align <= 1:
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
            sl_in_ticks = sl_dist / _info.trade_tick_size
            lot_size    = (balance * kelly_risk_pct) / (sl_in_ticks * _info.trade_tick_value)
            lot_size    = max(_info.volume_min,
                              round(lot_size / _info.volume_step) * _info.volume_step)
            risk_gbp    = round(lot_size * sl_in_ticks * _info.trade_tick_value, 2)

        # ── Log signal on new candle ──────────────────────────────────────────
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
                "prob_veto":   prob_veto,
                "edge_score":  edge_score,
                "d1_persist":  d1_persist,
                "d1_adx":      d1_adx,
                "signal_age":  age,
                **{f: now.get(f, 0.0) for f in cfg.FEATURES},
            })

        # ── Status tag ────────────────────────────────────────────────────────
        if corr_blocked:
            tag = " [exp]"
        elif in_cooldown:
            tag = f" [cd{_cd.get(symbol, 0)}]"
        else:
            tag = ""

        # Veto display: show prob_veto and edge_score for active signals,
        # prob_veto alone for near-miss rows so you can see what it's doing
        _veto_str  = f"{prob_veto*100:.0f}%"
        _edge_str  = f"{edge_score:+.2f}" if signal != "Neutral" else "—"

        signal_rows.append({
            "Symbol":    symbol,
            "Signal":    signal + tag,
            "ADR":       adr_p,
            "Prob":      f"{prob*100:.0f}%",
            "Veto":      _veto_str,
            "Edge":      _edge_str,
            "Lot":       round(lot_size, 2),
            "Risk GBP":  risk_gbp,
            "Price":     round(price, 5),
            "SL":        sl,
            "TP1":       tp1,
            "TP2":       tp2,
            "D1 bias":   f"{d1_persist:+.2f}",
            "D1 ADX":    round(d1_adx, 1),
            "Gates L":   flags_l,
            "Gates S":   flags_s,
        })

    signal_rows.sort(key=lambda x: (
        x["Signal"].split()[0] not in ("BUY", "SELL"),
        -float(x["Prob"][:-1])
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
            f"Gates: **T**rend(ema144) · **X**(ADX>{cfg.ADX_MIN}) · "
            f"**A**lign · **D**elta · **R**SI_MA · **V**ol  |  "
            f"[exp]=currency exposure block · [cdN]=cooldown bars remaining  |  "
            f"**Veto**=fail-fast prob · **Edge**=prob_win−prob_veto "
            f"(min {cfg.MIN_EDGE_SCORE:+.2f})"
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
                    "Symbol":   _sym,   "Side":      _side,
                    "Lots":     _lots,  "Entry":     round(_entry, 5),
                    "Now":      round(_price, 5),
                    "R":        _r,     "SL pips":   _sl_pips,
                    "SL":       round(_sl, 5) if _sl else "—",
                    "BE":       round(_entry, 5),
                    "TP1(50%)": _tp1_p, "TP2(25%)": _tp2_p,
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
