import marimo

__generated_with = "0.23.6"
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
    Runs once at startup. No refresh_timer dependency.

    Computes per-symbol:
      - ADR (14-day average daily range) for SL/TP sizing
      - AHR (average hourly range) bucketed into 4 trading sessions,
        used to estimate ETA to TP1 in both scanner and position manager

    Session buckets (UTC):
      Asia      00:00 – 06:59
      London    07:00 – 11:59
      Overlap   12:00 – 15:59
      Off-peak  16:00 – 23:59

    For each bucket we store the mean and std-dev of hourly ranges
    over the last AHR_LOOKBACK_DAYS days. ETA = pip distance / mean AHR,
    adjusted by current relative ATR vs the session's own average.

    AHR is fetched once here — no additional MT5 calls at refresh time.
    """
    with mo.status.spinner("Booting FinMetrix..."):
        from modules.data_engine import get_mt5_df

        # ── Session bucket boundaries (UTC hours, inclusive start) ────────────
        _SESSIONS = {
            "Asia":     (0,  6),
            "London":   (7,  11),
            "Overlap":  (12, 15),
            "OffPeak":  (16, 23),
        }

        def _session_of(hour):
            for name, (lo, hi) in _SESSIONS.items():
                if lo <= hour <= hi:
                    return name
            return "OffPeak"

        # ── D1 fetch: ADR ─────────────────────────────────────────────────────
        def _fetch_d1(symbol):
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
            _adr_p    = float((_df["high"] - _df["low"]).mean())
            _adr_pips = round(_adr_p / _pip, 1)
            return symbol, _adr_p, _adr_pips

        with ThreadPoolExecutor(max_workers=8) as _ex:
            _d1_res = list(_ex.map(_fetch_d1, cfg.SYMBOLS))

        adr_map      = {}
        adr_pips_map = {}
        for _sym, _ap, _apips in _d1_res:
            if _ap is not None:
                adr_map[_sym]      = _ap
                adr_pips_map[_sym] = _apips

        # ── H1 fetch: session-bucketed AHR ───────────────────────────────────
        # Fetch cfg.AHR_LOOKBACK_DAYS * 24 H1 bars per symbol.
        # Split bars into session buckets, compute mean and std of
        # (high - low) per bucket. Store as:
        #   ahr_map[symbol] = {
        #     "Asia":    {"mean": x, "std": y},
        #     "London":  {"mean": x, "std": y},
        #     "Overlap": {"mean": x, "std": y},
        #     "OffPeak": {"mean": x, "std": y},
        #     "pip":     pip_size,
        #   }

        def _fetch_ahr(symbol):
            _n = cfg.AHR_LOOKBACK_DAYS * 24
            _rates = mt5.copy_rates_from_pos(symbol, cfg.TIMEFRAME, 0, _n)
            if _rates is None or len(_rates) < 50:
                return symbol, None
            _info = mt5.symbol_info(symbol)
            if _info is None:
                return symbol, None
            _pip = _info.point * (10.0 if _info.digits in [3, 5] else 1.0)

            # Build numpy arrays
            _times  = _rates["time"].astype("int64")
            _highs  = _rates["high"].astype(float)
            _lows   = _rates["low"].astype(float)
            _ranges = (_highs - _lows) / _pip   # in pips

            # Convert timestamps to UTC hours
            _hours  = (_times % 86400) // 3600  # 0-23

            _buckets = {}
            for _sname in _SESSIONS:
                _buckets[_sname] = {"mean": 0.0, "std": 0.0}

            for _sname, (_lo, _hi) in _SESSIONS.items():
                _mask = (_hours >= _lo) & (_hours <= _hi)
                _vals = _ranges[_mask]
                if len(_vals) >= 5:
                    _buckets[_sname]["mean"] = float(np.mean(_vals))
                    _buckets[_sname]["std"]  = float(np.std(_vals))
                elif len(_vals) > 0:
                    _buckets[_sname]["mean"] = float(np.mean(_vals))
                    _buckets[_sname]["std"]  = _buckets[_sname]["mean"] * 0.3

            _buckets["pip"] = _pip
            return symbol, _buckets

        with ThreadPoolExecutor(max_workers=8) as _ex2:
            _ahr_res = list(_ex2.map(_fetch_ahr, cfg.SYMBOLS))

        ahr_map = {sym: b for sym, b in _ahr_res if b is not None}

        # ── D1 SMA for daily bias ─────────────────────────────────────────────
        _D1_BARS = max(cfg.ADR_LOOKBACK_DAYS, cfg.D1_SMA_SLOW) + 5

        def _fetch_d1sma(symbol):
            _rates = mt5.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_D1, 0, _D1_BARS
            )
            if _rates is None or len(_rates) < cfg.D1_SMA_SLOW:
                return symbol, None
            _closes = _rates["close"].astype(float)
            _fast   = float(np.mean(_closes[-cfg.D1_SMA_FAST:]))
            _slow   = float(np.mean(_closes[-cfg.D1_SMA_SLOW:]))
            return symbol, _fast > _slow

        with ThreadPoolExecutor(max_workers=8) as _ex3:
            _d1sma_res = list(_ex3.map(_fetch_d1sma, cfg.SYMBOLS))
        d1_bull_map = {sym: b for sym, b in _d1sma_res if b is not None}

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

    # Show AHR sample for EURUSD so you can verify it loaded correctly
    _ahr_sample = ahr_map.get("EURUSD", {})
    _ahr_lines  = "  |  ".join(
        f"{s}: {_ahr_sample.get(s, {}).get('mean', 0):.1f} pips"
        for s in ["Asia", "London", "Overlap", "OffPeak"]
    ) if _ahr_sample else "not loaded"

    mo.md(f"""
    ✅ **FinMetrix ready** — H1 · 5-gate signal · ETA enabled

    | | |
    | :--- | :--- |
    | **Signal logic** | H1 SMA({cfg.H1_SMA_FAST}/{cfg.H1_SMA_SLOW}) · D1 SMA({cfg.D1_SMA_FAST}/{cfg.D1_SMA_SLOW}) · RSI({cfg.RSI_PERIOD_SIG}) MA({cfg.RSI_MA_FAST}/{cfg.RSI_MA_SLOW}) · Rel ATR ≥ {cfg.REL_ATR_MIN} · Vol > avg |
    | **ADR loaded** | {len(adr_map)} / {len(cfg.SYMBOLS)} symbols |
    | **AHR loaded** | {len(ahr_map)} / {len(cfg.SYMBOLS)} symbols |
    | **D1 bias loaded** | {len(d1_bull_map)} / {len(cfg.SYMBOLS)} symbols |
    | **EURUSD AHR** | {_ahr_lines} |
    | **Long / Short rows** | {_n_long:,} / {_n_short:,} |
    | **Kelly risk** | {kelly_risk_pct*100:.2f}% ({_kelly_src}) |
    | **SL / TP1 / TP2** | {cfg.SL_ADR_FRACTION*100:.0f}% / {cfg.TP1_ADR_FRACTION*100:.0f}% / {cfg.TP2_ADR_FRACTION*100:.0f}% of {cfg.ADR_LOOKBACK_DAYS}-day ADR |
    | **ETA auto-close** | age > {cfg.ETA_AGE_MULTIPLIER}× ETA AND R < {cfg.ETA_MIN_R} |
    """)
    return (
        adr_map,
        adr_pips_map,
        ahr_map,
        d1_bull_map,
        kelly_risk_pct,
        model_long,
        model_short,
    )


@app.cell
def _(np):
    def session_of_hour(hour: int) -> str:
        """Returns session name for a UTC hour (0-23)."""
        if  0 <= hour <= 6:  return "Asia"
        if  7 <= hour <= 11: return "London"
        if 12 <= hour <= 15: return "Overlap"
        return "OffPeak"

    def calc_eta(
        pip_distance: float,
        symbol: str,
        current_hour: int,
        rel_atr: float,
        ahr_map: dict,
    ) -> float:
        """
        Estimated hours to cover pip_distance at current session velocity.
        rel_atr > 1.0 = excited market, shorter ETA.
        rel_atr < 1.0 = compression, longer ETA.
        Returns 99.0 if AHR data unavailable.
        """
        _buckets = ahr_map.get(symbol)
        if not _buckets:
            return 99.0
        _session = session_of_hour(current_hour)
        _ahr     = _buckets.get(_session, {}).get("mean", 0.0)
        if _ahr <= 0:
            return 99.0
        _adjusted = _ahr * max(rel_atr, 0.3)
        return round(pip_distance / _adjusted, 1)

    def calc_rsi(closes, period: int):
        """Wilder-smoothed RSI. Returns array aligned with input."""
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

    def sma(arr, period: int):
        """SMA of last `period` values. Returns None if too short."""
        if len(arr) < period:
            return None
        return float(np.mean(arr[-period:]))

    return calc_eta, calc_rsi, session_of_hour, sma


@app.cell
def _(
    ThreadPoolExecutor,
    adr_map,
    adr_pips_map,
    ahr_map,
    balance,
    calc_eta,
    calc_rsi,
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
    refresh_timer,
    session_of_hour,
    set_cooldown,
    sma,
    timezone,
):
    refresh_timer.value

    # ── Fetch H1 raw bars ─────────────────────────────────────────────────────
    def _fetch_h1(symbol):
        _r = mt5.copy_rates_from_pos(symbol, cfg.TIMEFRAME, 0, 300)
        if _r is None or len(_r) == 0:
            return None
        return symbol, _r

    with ThreadPoolExecutor(max_workers=8) as _ex:
        _h1_raw = list(_ex.map(_fetch_h1, cfg.SYMBOLS))
    _h1_map = {s: r for s, r in _h1_raw if r is not None}

    # ── Fetch engineered features (for ML prob and rel ATR) ───────────────────
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

    # ── Cooldown: only blocks if position already open on symbol ──────────────
    _cd = get_cooldown()
    if scan_market:
        _cd = {s: c - 1 for s, c in _cd.items() if c > 1}
        set_cooldown(_cd)

    _positions    = mt5.positions_get() or []
    _open_symbols = {p.symbol for p in _positions}
    _exposed_curr = {c for sym in _open_symbols for c in (sym[:3], sym[3:6])}

    def _is_blocked(symbol):
        if not _open_symbols:
            return False
        return symbol[:3] in _exposed_curr or symbol[3:6] in _exposed_curr

    _now_ts     = datetime.now(timezone.utc)
    _cur_hour   = _now_ts.hour
    signal_rows = []
    today_count = de.signals_today()

    for symbol in cfg.SYMBOLS:
        if symbol not in _h1_map:
            continue

        _h1_closes = _h1_map[symbol]["close"].astype(float)
        _h1_vols   = _h1_map[symbol]["tick_volume"].astype(float)
        _price     = float(_h1_closes[-1])
        _adr       = adr_map.get(symbol)
        _adr_p     = adr_pips_map.get(symbol, 0.0)

        # ── Gate 1: H1 SMA cross ─────────────────────────────────────────────
        _h1_fast = sma(_h1_closes, cfg.H1_SMA_FAST)
        _h1_slow = sma(_h1_closes, cfg.H1_SMA_SLOW)
        if _h1_fast is None or _h1_slow is None:
            continue
        _h1_bull = _h1_fast > _h1_slow

        # ── Gate 2: D1 SMA cross ─────────────────────────────────────────────
        _d1_bull = d1_bull_map.get(symbol)   # None = unavailable

        # ── Gate 3: RSI MA cross ─────────────────────────────────────────────
        _rsi_arr     = calc_rsi(_h1_closes, cfg.RSI_PERIOD_SIG)
        _rsi_valid   = _rsi_arr[~np.isnan(_rsi_arr)]
        _rsi_ma_fast = sma(_rsi_valid, cfg.RSI_MA_FAST)
        _rsi_ma_slow = sma(_rsi_valid, cfg.RSI_MA_SLOW)
        if _rsi_ma_fast is None or _rsi_ma_slow is None:
            continue
        _rsi_bull = _rsi_ma_fast > _rsi_ma_slow

        # ── Gate 4: Relative ATR ─────────────────────────────────────────────
        _rel_atr = 1.0
        if symbol in _feat_map:
            _atr_col = _feat_map[symbol]["atr"].drop_nulls().to_numpy()
            if len(_atr_col) >= 20:
                _rel_atr = float(_atr_col[-1]) / float(np.mean(_atr_col[-20:]))
        _atr_ok = _rel_atr >= cfg.REL_ATR_MIN

        # ── Gate 5: Volume above rolling average ─────────────────────────────
        _vol_avg = float(np.mean(_h1_vols[-cfg.VOL_AVG_BARS:])) if len(_h1_vols) >= cfg.VOL_AVG_BARS else 0.0
        _vol_now  = round(float(_h1_vols[-1] / _vol_avg), 2)

        # ── All five must agree ───────────────────────────────────────────────
        _d1_gate_l = (_d1_bull is True)  or (_d1_bull is None)
        _d1_gate_s = (_d1_bull is False) or (_d1_bull is None)

        _is_long  = _h1_bull        and _d1_gate_l and _rsi_bull        and _atr_ok
        _is_short = (not _h1_bull)  and _d1_gate_s and (not _rsi_bull)  and _atr_ok

        # ── Cooldown only blocks if position open on this symbol ──────────────
        _sym_open     = symbol in _open_symbols
        _corr_blocked = _is_blocked(symbol) and not _sym_open
        _in_cooldown  = (symbol in _cd) and _sym_open

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
        if _adr and _adr > 0:
            _sl_dist  = _adr  * cfg.SL_ADR_FRACTION
            _tp1_dist = _adr  * cfg.TP1_ADR_FRACTION
            _tp2_dist = _adr  * cfg.TP2_ADR_FRACTION
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

        # ── ETA to TP1 ───────────────────────────────────────────────────────
        # Convert pip distance to TP1 into expected hours at current
        # session velocity, adjusted for current market excitement (rel ATR).
        _pip_size    = ahr_map.get(symbol, {}).get("pip", 0.0001)
        _tp1_pips    = (_tp1_dist / _pip_size) if _pip_size > 0 else 0.0
        _eta_h       = calc_eta(_tp1_pips, symbol, _cur_hour, _rel_atr, ahr_map)
        _eta_str     = f"{_eta_h}h" if _eta_h < 99 else "?"
        _session_now = session_of_hour(_cur_hour)

        # ── Log signal on new candle ──────────────────────────────────────────
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
                "eta_h":       _eta_h,
                "session":     _session_now,
                "lot_size":    _lot_size,
                "prob":        _prob,
            })

        # Status tag
        _tag = ""
        if _corr_blocked:
            _tag = " [exp]"
        elif _in_cooldown:
            _tag = f" [cd{_cd.get(symbol, 0)}]"

        _d1_str = "bull" if _d1_bull is True else ("bear" if _d1_bull is False else "?")

        signal_rows.append({
            "Symbol":   symbol,
            "Signal":   signal + _tag,
            "ETA TP1":  _eta_str,
            "ADR pips": _adr_p,
            "Rel ATR":  round(_rel_atr, 2),
            "H1 MA":    "bull" if _h1_bull else "bear",
            "D1 MA":    _d1_str,
            "RSI MA":   "bull" if _rsi_bull else "bear",
            "Vol":      _vol_now,
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
def _(
    acc,
    balance,
    cfg,
    datetime,
    kelly_risk_pct,
    mo,
    session_of_hour,
    signal_rows,
    timezone,
    today_count,
):
    _now_ts2      = datetime.now(timezone.utc)
    _cur_hour2    = _now_ts2.hour
    _session_now2 = session_of_hour(_cur_hour2)

    mo.vstack([
        mo.Html("<style>td, th { font-size: 0.9rem !important; }</style>"),
        mo.md(
            f"### {acc.login} · {acc.currency} {balance:,.2f} · "
            f"{datetime.now().strftime('%H:%M:%S')} · "
            f"Signals today: {today_count} · "
            f"Kelly: {kelly_risk_pct*100:.2f}% · "
            f"Session: {_session_now2}"
        ),
        mo.md(
            f"Signal = H1 SMA({cfg.H1_SMA_FAST}/{cfg.H1_SMA_SLOW}) "
            f"+ D1 SMA({cfg.D1_SMA_FAST}/{cfg.D1_SMA_SLOW}) "
            f"+ RSI({cfg.RSI_PERIOD_SIG}) MA({cfg.RSI_MA_FAST}/{cfg.RSI_MA_SLOW}) "
            f"+ Rel ATR ≥ {cfg.REL_ATR_MIN} + Vol > avg  |  "
            f"ETA = estimated hours to TP1 at current session velocity  |  "
            f"[exp] = currency blocked · [cdN] = cooldown (position open)"
        ),
        mo.ui.table(signal_rows, pagination=False),
    ])
    return


@app.cell
def _(
    adr_map,
    ahr_map,
    calc_eta,
    calc_rsi,
    cfg,
    datetime,
    mo,
    mt5,
    np,
    refresh_timer,
    sma,
    timezone,
):
    refresh_timer.value

    _positions = mt5.positions_get() or []
    _now_ts    = datetime.now(timezone.utc)
    _cur_hour  = _now_ts.hour
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

            if _sl_d <= 0:
                _rows.append({
                    "Symbol": _sym, "Side": _side, "Lots": _lots,
                    "Entry": round(_entry, 5), "Now": round(_price, 5),
                    "Age(h)": "?", "ETA TP1": "?", "R": "—",
                    "SL pips": "—", "SL": "SET SL FIRST",
                    "TP1(50%)": "—", "TP2(25%)": "—", "Trail": "—",
                    "TP1": "—", "TP2": "—", "P&L": round(_pnl, 2),
                    "Flags": "—", "Action": "SET SL FIRST",
                })
                continue

            # ── R and TP targets ──────────────────────────────────────────────
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

            _sl_pips  = round(_sl_d / _pip, 1)

            # ── Position age ──────────────────────────────────────────────────
            _open_dt  = datetime.fromtimestamp(_pos.time, tz=timezone.utc)
            _age_h    = round((_now_ts - _open_dt).total_seconds() / 3600.0, 1)

            # ── Current relative ATR for this symbol ──────────────────────────
            _rel_atr_pm = 1.0
            _h1_pm = mt5.copy_rates_from_pos(_sym, cfg.TIMEFRAME, 0, 30)
            if _h1_pm is not None and len(_h1_pm) >= 20:
                _h1_ranges = (_h1_pm["high"] - _h1_pm["low"]).astype(float)
                _cur_range = float(_h1_ranges[-1])
                _avg_range = float(np.mean(_h1_ranges[-20:]))
                if _avg_range > 0:
                    _rel_atr_pm = _cur_range / _avg_range

            # ── ETA to TP1 ────────────────────────────────────────────────────
            # Remaining pip distance from current price to TP1.
            # Uses session-bucketed AHR adjusted by current rel ATR.
            _pip_size_pm = ahr_map.get(_sym, {}).get("pip", _pip)
            if _tp1_hit:
                _rem_pips = 0.0
            elif _side == "BUY":
                _rem_pips = max(0.0, (_tp1_p - _price) / _pip_size_pm)
            else:
                _rem_pips = max(0.0, (_price - _tp1_p) / _pip_size_pm)

            _eta_h = calc_eta(_rem_pips, _sym, _cur_hour, _rel_atr_pm, ahr_map)

            # ── ETA at entry (full TP1 distance) — for auto-close comparison ─
            _full_tp1_pips = _tp1_d / _pip_size_pm if _pip_size_pm > 0 else 0.0
            # We use the ETA that was valid at entry hour — approximate with
            # current session since we don't store entry session. Conservative.
            _entry_eta = calc_eta(_full_tp1_pips, _sym, _cur_hour, 1.0, ahr_map)

            # ── EXIT FLAGS ────────────────────────────────────────────────────
            _flags = []

            # RSI divergence: H1 MA and RSI MA disagree (after >= 2h open)
            if _age_h >= 2.0 and _h1_pm is not None and len(_h1_pm) >= cfg.H1_SMA_SLOW:
                _h1_c    = _h1_pm["close"].astype(float)
                _h1f     = sma(_h1_c, cfg.H1_SMA_FAST)
                _h1s     = sma(_h1_c, cfg.H1_SMA_SLOW)
                _rsi_pm  = calc_rsi(_h1_c, cfg.RSI_PERIOD_SIG)
                _rsiv_pm = _rsi_pm[~np.isnan(_rsi_pm)]
                _rsi_f   = sma(_rsiv_pm, cfg.RSI_MA_FAST)
                _rsi_s   = sma(_rsiv_pm, cfg.RSI_MA_SLOW)
                if _h1f and _h1s and _rsi_f and _rsi_s:
                    _h1_bull_now  = _h1f > _h1s
                    _rsi_bull_now = _rsi_f > _rsi_s
                    _diverge = (
                        (_side == "BUY"  and (not _h1_bull_now or not _rsi_bull_now)) or
                        (_side == "SELL" and (_h1_bull_now or _rsi_bull_now))
                    )
                    if _diverge:
                        _flags.append("RSI")

            # ATR compression flag
            if _rel_atr_pm < cfg.REL_ATR_COMPRESS:
                _flags.append("ATR")

            # ADR exhaustion flag
            _d1_today = mt5.copy_rates_from_pos(_sym, mt5.TIMEFRAME_D1, 0, 2)
            if _d1_today is not None and len(_d1_today) >= 1 and _adr and _adr > 0:
                _used = float(_d1_today["high"][-1]) - float(_d1_today["low"][-1])
                if (_used / _adr) * 100.0 >= cfg.ADR_USED_WARN_PCT:
                    _flags.append("ADR")

            _flags_str = " ".join(_flags) if _flags else "—"

            # ── AUTO-CLOSE ACTIONS ────────────────────────────────────────────
            _action    = "Hold"
            _auto_done = False

            # ETA-based auto-close: if age > ETA_AGE_MULTIPLIER × entry ETA
            # AND R < ETA_MIN_R, the trade is running much slower than
            # expected — close it.
            _eta_threshold = _entry_eta * cfg.ETA_AGE_MULTIPLIER
            if (not _auto_done
                    and _entry_eta < 99.0          # ETA data available
                    and _age_h >= _eta_threshold
                    and _r < cfg.ETA_MIN_R
                    and _info):
                _res = mt5.order_send({
                    "action":    mt5.TRADE_ACTION_DEAL,
                    "position":  _ticket,
                    "symbol":    _sym,
                    "volume":    _lots,
                    "type":      mt5.ORDER_TYPE_SELL if _side == "BUY" else mt5.ORDER_TYPE_BUY,
                    "price":     _price,
                    "deviation": 20,
                    "comment":   "FM_ETA_autoclose",
                })
                if _res and _res.retcode == mt5.TRADE_RETCODE_DONE:
                    _action    = f"✅ ETA closed (age={_age_h}h > {_eta_threshold:.1f}h, R={_r})"
                    _auto_done = True
                    _auto_log.append(f"{_sym} ETA→closed")
                else:
                    _action = f"⚠ ETA close failed (rc={_res.retcode if _res else '?'})"

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

            # Advisory suggestions
            if not _auto_done:
                if _tp2_hit:
                    _action = "Trail final 25% (manual)"
                elif _tp1_hit:
                    _action = "Close 25% + BE (manual)"
                elif _flags:
                    _action = f"Review: {_flags_str}"
                elif _age_h > _eta_threshold * 0.7 and _r < 0.2:
                    _action = f"Slowing — ETA was {_entry_eta}h"
                elif _r >= 0.8:
                    _action = "Approaching TP1"
                elif _r < -0.5:
                    _action = "Under pressure"

            _eta_disp = f"{_eta_h}h" if _eta_h < 99 else "?"

            _rows.append({
                "Symbol":   _sym,      "Side":      _side,
                "Lots":     _lots,     "Entry":     round(_entry, 5),
                "Now":      round(_price, 5),
                "Age(h)":   _age_h,    "ETA TP1":   _eta_disp,
                "R":        _r,        "SL pips":   _sl_pips,
                "SL":       round(_sl, 5) if _sl else "—",
                "TP1(50%)": _tp1_p,    "TP2(25%)":  _tp2_p,
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
                f"**ETA auto-close:** age > {cfg.ETA_AGE_MULTIPLIER}× entry ETA AND R < {cfg.ETA_MIN_R}  |  "
                f"**Flags:** RSI=diverging · ATR=compressing · ADR=exhausted"
            ),
            mo.ui.table(_rows, pagination=False),
        ])

    _mgmt
    return


if __name__ == "__main__":
    app.run()
