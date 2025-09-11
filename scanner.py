# scanner.py
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objs as go


# ─────────────────────────────────────────────────────────────
# Helpers & Indicators
# ─────────────────────────────────────────────────────────────

def fmt_num(x, d=2) -> str:
    """Safely format numeric values; returns '—' for non-numbers/NaN."""
    try:
        xf = float(x)
        if np.isnan(xf):
            return "—"
        return f"{xf:.{d}f}"
    except Exception:
        return "—"


def rsi_wilder(df_or_s, length: int = 14) -> pd.Series:
    """
    Wilder's RSI. Always returns a Series.
    Accepts either a Series or a single-column DataFrame.
    """
    if isinstance(df_or_s, pd.DataFrame):
        s = df_or_s.iloc[:, 0].astype(float).copy()
    else:
        s = pd.Series(df_or_s, copy=True).astype(float)

    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_sp500_tickers():
    """Scrape S&P 500 tickers (cached)."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
        return sorted(list(set(tickers)))
    except Exception:
        # Fallback
        return sorted(list(set([
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "XOM", "UNH"
        ])))


def latest_cross_flags(ma_fast: pd.Series, ma_slow: pd.Series, lookback: int):
    """
    Detect MA cross in the last `lookback` bars.
    Returns (golden_cross_recent, death_cross_recent).
    """
    cross_up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    cross_dn = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    gc = cross_up.rolling(lookback, min_periods=1).max().fillna(0).iloc[-1] == 1
    dc = cross_dn.rolling(lookback, min_periods=1).max().fillna(0).iloc[-1] == 1
    return bool(gc), bool(dc)


def macd(series: pd.Series):
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    line = (ema_fast - ema_slow).astype(float)
    signal = line.ewm(span=9, adjust=False).mean().astype(float)
    hist = (line - signal).astype(float)
    return line, signal, hist


def score_rsi_sweet(rsi_value: float):
    """Score 0..1, best near 50; symmetrical fade to 0 at 0 and 100."""
    if np.isnan(rsi_value):
        return 0.0
    return max(0.0, 1.0 - abs(rsi_value - 50.0) / 50.0)


def score_trend(price, ma50, ma200):
    """0..1; half point for price>MA50 and half for MA50>MA200."""
    s = 0.0
    if not any(np.isnan([price, ma50, ma200])):
        s += 0.5 if price > ma50 else 0.0
        s += 0.5 if ma50 > ma200 else 0.0
    return s


def score_macd_momentum(line_last, sig_last, price_series):
    """
    Normalize MACD momentum by recent volatility; squash to 0..1 via tanh.
    """
    if any(np.isnan([line_last, sig_last])) or price_series.isna().all():
        return 0.0
    mom = float(line_last - sig_last)
    vol = float(price_series.pct_change().std())
    z = mom if (vol <= 0 or np.isnan(vol)) else mom / (vol * 5.0)
    return float((np.tanh(z) + 1) / 2.0)


@st.cache_data(show_spinner=True, ttl=15 * 60)
def download_prices(tickers, period="1y"):
    """
    Download adjusted close for all tickers with yfinance (grouped).
    Returns dict ticker -> Series (Close/Adj Close).
    """
    data = {}
    if not tickers:
        return data

    try:
        df = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        df = None

    if df is None or df.empty:
        for t in tickers:
            try:
                d = yf.download(t, period=period, auto_adjust=True, progress=False)
                if isinstance(d, pd.DataFrame) and not d.empty:
                    s = d.get("Close", d.get("Adj Close"))
                    if s is not None:
                        s = s.dropna()
                        s.index = pd.to_datetime(s.index)
                        data[t] = s.astype(float)
            except Exception:
                pass
        return data

    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = df[t]["Close"].dropna()
                s.index = pd.to_datetime(s.index)
                data[t] = s.astype(float)
            except Exception:
                pass
    else:
        s = df.get("Close", df.get("Adj Close"))
        if s is not None:
            s = s.dropna()
            s.index = pd.to_datetime(s.index)
            data[tickers[0]] = s.astype(float)

    return data


# ─────────────────────────────────────────────────────────────
# Password Gate (optional)
# ─────────────────────────────────────────────────────────────

def check_password():
    expected = st.secrets.get("password", {}).get("app_password")
    if not expected:
        return True

    def pass_entered():
        st.session_state["password_ok"] = (
            st.session_state.get("password_input", "") == expected
        )

    if "password_ok" not in st.session_state:
        st.text_input(
            "Enter password", type="password", on_change=pass_entered, key="password_input"
        )
        st.stop()

    if not st.session_state["password_ok"]:
        st.error("Incorrect password")
        st.text_input(
            "Enter password", type="password", on_change=pass_entered, key="password_input"
        )
        st.stop()

    return True


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="StockPeers Screener", layout="wide", page_icon="📈")

if check_password():
    st.title("StockPeers Screener")

    # Sidebar filters
    st.sidebar.subheader("Filters")

    sp_tickers = get_sp500_tickers()
    st.sidebar.caption(f"S&P 500 universe loaded: **{len(sp_tickers)}** tickers")

    price_period = st.sidebar.selectbox("Price history period (for metrics)", ["1y", "2y"], index=0)
    rsi_min, rsi_max = st.sidebar.slider("RSI Range (14)", 0, 100, (30, 70))

    pv50 = st.sidebar.selectbox("Price vs 50-day MA", ["Any", "Above", "Below"], index=0)
    pv200 = st.sidebar.selectbox("Price vs 200-day MA", ["Any", "Above", "Below"], index=0)

    st.sidebar.markdown("---")
    cross_req = st.sidebar.checkbox("Require recent MA crossover (50 vs 200)?", value=False)
    cx_type = st.sidebar.selectbox("Crossover Type", ["Golden", "Death"], index=0, disabled=not cross_req)
    cx_look = st.sidebar.number_input("Lookback days for crossover", min_value=2, max_value=90, value=20, step=1, disabled=not cross_req)

    st.sidebar.markdown("---")
    st.sidebar.subheader("MACD")
    macd_enable = st.sidebar.checkbox("Enable MACD filter", value=False)
    macd_cond = st.sidebar.selectbox(
        "MACD condition",
        ["Line > Signal", "Line < Signal", "Bullish cross (recent)", "Bearish cross (recent)"],
        index=0,
        disabled=not macd_enable,
    )
    macd_look = st.sidebar.number_input(
        "Lookback days for MACD cross", min_value=2, max_value=60, value=10, step=1, disabled=not macd_enable
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Scoring & Ranking")
    use_score = st.sidebar.checkbox("Enable ranking / composite score", value=True)
    w_rsi = st.sidebar.slider("Weight: RSI sweet spot", 0.0, 2.0, 1.0, 0.05)
    w_trend = st.sidebar.slider("Weight: Trend vs MAs", 0.0, 2.0, 1.0, 0.05)
    w_macd = st.sidebar.slider("Weight: MACD momentum", 0.0, 2.0, 1.0, 0.05)
    w_cross = st.sidebar.slider("Weight: Golden/Death boost", -1.0, 2.0, 0.5, 0.05)
    top_n = st.sidebar.number_input("Show Top N", 5, 50, 25, 1)

    # ─────────────────────────────────────────────────────────
    # Compute metrics
    # ─────────────────────────────────────────────────────────
    with st.spinner("Downloading price data…"):
        price_map = download_prices(sp_tickers, period=price_period)

    rows = []
    for t, s in price_map.items():
        if s.size < 60:
            continue

        ma50 = s.rolling(50, min_periods=1).mean()
        ma200 = s.rolling(200, min_periods=1).mean()
        rsi14 = rsi_wilder(s, 14).astype(float)  # <— always Series
        m_line, m_sig, _ = macd(s)

        # Recent cross flags (50 vs 200)
        gc_recent, dc_recent = latest_cross_flags(ma50, ma200, lookback=int(cx_look))

        last_price = float(s.iloc[-1])
        last_ma50 = float(ma50.iloc[-1])
        last_ma200 = float(ma200.iloc[-1])
        last_rsi = float(rsi14.iloc[-1])
        last_macd = float(m_line.iloc[-1])
        last_sig = float(m_sig.iloc[-1])

        # Filters
        if not (rsi_min <= last_rsi <= rsi_max):
            continue
        if pv50 == "Above" and not (last_price > last_ma50):
            continue
        if pv50 == "Below" and not (last_price < last_ma50):
            continue
        if pv200 == "Above" and not (last_price > last_ma200):
            continue
        if pv200 == "Below" and not (last_price < last_ma200):
            continue
        if cross_req:
            if cx_type == "Golden" and not gc_recent:
                continue
            if cx_type == "Death" and not dc_recent:
                continue
        if macd_enable:
            cross_up = (m_line > m_sig) & (m_line.shift(1) <= m_sig.shift(1))
            cross_dn = (m_line < m_sig) & (m_line.shift(1) >= m_sig.shift(1))
            bull_recent = bool(cross_up.rolling(int(macd_look), min_periods=1).max().iloc[-1] == 1)
            bear_recent = bool(cross_dn.rolling(int(macd_look), min_periods=1).max().iloc[-1] == 1)

            if macd_cond == "Line > Signal" and not (last_macd > last_sig):
                continue
            if macd_cond == "Line < Signal" and not (last_macd < last_sig):
                continue
            if macd_cond == "Bullish cross (recent)" and not bull_recent:
                continue
            if macd_cond == "Bearish cross (recent)" and not bear_recent:
                continue

        # Scoring
        rsi_score = score_rsi_sweet(last_rsi)
        trend_score = score_trend(last_price, last_ma50, last_ma200)
        macd_score = score_macd_momentum(last_macd, last_sig, s)
        cross_boost = (1.0 if gc_recent else 0.0) - (1.0 if dc_recent else 0.0)

        composite = (w_rsi * rsi_score) + (w_trend * trend_score) + (w_macd * macd_score) + (w_cross * cross_boost)

        rows.append(
            {
                "Ticker": t,
                "Score": composite,
                "Price": last_price if last_price < 50000 else last_price / 100.0,  # avoid huge axis values
                "RSI(14)": last_rsi,
                "MA50": last_ma50,
                "MA200": last_ma200,
            }
        )

    results = pd.DataFrame(rows).set_index("Ticker")
    if results.empty:
        st.warning("No matches with current filters.")
        st.stop()

    if use_score:
        results = results.sort_values("Score", ascending=False)
    results.insert(0, "Rank", range(1, len(results) + 1))
    st.success(f"Found {len(results)} match(es).")
    st.dataframe(results.head(int(top_n)), use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # Chart (robust)
    # ─────────────────────────────────────────────────────────
    with st.expander("Open full interactive chart"):
        sel = st.selectbox("Choose ticker", results.index.tolist(), key="chart_ticker")
        chart_period = st.selectbox("Chart period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=1, key="chart_period")
        view_mode = st.radio(
            "View mode",
            ["Price (actual)", "% change (rebased)", "Log price"],
            index=0,
            horizontal=True,
            help="Use % change to compare moves; Log helps long horizons.",
        )

        def fetch_series(ticker, per):
            interval = "1d" if per in ["6mo", "1y", "2y"] else "1wk"
            df = yf.download(ticker, period=per, interval=interval, auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame):
                if "Close" in df.columns:
                    s = df["Close"].dropna()
                elif "Adj Close" in df.columns:
                    s = df["Adj Close"].dropna()
                else:
                    s = df.select_dtypes(include=[np.number]).iloc[:, -1].dropna()
            else:
                s = pd.Series(dtype=float)
            try:
                s.index = pd.to_datetime(s.index)
            except Exception:
                pass
            return s.astype(float)

        def fetch_volume(ticker, per):
            interval = "1d" if per in ["6mo", "1y", "2y"] else "1wk"
            df = yf.download(ticker, period=per, interval=interval, auto_adjust=False, progress=False)
            if isinstance(df, pd.DataFrame) and "Volume" in df.columns:
                v = df["Volume"].dropna()
                try:
                    v.index = pd.to_datetime(v.index)
                except Exception:
                    pass
                return v
            return pd.Series(dtype=float)

        if not sel:
            st.info("Choose a ticker above to draw the chart.")
        else:
            s = fetch_series(sel, chart_period)
            widen = {"6mo": "1y", "1y": "2y", "2y": "5y", "5y": "10y", "10y": "max"}
            p = chart_period
            while len(s) < 60 and p in widen:
                p = widen[p]
                s = fetch_series(sel, p)

            if s.empty:
                st.info("No chartable data returned. Try a longer chart period.")
            else:
                ma50 = s.rolling(50, min_periods=1).mean()
                ma200 = s.rolling(200, min_periods=1).mean()
                rsi14 = rsi_wilder(s, 14).astype(float)   # <— RSI from Series
                ema_fast = s.ewm(span=12, adjust=False).mean()
                ema_slow = s.ewm(span=26, adjust=False).mean()
                macd_line = (ema_fast - ema_slow).astype(float)
                macd_signal = macd_line.ewm(span=9, adjust=False).mean().astype(float)
                macd_hist = (macd_line - macd_signal).astype(float)

                v = fetch_volume(sel, p)
                if not v.empty:
                    v = v.loc[s.index.min() : s.index.max()]
                have_volume = not v.empty

                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    row_heights=[0.55, 0.15, 0.15, 0.15],
                    vertical_spacing=0.03,
                )

                if view_mode == "% change (rebased)":
                    base = float(s.iloc[0])
                    y = (s / base - 1.0) * 100.0
                    fig.add_trace(go.Scatter(x=y.index, y=y, name="% change", mode="lines"), row=1, col=1)
                    fig.add_trace(
                        go.Scatter(x=ma50.index, y=(ma50 / base - 1.0) * 100.0, name="MA50", mode="lines"), row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=ma200.index, y=(ma200 / base - 1.0) * 100.0, name="MA200", mode="lines"),
                        row=1,
                        col=1,
                    )
                    price_title = "Return (%)"
                    ytype = "linear"
                else:
                    fig.add_trace(go.Scatter(x=s.index, y=s, name="Adj Close", mode="lines"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="MA50", mode="lines"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="MA200", mode="lines"), row=1, col=1)
                    price_title = "Price"
                    ytype = "log" if view_mode == "Log price" else "linear"

                if have_volume:
                    fig.add_trace(
                        go.Bar(x=v.index, y=v, name="Volume", marker_color="rgba(70,130,180,0.6)"), row=2, col=1
                    )

                fig.add_trace(
                    go.Scatter(x=rsi14.index, y=rsi14, name="RSI(14)", mode="lines", line=dict(color="#666")),
                    row=3,
                    col=1,
                )
                fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="rgba(200,200,200,0.18)", row=3, col=1)
                fig.add_hline(y=30, line_width=1, line_color="crimson", row=3, col=1)
                fig.add_hline(y=70, line_width=1, line_color="seagreen", row=3, col=1)

                hist = macd_hist.fillna(0.0)
                colors = np.where(hist >= 0, "rgba(0,160,0,0.6)", "rgba(200,0,0,0.6)")
                fig.add_trace(go.Bar(x=hist.index, y=hist, name="Hist", marker_color=colors), row=4, col=1)
                fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD", mode="lines"), row=4, col=1)
                fig.add_trace(go.Scatter(x=macd_signal.index, y=macd_signal, name="Signal", mode="lines"), row=4, col=1)
                fig.add_hline(y=0, line_width=1, line_color="gray", row=4, col=1)

                fig.update_layout(
                    height=860,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_rangeslider_visible=False,
                )
                fig.update_yaxes(type=ytype, title_text=price_title, row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1, tickformat=",.0f")
                fig.update_yaxes(title_text="RSI(14)", row=3, col=1, range=[0, 100])
                fig.update_yaxes(title_text="MACD", row=4, col=1)

                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"{sel} | Points: {len(s)} | "
                    f"Last Price: {fmt_num(s.iloc[-1], 2)} | "
                    f"RSI(14): {fmt_num(rsi14.iloc[-1], 1)} | "
                    f"MA50/MA200: {fmt_num(ma50.iloc[-1], 2)} / {fmt_num(ma200.iloc[-1], 2)}"
                )
