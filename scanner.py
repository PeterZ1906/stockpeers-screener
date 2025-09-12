# scanner.py
# Streamlit app: StockPeers Screener (S&P 500 by default)
# -------------------------------------------------------
# - Technical filters: RSI range, price vs MA50/MA200, recent MA crossover, MACD
# - Optional fundamentals: simple market-cap bounds (disabled by default)
# - Composite score & ranking
# - Interactive Plotly chart with Price+MA, Volume, RSI and MACD
# - Caching for downloads; robust to yfinance MultiIndex layouts
# -------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# -----------------------------
# ---------- Utils ------------
# -----------------------------

st.set_page_config(page_title="StockPeers Screener", layout="wide")

# Small, safe theme tweaks
PRIMARY = "#2E86DE"
BAD = "#C0392B"
GOOD = "#1E8449"


# A small, static S&P 500 fallback set.
# We keep the list short to minimize yfinance throttling in free tier;
# the live function below tries to fetch the full list when available.
SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "BRK-B", "UNH", "XOM", "JNJ",
    "JPM", "V", "PG", "HD", "MA", "LLY", "MRK",
    "AVGO", "PEP", "ABBV", "KO", "COST", "PFE",
    "MCD", "ADBE", "CSCO", "TMO", "DHR",
]

PERIOD_OPTIONS = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}


# -----------------------------
# ----- Data acquisition ------
# -----------------------------

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_sp500_tickers() -> List[str]:
    """
    Try to fetch S&P 500 tickers. If pandas.read_html/lxml is not available
    (common on free environments), fall back to a compact static list.
    """
    try:
        # The table on Wikipedia is often available but needs lxml.
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            flavor=None  # let pandas auto-select parser if available
        )
        df = tables[0]
        # Some tickers contain dots (e.g., BRK.B). yfinance expects hyphen.
        tickers = (
            df["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        # Keep it to a reasonable size to avoid throttling
        if len(tickers) > 200:
            tickers = tickers[:200]
        return tickers
    except Exception:
        return SP500_FALLBACK


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def download_prices(tickers: List[str], period: str) -> pd.DataFrame:
    """
    Download daily prices for multiple tickers.
    Returns a wide DataFrame of Adjusted Close (or Close) with Date index.
    Robust to the MultiIndex structure returned by yfinance.
    """
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # Multi-ticker -> MultiIndex (top level=ticker, second level=field)
    if isinstance(data.columns, pd.MultiIndex):
        # Determine which level contains "Adj Close"/"Close"
        field_level = None
        for lvl in range(data.columns.nlevels):
            values = data.columns.get_level_values(lvl).unique().tolist()
            if ("Adj Close" in values) or ("Close" in values):
                field_level = lvl
                break
        if field_level is None:
            return pd.DataFrame()

        fields = data.columns.get_level_values(field_level).unique().tolist()
        target = "Adj Close" if "Adj Close" in fields else "Close"

        # Take a cross-section at the field level -> columns become tickers
        close = data.xs(target, axis=1, level=field_level)

        # Keep tickers in the original order, dropping missing ones
        existing = [t for t in tickers if t in close.columns]
        close = close[existing]

    else:
        # Single ticker case
        key = "Adj Close" if "Adj Close" in data.columns else "Close"
        close = data[[key]].copy()
        t = tickers[0] if tickers else "TICKER"
        close.columns = [t]

    close = close.sort_index()
    close = close.dropna(how="all")
    return close


# -----------------------------
# ------ TA Calculations ------
# -----------------------------

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder's RSI implementation; returns float series between 0..100.
    """
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def macd_calc(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def recent_cross(up: pd.Series, down: pd.Series, lookback: int) -> Tuple[pd.Series, pd.Series]:
    """
    Returns a tuple (golden_cross_recent, death_cross_recent), boolean series.
    Golden cross: up crosses above down.
    Death cross: up crosses below down.
    """
    cross_up = (up > down) & (up.shift(1) <= down.shift(1))
    cross_down = (up < down) & (up.shift(1) >= down.shift(1))
    gc_recent = cross_up.rolling(lookback, min_periods=1).max().astype(bool)
    dc_recent = cross_down.rolling(lookback, min_periods=1).max().astype(bool)
    return gc_recent, dc_recent


# -----------------------------
# ---- Compute Indicators -----
# -----------------------------

def compute_indicators(close: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Given wide Adjusted Close prices, compute indicators per ticker.
    Returns dict of DataFrames, each DataFrame is wide (Date index, columns=tickers).
    """
    ma50 = close.apply(lambda s: moving_average(s, 50))
    ma200 = close.apply(lambda s: moving_average(s, 200))
    rsi14 = close.apply(lambda s: rsi_wilder(s, 14))
    macd_line = pd.DataFrame(index=close.index)
    signal_line = pd.DataFrame(index=close.index)
    hist = pd.DataFrame(index=close.index)
    for c in close.columns:
        m, sig, h = macd_calc(close[c])
        macd_line[c] = m
        signal_line[c] = sig
        hist[c] = h

    gc_recent = pd.DataFrame(index=close.index)
    dc_recent = pd.DataFrame(index=close.index)
    for c in close.columns:
        gc, dc = recent_cross(ma50[c], ma200[c], lookback=20)
        gc_recent[c] = gc
        dc_recent[c] = dc

    return dict(
        close=close,
        ma50=ma50,
        ma200=ma200,
        rsi14=rsi14,
        macd=macd_line,
        macd_signal=signal_line,
        macd_hist=hist,
        gc_recent=gc_recent,
        dc_recent=dc_recent,
    )


# -----------------------------
# ---------- Filters ----------
# -----------------------------

@dataclass
class Settings:
    universe: List[str]
    period: str
    rsi_min: float
    rsi_max: float
    price_vs_ma50: str  # Any / Above / Below
    price_vs_ma200: str
    require_recent_cross: bool
    cross_type: str      # Golden / Death
    cross_lookback: int

    macd_enable: bool
    macd_condition: str  # Line > Signal / Line < Signal / Bull cross / Bear cross
    macd_lookback: int

    fundamentals_enable: bool
    mcap_min: Optional[float]
    mcap_max: Optional[float]

    # ranking
    ranking_enable: bool
    weight_rsi: float
    weight_trend: float
    weight_macd: float
    weight_cross: float
    top_n: int


def pass_price_vs_ma(relation: str, price: float, ma: float) -> bool:
    if np.isnan(price) or np.isnan(ma):
        return False
    if relation == "Any":
        return True
    if relation == "Above":
        return price >= ma
    if relation == "Below":
        return price <= ma
    return True


def macd_filter(cond: str,
                macd_line: float,
                signal_line: float,
                hist_series: pd.Series,
                lookback: int) -> bool:
    if np.isnan(macd_line) or np.isnan(signal_line):
        return False
    if cond == "Line > Signal":
        return macd_line > signal_line
    if cond == "Line < Signal":
        return macd_line < signal_line
    if cond == "Bull cross (last N days)":
        # Hist turns positive
        return (hist_series.tail(lookback) > 0).any() and (hist_series.shift(1).tail(lookback) <= 0).any()
    if cond == "Bear cross (last N days)":
        return (hist_series.tail(lookback) < 0).any() and (hist_series.shift(1).tail(lookback) >= 0).any()
    return True


def fundamentals_gate(enabled: bool, ticker: str, min_cap: Optional[float], max_cap: Optional[float]) -> bool:
    if not enabled:
        return True
    try:
        info = yf.Ticker(ticker).fast_info  # fast_info is cheaper than .info
        mcap = float(info.get("market_cap") or np.nan)
        if np.isnan(mcap):
            return False
        if min_cap is not None and mcap < min_cap:
            return False
        if max_cap is not None and mcap > max_cap:
            return False
        return True
    except Exception:
        # If we cannot fetch, be conservative and exclude
        return False


# -----------------------------
# --------- Scoring -----------
# -----------------------------

def score_one(
    tkr: str,
    idx: int,
    ind: Dict[str, pd.DataFrame],
    cfg: Settings,
) -> Optional[Dict]:
    """
    Returns dict with fields or None if it fails any filter.
    """
    close = ind["close"][tkr].iloc[: idx + 1]
    ma50 = ind["ma50"][tkr].iloc[: idx + 1]
    ma200 = ind["ma200"][tkr].iloc[: idx + 1]
    rsi = ind["rsi14"][tkr].iloc[: idx + 1]
    macd_line = ind["macd"][tkr].iloc[: idx + 1]
    macd_sig = ind["macd_signal"][tkr].iloc[: idx + 1]
    hist = ind["macd_hist"][tkr].iloc[: idx + 1]
    gc_recent = ind["gc_recent"][tkr].iloc[: idx + 1]
    dc_recent = ind["dc_recent"][tkr].iloc[: idx + 1]

    if close.empty:
        return None

    last_p = float(close.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_ma50 = float(ma50.iloc[-1]) if not np.isnan(ma50.iloc[-1]) else np.nan
    last_ma200 = float(ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else np.nan
    last_macd = float(macd_line.iloc[-1])
    last_sig = float(macd_sig.iloc[-1])

    # RSI range
    if not (cfg.rsi_min <= last_rsi <= cfg.rsi_max):
        return None

    # Price vs MAs
    if not pass_price_vs_ma(cfg.price_vs_ma50, last_p, last_ma50):
        return None
    if not pass_price_vs_ma(cfg.price_vs_ma200, last_p, last_ma200):
        return None

    # Recent cross
    if cfg.require_recent_cross:
        look = cfg.cross_lookback
        if cfg.cross_type == "Golden":
            if not bool(gc_recent.tail(look).any()):
                return None
        else:
            if not bool(dc_recent.tail(look).any()):
                return None

    # MACD filter
    if cfg.macd_enable:
        if not macd_filter(cfg.macd_condition, last_macd, last_sig, hist, cfg.macd_lookback):
            return None

    # Fundamentals
    if not fundamentals_gate(cfg.fundamentals_enable, tkr, cfg.mcap_min, cfg.mcap_max):
        return None

    # ---------- scoring ----------
    score = 0.0
    weights = 0.0

    if cfg.ranking_enable:
        # (1) RSI sweet spot (~50-60) -> score uses Gaussian centered at 55
        w = cfg.weight_rsi
        comp = math.exp(-((last_rsi - 55.0) ** 2) / (2 * 12.0 ** 2))
        score += w * comp
        weights += w

        # (2) Trend vs MAs: higher if price above both and MA50 > MA200
        w = cfg.weight_trend
        trend = 0.0
        if not np.isnan(last_ma50) and not np.isnan(last_ma200):
            trend += 0.5 if last_p >= last_ma50 else -0.2
            trend += 0.5 if last_p >= last_ma200 else -0.2
            trend += 0.5 if last_ma50 >= last_ma200 else -0.5
        score += w * (trend / 1.5)  # normalize roughly between -1..1
        weights += w

        # (3) MACD momentum: histogram z-score-ish
        w = cfg.weight_macd
        h_tail = hist.tail(20).dropna()
        macd_mom = float(h_tail.mean()) if not h_tail.empty else 0.0
        score += w * (1 / (1 + math.exp(-5 * macd_mom)))  # squashed 0..1
        weights += w

        # (4) Golden cross recency bonus
        w = cfg.weight_cross
        cross_bonus = 1.0 if bool(gc_recent.tail(60).any()) else 0.0
        score += w * cross_bonus
        weights += w

    final = score / weights if weights > 0 else 0.0

    return dict(
        Ticker=tkr,
        Price=last_p,
        RSI14=last_rsi,
        MA50=last_ma50,
        MA200=last_ma200,
        MACD=last_macd,
        MACDSignal=last_sig,
        Score=final,
    )


def run_scan_once(cfg: Settings) -> pd.DataFrame:
    tickers = cfg.universe
    close = download_prices(tickers, cfg.period)
    if close.empty:
        return pd.DataFrame()

    ind = compute_indicators(close)
    idx_last = len(ind["close"].index) - 1
    if idx_last < 0:
        return pd.DataFrame()

    rows = []
    for t in close.columns:
        res = score_one(t, idx_last, ind, cfg)
        if res:
            rows.append(res)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if cfg.ranking_enable:
        df = df.sort_values("Score", ascending=False)
        df.insert(0, "Rank", range(1, len(df) + 1))
    else:
        df = df.sort_values("Ticker")
    return df.reset_index(drop=True)


# -----------------------------
# --------- Charting ----------
# -----------------------------

def plot_interactive(
    tkr: str,
    ind: Dict[str, pd.DataFrame],
) -> go.Figure:
    close = ind["close"][tkr]
    ma50 = ind["ma50"][tkr]
    ma200 = ind["ma200"][tkr]
    rsi = ind["rsi14"][tkr]
    macd_line = ind["macd"][tkr]
    macd_sig = ind["macd_signal"][tkr]
    hist = ind["macd_hist"][tkr]

    fig = make_price_rsi_macd(close, ma50, ma200, rsi, macd_line, macd_sig, hist, title=f"{tkr} — Interactive")

    return fig


def make_price_rsi_macd(
    close: pd.Series,
    ma50: pd.Series,
    ma200: pd.Series,
    rsi: pd.Series,
    macd_line: pd.Series,
    macd_sig: pd.Series,
    hist: pd.Series,
    title: str = "Interactive Chart",
) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=("Price", "RSI (14)", "MACD")
    )

    # Price + MAs
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Adj Close", line=dict(color=PRIMARY, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="MA50", line=dict(color="#6ab04c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="MA200", line=dict(color="#e74c3c")), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI(14)", line=dict(color="#7f8c8d")), row=2, col=1)
    fig.add_hline(y=70, line=dict(color=BAD, width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=GOOD, width=1, dash="dot"), row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, name="MACD", line=dict(color="#8e44ad")), row=3, col=1)
    fig.add_trace(go.Scatter(x=macd_sig.index, y=macd_sig, name="Signal", line=dict(color="#f39c12")), row=3, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist, name="Hist",
                         marker_color=np.where(hist >= 0, "#27ae60", "#c0392b"),
                         opacity=0.6), row=3, col=1)

    fig.update_layout(
        title=title,
        height=760,
        showlegend=True,
        margin=dict(t=40, r=10, l=10, b=10),
    )
    return fig


# Helper because Plotly subplots are used above
from plotly.subplots import make_subplots


# -----------------------------
# ------------ UI -------------
# -----------------------------

def render_filters() -> Settings:
    st.sidebar.title("Filters")

    # Universe
    with st.sidebar.expander("Universe", expanded=True):
        universe_choice = st.radio(
            "Stock universe",
            ["S&P 500 (auto)", "Fallback sample (fast)"],
            index=0,
        )
        if universe_choice == "S&P 500 (auto)":
            universe = get_sp500_tickers()
        else:
            universe = SP500_FALLBACK

        period_label = st.selectbox("Price history period", list(PERIOD_OPTIONS.keys()), index=1)
        period = PERIOD_OPTIONS[period_label]

    # Technical
    with st.sidebar.expander("Technical filters", expanded=True):
        rsi_min, rsi_max = st.slider("RSI Range (14)", 0, 100, (30, 70))

        colA, colB = st.columns(2)
        with colA:
            price_vs_ma50 = st.selectbox("Price vs 50-day MA", ["Any", "Above", "Below"], index=0)
        with colB:
            price_vs_ma200 = st.selectbox("Price vs 200-day MA", ["Any", "Above", "Below"], index=0)

        req_cross = st.checkbox("Require recent MA crossover (50 vs 200)?", value=False)
        cross_type = "Golden"
        look_cross = 10
        if req_cross:
            col1, col2 = st.columns(2)
            with col1:
                cross_type = st.selectbox("Crossover type", ["Golden", "Death"], index=0)
            with col2:
                look_cross = st.number_input("Lookback days", min_value=1, max_value=60, value=10, step=1)

    # MACD
    with st.sidebar.expander("MACD filter", expanded=False):
        macd_enable = st.checkbox("Enable MACD filter", value=False)
        macd_cond = "Line > Signal"
        macd_look = 10
        if macd_enable:
            macd_cond = st.selectbox(
                "Condition",
                ["Line > Signal", "Line < Signal", "Bull cross (last N days)", "Bear cross (last N days)"],
                index=0
            )
            macd_look = st.number_input("Lookback days for cross", 1, 60, 10, 1)

    # Fundamentals (only rendered if enabled to avoid ValueAboveMax errors)
    with st.sidebar.expander("Fundamentals (optional)", expanded=False):
        fundamentals_enable = st.checkbox("Enable fundamentals filters", value=False, help="Simple market-cap bounds.")
        min_cap = None
        max_cap = None
        if fundamentals_enable:
            min_cap = st.number_input("Market cap min ($)", value=0.0, min_value=0.0, step=1_000_000_000.0, format="%.2f")
            max_cap = st.number_input("Market cap max ($)", value=0.0, min_value=0.0, step=1_000_000_000.0, format="%.2f")
            if max_cap == 0.0:
                max_cap = None

    # Ranking & weights
    with st.sidebar.expander("Scoring & ranking", expanded=True):
        ranking_enable = st.checkbox("Enable ranking / composite score", value=True)
        weight_rsi = 1.0
        weight_trend = 1.0
        weight_macd = 1.0
        weight_cross = 0.5
        if ranking_enable:
            weight_rsi = st.slider("Weight: RSI sweet spot", 0.0, 2.0, 1.0, 0.05)
            weight_trend = st.slider("Weight: Trend vs MAs", 0.0, 2.0, 1.0, 0.05)
            weight_macd = st.slider("Weight: MACD momentum", 0.0, 2.0, 1.0, 0.05)
            weight_cross = st.slider("Weight: Golden/Death cross", 0.0, 2.0, 0.5, 0.05)

        top_n = st.number_input("Show Top N", min_value=5, max_value=100, value=25, step=1)

    return Settings(
        universe=universe,
        period=period,
        rsi_min=float(rsi_min),
        rsi_max=float(rsi_max),
        price_vs_ma50=price_vs_ma50,
        price_vs_ma200=price_vs_ma200,
        require_recent_cross=req_cross,
        cross_type=cross_type,
        cross_lookback=int(look_cross),
        macd_enable=macd_enable,
        macd_condition=macd_cond,
        macd_lookback=int(macd_look),
        fundamentals_enable=fundamentals_enable,
        mcap_min=float(min_cap) if fundamentals_enable else None,
        mcap_max=float(max_cap) if (fundamentals_enable and max_cap is not None) else None,
        ranking_enable=ranking_enable,
        weight_rsi=float(weight_rsi),
        weight_trend=float(weight_trend),
        weight_macd=float(weight_macd),
        weight_cross=float(weight_cross),
        top_n=int(top_n),
    )


def main():
    st.title("StockPeers Screener")

    # --- Left: filters / Right: output ---
    cfg = render_filters()

    # Use a form to avoid immediate reruns on every widget change.
    with st.form("run_form", clear_on_submit=False):
        submitted = st.form_submit_button("Run Scan", use_container_width=True)

    # We always keep the latest settings in session state so chart selection won't wipe results
    st.session_state.setdefault("latest_settings", cfg)
    st.session_state["latest_settings"] = cfg

    if submitted or ("results_df" not in st.session_state):
        with st.spinner("Scanning..."):
            df = run_scan_once(cfg)
        st.session_state["results_df"] = df

        # Also keep indicators & prices for charting (only for tickers that passed)
        if not df.empty:
            tickers = df["Ticker"].tolist()
            close = download_prices(tickers, cfg.period)
            st.session_state["indicators"] = compute_indicators(close)
        else:
            st.session_state["indicators"] = None

    results_df: pd.DataFrame = st.session_state.get("results_df", pd.DataFrame())

    if results_df.empty:
        st.success("Found 0 match(es). Adjust filters and press **Run Scan**.")
        return

    # Show top N
    show_df = results_df.copy()
    if cfg.top_n and len(show_df) > cfg.top_n:
        show_df = show_df.head(cfg.top_n)

    st.success(f"Found {len(results_df)} match(es). Showing top {min(cfg.top_n, len(results_df))}.")

    # Nicely formatted table
    fmt = show_df.copy()
    if "Price" in fmt.columns:
        fmt["Price"] = fmt["Price"].map(lambda x: f"${x:,.2f}")
    for c in ["RSI14", "MA50", "MA200"]:
        if c in fmt.columns:
            fmt[c] = fmt[c].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    if "Score" in fmt.columns:
        fmt["Score"] = fmt["Score"].map(lambda x: f"{x:0.3f}")

    st.dataframe(fmt, use_container_width=True, hide_index=True)

    # ------------- Chart -------------
    st.subheader("Interactive Chart")

    ind = st.session_state.get("indicators")
    if ind is None:
        st.info("No indicators computed (no matches).")
        return

    tickers = show_df["Ticker"].tolist()
    chart_tkr = st.selectbox("Choose ticker", tickers, key="chart_ticker")
    fig = plot_interactive(chart_tkr, ind)
    st.plotly_chart(fig, use_container_width=True)

    # Optional debug (collapsed)
    with st.expander("Debug (data sanity check)", expanded=False):
        t = chart_tkr
        close = ind["close"][t]
        ma50 = ind["ma50"][t]
        ma200 = ind["ma200"][t]
        rsi = ind["rsi14"][t]
        macd_line = ind["macd"][t]
        macd_sig = ind["macd_signal"][t]
        hist = ind["macd_hist"][t]
        dbg = make_price_rsi_macd(close, ma50, ma200, rsi, macd_line, macd_sig, hist, title=f"{t} — Debug")
        st.plotly_chart(dbg, use_container_width=True)


if __name__ == "__main__":
    main()
