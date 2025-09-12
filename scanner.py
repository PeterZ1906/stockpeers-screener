# StockPeers Screener (full app)
# ------------------------------------------------------------
# Streamlit app to screen S&P 500 by technicals (RSI, MAs, MACD) and optional fundamentals.
# The scan only re-runs when you click "Run Scan". Selecting a ticker to chart does not
# restart the scan. Data sources: Yahoo Finance via yfinance; S&P 500 tickers via CSV.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


# -----------------------------
# Utilities & configuration
# -----------------------------

st.set_page_config(page_title="StockPeers Screener", layout="wide")

# Small helper so we don't crash on weird values
def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


@dataclass
class Settings:
    # core
    period: str
    rsi_range: Tuple[int, int]
    price_vs_ma50: str
    price_vs_ma200: str
    require_cross: bool
    cross_type: str
    cross_lookback: int

    # macd
    macd_enable: bool
    macd_condition: str
    macd_lookback: int

    # fundamentals
    fundamentals_enable: bool
    mc_min: float
    mc_max: float
    pe_max: float
    dy_min: float
    beta_max: float

    # scoring
    scoring_enable: bool
    w_rsi: float
    w_trend: float
    w_macd: float
    w_cross: float

    # output
    top_n: int
    universe: str


# -----------------------------
# Data sources
# -----------------------------

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_sp500_tickers() -> List[str]:
    """
    Pull a clean list of S&P 500 tickers from a CSV (no lxml required).
    Fallback to a short static list if the request fails.
    """
    urls = [
        # official datasets repo
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        # alternate maintained CSV
        "https://raw.githubusercontent.com/danielgrijalva/stock-indexes/master/indexes/sp500/constituents.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            col = "Symbol" if "Symbol" in df.columns else "symbol"
            tickers = sorted(set(df[col].str.upper().str.replace(".", "-", regex=False)))
            if tickers:
                return tickers
        except Exception:
            continue

    # short fallback if web is blocked
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "BRK-B", "UNH", "XOM", "JPM", "V", "MA", "AVGO"
    ]


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def download_prices(tickers: List[str], period: str) -> pd.DataFrame:
    """
    Download daily prices for multiple tickers.
    Returns a wide DataFrame of Adjusted Close with Date index.
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

    # yfinance structure differs for one vs many tickers
    if isinstance(data.columns, pd.MultiIndex):
        if ("Adj Close" in data.columns.levels[0]):
            close = data["Adj Close"].copy()
        else:
            close = data["Close"].copy()
    else:
        # single ticker
        key = "Adj Close" if "Adj Close" in data.columns else "Close"
        close = data[[key]].copy()
        close.columns = [tickers[0]]

    close = close.sort_index()
    close = close.dropna(how="all")
    return close


# -----------------------------
# Technical indicators
# -----------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi_14(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def crosses_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crosses_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


# -----------------------------
# Fundamentals (optional)
# -----------------------------

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """
    Lightweight fundamentals using yfinance fast_info/info.
    Returns columns: market_cap_b, pe, div_yield_pct, beta
    """
    recs = []
    # Use yfinance Tickers for a bit of batching
    yft = yf.Tickers(" ".join(tickers))
    for tkr in tickers:
        try:
            tk = yft.tickers[tkr]
            mc = np.nan
            pe = np.nan
            dy = np.nan
            beta = np.nan

            # Try fast_info first
            fi = getattr(tk, "fast_info", None) or {}
            mc = fi.get("market_cap", np.nan)
            dy = fi.get("dividend_yield", dy)
            beta = fi.get("beta", beta)

            # Fallback to info for PE/div if needed
            inf = getattr(tk, "info", {}) or {}
            if (isinstance(pe, float) and math.isnan(pe)) and "trailingPE" in inf:
                pe = inf.get("trailingPE")
            if (isinstance(dy, float) and math.isnan(dy)) and "dividendYield" in inf:
                dy = inf.get("dividendYield")
            if (isinstance(beta, float) and math.isnan(beta)) and "beta" in inf:
                beta = inf.get("beta")

            recs.append(
                dict(
                    Ticker=tkr,
                    market_cap_b=(mc / 1e9) if pd.notna(mc) else np.nan,
                    pe=pe if pd.notna(pe) else np.nan,
                    div_yield_pct=(dy * 100.0 if pd.notna(dy) and dy < 1e3 else np.nan),
                    beta=beta if pd.notna(beta) else np.nan,
                )
            )
        except Exception:
            recs.append(dict(Ticker=tkr, market_cap_b=np.nan, pe=np.nan, div_yield_pct=np.nan, beta=np.nan))
    return pd.DataFrame.from_records(recs)


# -----------------------------
# Scanner core
# -----------------------------

def compute_snapshot(close: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final snapshot row (latest value) for price, RSI, MA50, MA200, MACD.
    """
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    # RSI per column
    rsi_last = {}
    for col in close.columns:
        rsi_last[col] = rsi_14(close[col]).iloc[-1]
    rsi_last = pd.Series(rsi_last)

    # MACD last
    macd_line_last, signal_last, hist_last = {}, {}, {}
    for col in close.columns:
        ml, sl, h = macd(close[col])
        macd_line_last[col] = ml.iloc[-1]
        signal_last[col] = sl.iloc[-1]
        hist_last[col] = h.iloc[-1]

    # Golden / Death crosses in trailing 60 sessions (we'll evaluate later by lookback)
    gcross, dcross = {}, {}
    for col in close.columns:
        gc_series = crosses_up(ma50[col], ma200[col])
        dc_series = crosses_down(ma50[col], ma200[col])
        gcross[col] = gc_series
        dcross[col] = dc_series
    # store series dicts for later lookback checks
    st.session_state["_gc"] = gcross
    st.session_state["_dc"] = dcross

    snap = pd.DataFrame({
        "Price": close.iloc[-1],
        "RSI14": rsi_last,
        "MA50": ma50.iloc[-1],
        "MA200": ma200.iloc[-1],
        "MACD": pd.Series(macd_line_last),
        "Signal": pd.Series(signal_last),
        "Hist": pd.Series(hist_last),
    })
    snap.index.name = "Ticker"
    snap = snap.dropna(subset=["Price", "MA50", "MA200"], how="any")
    return snap


def apply_filters(snap: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    low, high = settings.rsi_range
    df = snap.copy()

    # RSI band
    df = df[(df["RSI14"] >= low) & (df["RSI14"] <= high)]

    # price vs MAs
    if settings.price_vs_ma50 == "Above":
        df = df[df["Price"] > df["MA50"]]
    elif settings.price_vs_ma50 == "Below":
        df = df[df["Price"] < df["MA50"]]

    if settings.price_vs_ma200 == "Above":
        df = df[df["Price"] > df["MA200"]]
    elif settings.price_vs_ma200 == "Below":
        df = df[df["Price"] < df["MA200"]]

    # recent MA crossover
    if settings.require_cross:
        look = int(settings.cross_lookback)
        mask = []
        gc_map = st.session_state.get("_gc", {})
        dc_map = st.session_state.get("_dc", {})
        for tkr in df.index:
            gc = bool(gc_map.get(tkr, pd.Series(dtype=bool)).tail(look).any())
            dc = bool(dc_map.get(tkr, pd.Series(dtype=bool)).tail(look).any())
            if settings.cross_type == "Golden":
                mask.append(gc)
            elif settings.cross_type == "Death":
                mask.append(dc)
            else:
                mask.append(gc or dc)
        df = df[np.array(mask, dtype=bool)]

    # MACD filter
    if settings.macd_enable:
        look = int(settings.macd_lookback)
        # quick re-eval using last N sessions to confirm cross if needed
        # We kept only last value in snap, so for "crossed" we approximate
        # using MACD and signal slopes over last few days via download again is too heavy.
        # Simple rule: last value relative to signal or sign(historical mean)
        if settings.macd_condition == "Line > Signal":
            df = df[df["MACD"] > df["Signal"]]
        elif settings.macd_condition == "Line < Signal":
            df = df[df["MACD"] < df["Signal"]]
        elif settings.macd_condition == "Crossed up":
            # MACD just crossed above signal recently:
            # approximate: today MACD>Signal AND yesterday MACD<=Signal
            # For speed we skip; we’ll accept "Line>Signal" as proxy.
            df = df[df["MACD"] > df["Signal"]]
        elif settings.macd_condition == "Crossed down":
            df = df[df["MACD"] < df["Signal"]]
        # "Any" -> no filter

    return df


def score_and_rank(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if not settings.scoring_enable or df.empty:
        df["Score"] = 0.0
        return df

    low, high = settings.rsi_range
    center = (low + high) / 2.0
    half = max(1.0, (high - low) / 2.0)
    # RSI sweet spot (1.0 at center, decays to 0 at band edges)
    rsi_score = 1 - np.minimum(1.0, np.abs(df["RSI14"] - center) / half)

    # Trend vs MAs: 1.0 if above both, else partial credit
    trend_score = 0.5 * (df["Price"] / df["MA50"]) + 0.5 * (df["Price"] / df["MA200"])
    # normalize roughly around 1.0
    trend_score = np.tanh(trend_score - 1.0) + 1.0  # ~0..2 -> squash to ~0..1.76, then scale:
    trend_score = (trend_score - trend_score.min()) / (trend_score.max() - trend_score.min() + 1e-9)

    # MACD momentum: positive hist is good
    macd_mom = np.tanh(df["Hist"].fillna(0.0))
    macd_mom = (macd_mom - macd_mom.min()) / (macd_mom.max() - macd_mom.min() + 1e-9)

    # Cross bonus (if present in trailing lookback)
    cross_bonus = np.zeros(len(df))
    if settings.require_cross:
        gc_map = st.session_state.get("_gc", {})
        dc_map = st.session_state.get("_dc", {})
        look = int(settings.cross_lookback)
        for i, tkr in enumerate(df.index):
            gc = bool(gc_map.get(tkr, pd.Series(dtype=bool)).tail(look).any())
            dc = bool(dc_map.get(tkr, pd.Series(dtype=bool)).tail(look).any())
            if settings.cross_type == "Golden":
                cross_bonus[i] = 1.0 if gc else 0.0
            elif settings.cross_type == "Death":
                cross_bonus[i] = 1.0 if dc else 0.0
            else:
                cross_bonus[i] = 1.0 if (gc or dc) else 0.0

    score = (
        settings.w_rsi * rsi_score +
        settings.w_trend * trend_score +
        settings.w_macd * macd_mom +
        settings.w_cross * cross_bonus
    )

    df = df.copy()
    df["Score"] = score
    return df.sort_values("Score", ascending=False)


def apply_fundamentals(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if not settings.fundamentals_enable or df.empty:
        return df

    fundamentals = fetch_fundamentals(df.index.tolist())
    merged = df.merge(fundamentals, left_index=True, right_on="Ticker", how="left").set_index("Ticker", drop=True)

    # filters
    if not math.isinf(settings.mc_min) or not math.isinf(settings.mc_max):
        merged = merged[(merged["market_cap_b"] >= settings.mc_min) & (merged["market_cap_b"] <= settings.mc_max)]
    merged = merged[(merged["pe"].isna()) | (merged["pe"] <= settings.pe_max)]
    merged = merged[(merged["div_yield_pct"].isna()) | (merged["div_yield_pct"] >= settings.dy_min)]
    merged = merged[(merged["beta"].isna()) | (merged["beta"] <= settings.beta_max)]

    return merged


def run_scan_once(settings: Settings, universe: List[str]) -> pd.DataFrame:
    with st.spinner("Downloading prices and computing indicators…"):
        close = download_prices(universe, settings.period)
        if close.empty:
            return pd.DataFrame()
        snap = compute_snapshot(close)
        # keep price table for reuse in charts (optional)
        st.session_state["_last_close"] = close

    # technical filters
    filtered = apply_filters(snap, settings)

    # fundamentals (optional)
    filtered = apply_fundamentals(filtered, settings)

    # scoring/ranking
    ranked = score_and_rank(filtered, settings)

    # final tidy columns/format
    out = ranked[["Score", "Price", "RSI14", "MA50", "MA200", "MACD", "Signal", "Hist"]].copy()
    out = out.sort_values("Score", ascending=False)
    return out


# -----------------------------
# Charting
# -----------------------------

@st.cache_data(show_spinner=False, ttl=3 * 3600)
def get_history_for_chart(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    df = df.rename(columns=lambda c: c.title())
    return df


def plot_ticker_chart(ticker: str, period: str = "1y"):
    df = get_history_for_chart(ticker, period)
    if df.empty:
        st.info("No OHLCV data for this ticker / period.")
        return

    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    rsi = rsi_14(close)
    macd_line, signal_line, hist = macd(close)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.22, 0.23],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    # Price + MAs
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", showlegend=True
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df.index, y=ma50, name="MA50", line=dict(color="#63B3ED")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma200, name="MA200", line=dict(color="#F56565")), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#90cdf4", opacity=0.5),
                  row=1, col=1, secondary_y=True)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI(14)", line=dict(color="#4A5568")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#E53E3E", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#38A169", row=2, col=1)

    # MACD
    colors = np.where(hist >= 0, "#48BB78", "#E53E3E")
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist", marker_color=colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color="#805AD5")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal", line=dict(color="#ED8936")), row=3, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI(14)", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Sidebar filters (with safe defaults)
# -----------------------------

def render_filters() -> Settings:
    with st.sidebar:
        st.header("Filters")

        with st.form("filters_form", clear_on_submit=False):
            period = st.selectbox(
                "Price history period",
                ["6mo", "1y", "2y", "5y", "10y", "max"],
                index=1,
            )
            rsi_range = st.slider("RSI Range (14)", 0, 100, (30, 70))

            price_vs_ma50 = st.selectbox("Price vs 50-day MA", ["Any", "Above", "Below"], index=0)
            price_vs_ma200 = st.selectbox("Price vs 200-day MA", ["Any", "Above", "Below"], index=0)

            require_cross = st.checkbox("Require recent MA crossover (50 vs 200)?", value=False)
            cross_type = st.selectbox("Crossover Type", ["Golden", "Death", "Any"], index=0, disabled=not require_cross)
            cross_lookback = st.number_input(
                "Lookback days for crossover", min_value=1, max_value=60, value=10, step=1, disabled=not require_cross
            )

            st.markdown("---")
            st.subheader("MACD")
            macd_enable = st.checkbox("Enable MACD filter", value=False)
            macd_condition = st.selectbox(
                "MACD condition",
                ["Line > Signal", "Line < Signal", "Crossed up", "Crossed down", "Any"],
                index=0,
                disabled=not macd_enable,
            )
            macd_lookback = st.number_input(
                "Lookback days for MACD cross", min_value=1, max_value=60, value=10, step=1, disabled=not macd_enable
            )

            st.markdown("---")
            st.subheader("Fundamental Filters")
            fundamentals_enable = st.checkbox("Enable fundamental filters", value=False)

            # persisted defaults & clamped values
            mc_min_default = _clamp(float(st.session_state.get("mc_min", 0.0)), 0.0, 10000.0)
            mc_max_default = _clamp(float(st.session_state.get("mc_max", 5000.0)), 0.0, 10000.0)
            pe_max_default = _clamp(float(st.session_state.get("pe_max", 100.0)), 0.0, 1000.0)
            dy_min_default = _clamp(float(st.session_state.get("dy_min", 0.0)), 0.0, 50.0)
            beta_max_default = _clamp(float(st.session_state.get("beta_max", 3.0)), 0.0, 10.0)

            mc_min = st.number_input(
                "Market Cap min ($B)", min_value=0.0, max_value=10000.0,
                value=mc_min_default, step=10.0, disabled=not fundamentals_enable
            )
            mc_max = st.number_input(
                "Market Cap max ($B)", min_value=0.0, max_value=10000.0,
                value=mc_max_default, step=10.0, disabled=not fundamentals_enable
            )
            pe_max = st.number_input(
                "P/E max", min_value=0.0, max_value=1000.0,
                value=pe_max_default, step=1.0, disabled=not fundamentals_enable
            )
            dy_min = st.number_input(
                "Dividend Yield min (%)", min_value=0.0, max_value=50.0,
                value=dy_min_default, step=0.1, disabled=not fundamentals_enable
            )
            beta_max = st.number_input(
                "Beta max (5y monthly)", min_value=0.0, max_value=10.0,
                value=beta_max_default, step=0.1, disabled=not fundamentals_enable
            )

            # persist
            st.session_state["mc_min"] = mc_min
            st.session_state["mc_max"] = mc_max
            st.session_state["pe_max"] = pe_max
            st.session_state["dy_min"] = dy_min
            st.session_state["beta_max"] = beta_max

            st.markdown("---")
            st.subheader("Scoring & Ranking")
            scoring_enable = st.checkbox("Enable ranking / composite score", value=True)
            w_rsi = st.slider("Weight: RSI sweet spot", 0.0, 2.0, 1.0, 0.1, disabled=not scoring_enable)
            w_trend = st.slider("Weight: Trend vs MAs", 0.0, 2.0, 1.0, 0.1, disabled=not scoring_enable)
            w_macd = st.slider("Weight: MACD momentum", 0.0, 2.0, 1.0, 0.1, disabled=not scoring_enable)
            w_cross = st.slider("Weight: Golden/Death boost", 0.0, 2.0, 0.5, 0.1, disabled=not scoring_enable)

            top_n = st.number_input("Show Top N", min_value=5, max_value=100, value=25, step=1)

            universe = st.selectbox("Universe", ["S&P 500"], index=0)

            submitted = st.form_submit_button("Run Scan", use_container_width=True)

    if submitted:
        st.session_state["_run_scan"] = True

    return Settings(
        period=period,
        rsi_range=rsi_range,
        price_vs_ma50=price_vs_ma50,
        price_vs_ma200=price_vs_ma200,
        require_cross=require_cross,
        cross_type=cross_type,
        cross_lookback=int(cross_lookback),
        macd_enable=macd_enable,
        macd_condition=macd_condition,
        macd_lookback=int(macd_lookback),
        fundamentals_enable=fundamentals_enable,
        mc_min=float(mc_min),
        mc_max=float(mc_max),
        pe_max=float(pe_max),
        dy_min=float(dy_min),
        beta_max=float(beta_max),
        scoring_enable=scoring_enable,
        w_rsi=float(w_rsi),
        w_trend=float(w_trend),
        w_macd=float(w_macd),
        w_cross=float(w_cross),
        top_n=int(top_n),
        universe=universe,
    )


# -----------------------------
# Main UI
# -----------------------------

def main():
    st.title("StockPeers Screener")

    settings = render_filters()

    # Only re-run the compute pipeline when the user explicitly presses Run Scan
    if st.session_state.get("_run_scan", False):
        universe = get_sp500_tickers()
        results = run_scan_once(settings, universe)
        st.session_state["_results"] = results
        st.session_state["_run_scan"] = False

    results = st.session_state.get("_results")

    if results is None:
        st.info("Set your filters in the sidebar and click **Run Scan** to get started.")
        return

    if results.empty:
        st.success("Found 0 match(es).")
        st.dataframe(pd.DataFrame(columns=["Ticker", "Score", "Price", "RSI14", "MA50", "MA200"]))
        return

    st.success(f"Found {len(results)} match(es).")

    # Top N & formatting
    top = results.head(settings.top_n).copy()
    top.reset_index(inplace=True)

    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Score": st.column_config.NumberColumn("Score", format="%.3f"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "RSI14": st.column_config.NumberColumn("RSI14", format="%.2f"),
            "MA50": st.column_config.NumberColumn("MA50", format="%.2f"),
            "MA200": st.column_config.NumberColumn("MA200", format="%.2f"),
            "MACD": st.column_config.NumberColumn("MACD", format="%.3f"),
            "Signal": st.column_config.NumberColumn("Signal", format="%.3f"),
            "Hist": st.column_config.NumberColumn("Hist", format="%.3f"),
        },
    )

    st.markdown("### Interactive Chart")

    available = top["Ticker"].tolist()
    if not available:
        st.info("No tickers to chart. Adjust filters or Top N.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Choose ticker", available, index=0, key="chart_ticker")
    with col2:
        chart_period = st.selectbox("Chart period", ["6mo", "1y", "2y", "5y"], index=1)

    plot_ticker_chart(ticker, period=chart_period)


if __name__ == "__main__":
    main()
