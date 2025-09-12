# scanner.py
# StockPeers Screener (S&P 500) — Streamlit app
# Features:
# - RSI, MA50/MA200, MACD indicators
# - Golden/Death cross filter
# - MACD filters
# - Composite ranking with tunable weights
# - Cached results; selecting a chart ticker won't reset your scan
# - Plotly chart with Price, Volume, RSI, MACD
#
# Run: streamlit run scanner.py

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ----------------------------- Streamlit page setup -----------------------------

st.set_page_config(
    page_title="StockPeers Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("StockPeers Screener")

# --------------------------------- Utilities ----------------------------------


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_sp500_tickers() -> list[str]:
    """
    Scrape the current S&P 500 tickers from Wikipedia (cached for 1 hour).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    # remove weird placeholders
    tickers = [t.strip().upper() for t in tickers if t and t != "BF.B"]
    return tickers


@st.cache_data(show_spinner=False)
def fetch_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Download daily data for one ticker and return a flat-column DataFrame
    with an 'AdjClose' column (auto-adjusted) for indicator/plotting.
    Robust against MultiIndex columns that yfinance sometimes returns.
    """
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=True,  # dividends/splits handled
            progress=False,
            threads=False,
            group_by="column",  # <- avoid wide MultiIndex when possible
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # If we *still* get a MultiIndex (rare), flatten by taking the field name
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Standardize the adjusted column name
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    elif "Close" in df.columns and "AdjClose" not in df.columns:
        df = df.rename(columns={"Close": "AdjClose"})  # fallback

    # Clean
    df = df.dropna(how="all")
    if "AdjClose" in df.columns:
        df["AdjClose"] = df["AdjClose"].astype(float)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MA50/MA200, RSI14, MACD/Signal/Hist.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
    if price_col not in df.columns:
        return pd.DataFrame()

    close = df[price_col].astype(float)

    # MAs
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # RSI
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(
        span=26, adjust=False
    ).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd_line
    df["Signal"] = signal_line
    df["Hist"] = macd_line - signal_line

    # Cross flags
    df["GoldenCross"] = (df["MA50"] > df["MA200"]).astype(int)
    df["DeathCross"] = (df["MA50"] < df["MA200"]).astype(int)
    return df


def recent_cross(df: pd.DataFrame, lookback: int, kind: str = "golden") -> bool:
    """
    Did we have a golden/death cross within the last N trading days?
    """
    if df.empty or "MA50" not in df or "MA200" not in df:
        return False
    s = (df["MA50"] > df["MA200"]).astype(int)
    last = s.iloc[-(lookback + 1) :].values
    if len(last) < 2:
        return False
    crossed = (np.diff(last) != 0).nonzero()[0]  # indices where it changed
    if crossed.size == 0:
        return False
    # last change within the lookback window
    idx = crossed[-1]
    if kind == "golden":
        return last[idx] == 0 and last[idx + 1] == 1
    else:
        return last[idx] == 1 and last[idx + 1] == 0


def macd_condition(df: pd.DataFrame, lookback: int, mode: str) -> bool:
    """
    MACD filter over a recent lookback window.
    mode: 'line>signal', 'line<signal', 'hist>0', 'hist<0'
    """
    if df.empty or "MACD" not in df or "Signal" not in df or "Hist" not in df:
        return False
    d = df.iloc[-lookback:]
    if d.empty:
        return False
    if mode == "line>signal":
        return (d["MACD"] > d["Signal"]).all()
    if mode == "line<signal":
        return (d["MACD"] < d["Signal"]).all()
    if mode == "hist>0":
        return (d["Hist"] > 0).all()
    if mode == "hist<0":
        return (d["Hist"] < 0).all()
    return False


def rsi_sweet_spot_score(rsi: float, center: float = 50.0, width: float = 25.0) -> float:
    """
    Score higher for RSI near the center. Gaussian-like bell curve.
    Returns [0..1].
    """
    if math.isnan(rsi):
        return 0.0
    # 50 +/- width gives about half score
    z = (rsi - center) / width
    return float(np.exp(-z * z))


def trend_score(price: float, ma50: float, ma200: float) -> float:
    """
    Trend score based on price vs MA50/MA200 (normalized).
    """
    score = 0.0
    if ma50 > 0:
        score += (price - ma50) / ma50
    if ma200 > 0:
        score += (price - ma200) / ma200
    # clamp to [0..1] after shifting
    score = (score + 1.0) / 2.0
    return float(max(0.0, min(1.0, score)))


def macd_momentum_score(hist: float) -> float:
    """
    More positive histogram → higher score. Normalized via tanh.
    """
    if math.isnan(hist):
        return 0.0
    return float((np.tanh(hist) + 1) / 2)  # [0..1]


def composite_score(
    row: pd.Series,
    w_rsi: float,
    w_trend: float,
    w_macd: float,
    w_cross: float,
    crossed_recently: bool,
) -> float:
    rsi_part = rsi_sweet_spot_score(row.get("RSI14", np.nan))
    trend_part = trend_score(row.get("AdjClose", np.nan), row.get("MA50", np.nan), row.get("MA200", np.nan))
    macd_part = macd_momentum_score(row.get("Hist", np.nan))
    cross_part = 1.0 if crossed_recently else 0.0
    score = (
        w_rsi * rsi_part + w_trend * trend_part + w_macd * macd_part + w_cross * cross_part
    )
    return float(score)


# ------------------------------- Sidebar UI -----------------------------------

st.sidebar.header("Filters")

period = st.sidebar.selectbox("Price history period", ["6mo", "1y", "2y", "5y", "10y"], index=1)

rsi_min, rsi_max = st.sidebar.slider("RSI Range (14)", 0, 100, (30, 70), step=1)

st.sidebar.write("**MA crossover (50 vs 200)**")
require_ma_cross = st.sidebar.checkbox("Require recent MA crossover?")
crossover_type = st.sidebar.selectbox("Crossover Type", ["Golden", "Death"], index=0)
cross_lookback = st.sidebar.number_input("Lookback days for crossover", 5, 90, value=20, step=1)

st.sidebar.write("**MACD**")
enable_macd_filter = st.sidebar.checkbox("Enable MACD filter")
macd_mode = st.sidebar.selectbox(
    "MACD condition",
    ["Line > Signal", "Line < Signal", "Hist > 0", "Hist < 0"],
    index=0,
)
macd_lookback = st.sidebar.number_input("Lookback days for MACD", 5, 60, value=10, step=1)

st.sidebar.write("**Scoring & Ranking**")
enable_ranking = st.sidebar.checkbox("Enable ranking / composite score", value=True)

w_rsi = st.sidebar.slider("Weight: RSI sweet spot", 0.0, 2.0, 1.0, 0.05)
w_trend = st.sidebar.slider("Weight: Trend vs MAs", 0.0, 2.0, 1.0, 0.05)
w_macd = st.sidebar.slider("Weight: MACD momentum", 0.0, 2.0, 1.0, 0.05)
w_cross = st.sidebar.slider("Weight: Golden/Death boost", 0.0, 2.0, 0.5, 0.05)

top_n = st.sidebar.number_input("Show Top N", 5, 100, value=25, step=1)

st.sidebar.write("---")
max_tickers = st.sidebar.slider("Max tickers to scan (speed control)", 50, 500, 200, 50)

run_scan = st.sidebar.button("Run Scan", type="primary")

# Preserve settings/results across UI changes
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "last_params" not in st.session_state:
    st.session_state["last_params"] = None

# ---------------------------------- Scanner -----------------------------------


def should_keep(df: pd.DataFrame) -> bool:
    """Basic quality gate: enough data, valid prices."""
    if df is None or df.empty:
        return False
    return df["AdjClose"].notna().sum() > 30


def run_scan_once() -> pd.DataFrame:
    sp500 = get_sp500_tickers()
    if not sp500:
        st.warning("Could not load S&P 500 universe.")
        return pd.DataFrame()

    # limit scan size for speed
    universe = sp500[: max_tickers]

    rows = []
    prog = st.progress(0.0, text="Scanning tickers...")
    for i, ticker in enumerate(universe, 1):
        df = fetch_prices(ticker, period=period)
        if df.empty:
            prog.progress(i / len(universe), text=f"{ticker}: no data")
            continue
        df = compute_indicators(df)
        if not should_keep(df):
            prog.progress(i / len(universe), text=f"{ticker}: insufficient data")
            continue

        # Filters
        latest = df.iloc[-1]
        rsi_ok = rsi_min <= latest["RSI14"] <= rsi_max

        cross_ok = True
        crossed_recently = False
        if require_ma_cross:
            kind = "golden" if crossover_type.lower().startswith("g") else "death"
            crossed_recently = recent_cross(df, cross_lookback, kind=kind)
            cross_ok = crossed_recently

        macd_ok = True
        if enable_macd_filter:
            mode_map = {
                "Line > Signal": "line>signal",
                "Line < Signal": "line<signal",
                "Hist > 0": "hist>0",
                "Hist < 0": "hist<0",
            }
            macd_ok = macd_condition(df, macd_lookback, mode_map[macd_mode])

        if rsi_ok and cross_ok and macd_ok:
            # Compose score
            score = composite_score(
                latest,
                w_rsi=w_rsi,
                w_trend=w_trend,
                w_macd=w_macd,
                w_cross=w_cross,
                crossed_recently=crossed_recently,
            )
            rows.append(
                {
                    "Ticker": ticker,
                    "Score": score,
                    "Price": float(latest["AdjClose"]),
                    "RSI14": float(latest["RSI14"]),
                    "MA50": float(latest["MA50"]),
                    "MA200": float(latest["MA200"]),
                    "MACD": float(latest["MACD"]),
                    "Signal": float(latest["Signal"]),
                    "Hist": float(latest["Hist"]),
                }
            )
        prog.progress(i / len(universe), text=f"Processed {ticker}")

    prog.empty()

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("Ticker")
    if enable_ranking:
        out = out.sort_values("Score", ascending=False)
        out["Rank"] = range(1, len(out) + 1)
        out = out[["Rank", "Score", "Price", "RSI14", "MA50", "MA200", "MACD", "Signal", "Hist"]]
    else:
        out = out.sort_values("Ticker")

    # keep only Top N (but store full result in session_state)
    return out


# Trigger scan only when the button is pressed; otherwise preserve old results.
params_now = dict(
    period=period,
    rsi_min=rsi_min,
    rsi_max=rsi_max,
    require_ma_cross=require_ma_cross,
    crossover_type=crossover_type,
    cross_lookback=cross_lookback,
    enable_macd_filter=enable_macd_filter,
    macd_mode=macd_mode,
    macd_lookback=macd_lookback,
    enable_ranking=enable_ranking,
    w_rsi=w_rsi,
    w_trend=w_trend,
    w_macd=w_macd,
    w_cross=w_cross,
    top_n=top_n,
    max_tickers=max_tickers,
)

if run_scan:
    results_df_full = run_scan_once()
    st.session_state["results_df"] = results_df_full
    st.session_state["last_params"] = params_now

results_df_full = st.session_state.get("results_df")

# ------------------------------ Results & Charts -------------------------------

if results_df_full is None or results_df_full.empty:
    st.info("Set your filters and click **Run Scan** to see results.")
    st.stop()

st.success(f"Found {len(results_df_full)} match(es).")

# Show top N in the table
results_show = results_df_full.copy()
if enable_ranking and "Rank" in results_show.columns:
    results_show = results_show.nsmallest(top_n, "Rank")
else:
    results_show = results_show.head(top_n)

# Nicely formatted table
# Streamlit column config for formatting
col_cfg = {
    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
    "MA50": st.column_config.NumberColumn("MA50", format="$%.2f"),
    "MA200": st.column_config.NumberColumn("MA200", format="$%.2f"),
    "RSI14": st.column_config.NumberColumn("RSI14", format="%.2f"),
    "MACD": st.column_config.NumberColumn("MACD", format="%.2f"),
    "Signal": st.column_config.NumberColumn("Signal", format="%.2f"),
    "Hist": st.column_config.NumberColumn("Hist", format="%.2f"),
    "Score": st.column_config.NumberColumn("Score", format="%.4f"),
    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
}

st.dataframe(
    results_show,
    use_container_width=True,
    hide_index=False,
    column_config=col_cfg,
)

# ------------------------------ Interactive Chart ------------------------------

with st.expander("Open full interactive chart", expanded=True):

    chart_ticker = st.selectbox(
        "Choose ticker", results_show.index.tolist(), key="chart_ticker"
    )
    chart_period = st.selectbox(
        "Chart period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=1
    )

    # pull fresh data for the chart period (independent of scan)
    ch = fetch_prices(chart_ticker, period=chart_period)
    ch = compute_indicators(ch)

    if ch.empty:
        st.warning("Not enough OHLC data to chart this ticker right now.")
        st.stop()

    # 3 rows: Price + Volume, RSI, MACD
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.20, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Price
    fig.add_trace(go.Scatter(x=ch.index, y=ch["AdjClose"], name="Adj Close", line=dict(color="#1f77b4")), row=1, col=1)
    if "MA50" in ch:
        fig.add_trace(go.Scatter(x=ch.index, y=ch["MA50"], name="MA50", line=dict(color="#82c6ff")), row=1, col=1)
    if "MA200" in ch:
        fig.add_trace(go.Scatter(x=ch.index, y=ch["MA200"], name="MA200", line=dict(color="#ff6e6e")), row=1, col=1)

    # Volume (area)
    if "Volume" in ch.columns:
        fig.add_trace(
            go.Bar(x=ch.index, y=ch["Volume"], name="Volume", marker_color="lightgray"),
            row=1,
            col=1,
        )

    # RSI panel
    fig.add_trace(go.Scatter(x=ch.index, y=ch["RSI14"], name="RSI(14)", line=dict(color="gray")), row=2, col=1)
    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="lightgray", opacity=0.15, row=2, col=1)
    fig.add_hline(y=30, line=dict(color="red", width=1), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="green", width=1), row=2, col=1)

    # MACD panel
    fig.add_trace(go.Scatter(x=ch.index, y=ch["MACD"], name="MACD", line=dict(color="purple")), row=3, col=1)
    fig.add_trace(go.Scatter(x=ch.index, y=ch["Signal"], name="Signal", line=dict(color="orange")), row=3, col=1)
    fig.add_trace(
        go.Bar(x=ch.index, y=ch["Hist"], name="Hist", marker_color=np.where(ch["Hist"] >= 0, "green", "red")),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=720,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30, b=30, l=10, r=10),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI(14)", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------- Debug tools ---------------------------------

with st.expander("Debug (data sanity check)"):
    t = results_show.index.tolist()[0]
    dtest = compute_indicators(fetch_prices(t, period=period))
    st.line_chart(dtest[["AdjClose", "MA50", "MA200"]].dropna().tail(200), height=160)
    st.line_chart(dtest["RSI14"].dropna().tail(200), height=160)
    st.line_chart(dtest[["MACD", "Signal", "Hist"]].dropna().tail(200), height=160)
