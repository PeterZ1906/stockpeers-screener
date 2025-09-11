# scanner.py
# StockPeers Screener — session-safe version
# Selecting a chart ticker will NOT reset the scan results.

import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------
# 0) Streamlit page config
# ---------------------------
st.set_page_config(page_title="StockPeers Screener", layout="wide")
st.title("StockPeers Screener")

# Initialize session defaults
if "results" not in st.session_state:
    st.session_state.results = None  # pandas DataFrame or None
if "last_settings" not in st.session_state:
    st.session_state.last_settings = {}
if "chart_ticker" not in st.session_state:
    st.session_state.chart_ticker = None


# ---------------------------
# 1) Utilities & indicators
# ---------------------------

@st.cache_data(show_spinner=False)
def load_sp500_tickers() -> list[str]:
    # simple static list fallback; replace with Wikipedia scrape if desired
    # keeping a solid subset so demos load quickly
    base = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","UNH","XOM",
        "JNJ","V","WMT","JPM","PG","MA","HD","CVX","LLY","BAC","KO","PEP","PFE",
        "ABBV","COST","AVGO","MRK","ADBE","MCD","TMO","CSCO","CRM","ABT","NFLX",
        "ACN","NKE","LIN","DHR","TXN","AMD","CMCSA","INTC","VZ","QCOM","HON",
        "NEE","LOW","ORCL","BMY","UPS","PM","RTX","MS","INTU","AMGN","SCHW",
        "BLK","PLD","C","GE","CAT","NOW","BKNG","LRCX","MU","KLAC","ASML","FICO",
        "NVR","TDG","KLA","TPL","COIN","PLTR","AMZN","AAPL","MSFT","GOOGL"
    ]
    return sorted(list(set(base)))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def last_cross(s1: pd.Series, s2: pd.Series, lookback: int) -> str | None:
    """Return 'golden', 'death', or None if no cross in last N bars."""
    if len(s1) < 2 or len(s2) < 2:
        return None
    s1, s2 = s1.dropna(), s2.dropna()
    n = min(lookback, len(s1)-1, len(s2)-1)
    if n <= 1:
        return None
    s1_c = s1.iloc[-n:]
    s2_c = s2.iloc[-n:]
    cross_up = (s1_c > s2_c) & (s1_c.shift(1) <= s2_c.shift(1))
    cross_dn = (s1_c < s2_c) & (s1_c.shift(1) >= s2_c.shift(1))
    if cross_up.any():
        return "golden"
    if cross_dn.any():
        return "death"
    return None


@st.cache_data(show_spinner=False)
def fetch_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    # Auto-adjust to get clean OHLC (dividends/splits handled)
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"Adj Close": "AdjClose"})
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = df["AdjClose"]

    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    df["RSI14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    df["MACD"] = macd_line
    df["Signal"] = signal_line
    df["Hist"] = hist
    return df


# ---------------------------
# 2) Screener logic
# ---------------------------

def passes_filters(
    df: pd.DataFrame,
    rsi_min: float, rsi_max: float,
    price_vs_50: str, price_vs_200: str,
    want_cross: bool, cross_type: str, cross_lookback: int,
    macd_on: bool, macd_mode: str, macd_lookback: int
) -> bool:
    if df.empty:
        return False

    # require enough data for indicators
    if df["AdjClose"].count() < 60:
        return False

    last = df.iloc[-1]
    rsi_ok = rsi_min <= last["RSI14"] <= rsi_max
    if not rsi_ok:
        return False

    # price vs MAs
    def cmp_price_vs_ma(which: str) -> bool:
        if which == "Any":
            return True
        if which == "Above":
            return last["AdjClose"] > last["MA" + ("50" if which == "Above" or which == "Below" else "50")]
        if which == "Below":
            return last["AdjClose"] < last["MA" + ("50" if which == "Below" or which == "Above" else "50")]
        return True

    # explicit handling for 50 and 200
    if price_vs_50 != "Any":
        if price_vs_50 == "Above" and not (last["AdjClose"] > last["MA50"]):
            return False
        if price_vs_50 == "Below" and not (last["AdjClose"] < last["MA50"]):
            return False
    if price_vs_200 != "Any":
        if price_vs_200 == "Above" and not (last["AdjClose"] > last["MA200"]):
            return False
        if price_vs_200 == "Below" and not (last["AdjClose"] < last["MA200"]):
            return False

    # MA cross
    if want_cross:
        c = last_cross(df["MA50"], df["MA200"], cross_lookback)
        if c is None:
            return False
        if cross_type == "Golden" and c != "golden":
            return False
        if cross_type == "Death" and c != "death":
            return False

    # MACD modes
    if macd_on:
        if macd_mode == "Line > Signal":
            if not (last["MACD"] > last["Signal"]):
                return False
        elif macd_mode == "Line < Signal":
            if not (last["MACD"] < last["Signal"]):
                return False
        elif macd_mode == "Recent cross (lookback)":
            c = last_cross(df["MACD"], df["Signal"], macd_lookback)
            if c is None:
                return False

    return True


def score_row(
    df: pd.DataFrame,
    w_rsi: float, w_trend: float, w_macd: float, w_cross: float
) -> float:
    """Simple composite score; 0..10-ish scale."""
    last = df.iloc[-1]

    # RSI sweet spot near 50
    rsi_score = 1 - (abs(last["RSI14"] - 50) / 50)  # near 50 => closer to 1
    rsi_score = max(0, min(1, rsi_score))

    # Trend: price above 50 and 200 => 1.0; else partial
    trend_score = 0.0
    trend_score += 0.5 if last["AdjClose"] > last["MA50"] else 0
    trend_score += 0.5 if last["AdjClose"] > last["MA200"] else 0

    # MACD momentum: positive hist => 1.0-ish
    macd_score = 0.5 + 0.5 * np.tanh(last["Hist"] if not math.isnan(last["Hist"]) else 0)

    # Cross bonus
    cross_bonus = 0.0
    c = last_cross(df["MA50"], df["MA200"], 20)
    if c == "golden":
        cross_bonus = 1.0
    elif c == "death":
        cross_bonus = -1.0

    total = (
        w_rsi * rsi_score +
        w_trend * trend_score +
        w_macd * macd_score +
        w_cross * cross_bonus
    )
    return float(total)


def run_scan_once(
    universe: list[str],
    period: str,
    rsi_range: tuple[float,float],
    price_vs_50: str, price_vs_200: str,
    want_cross: bool, cross_type: str, cross_lookback: int,
    macd_on: bool, macd_mode: str, macd_lookback: int,
    do_rank: bool, weights: dict, top_n: int
) -> pd.DataFrame:

    rsi_min, rsi_max = rsi_range
    rows = []
    for tkr in universe:
        df = fetch_prices(tkr, period=period)
        if df.empty:
            continue
        df = compute_indicators(df)
        if not passes_filters(
            df,
            rsi_min, rsi_max,
            price_vs_50, price_vs_200,
            want_cross, cross_type, cross_lookback,
            macd_on, macd_mode, macd_lookback
        ):
            continue
        last = df.iloc[-1]
        row = {
            "Ticker": tkr,
            "Price": float(last["AdjClose"]),
            "RSI(14)": float(last["RSI14"]),
            "MA50": float(last["MA50"]),
            "MA200": float(last["MA200"]),
            "MACD": float(last["MACD"]),
            "Signal": float(last["Signal"]),
            "Hist": float(last["Hist"]),
        }
        if do_rank:
            sc = score_row(
                df,
                weights["w_rsi"],
                weights["w_trend"],
                weights["w_macd"],
                weights["w_cross"]
            )
            row["Score"] = sc
        rows.append(row)

    results = pd.DataFrame(rows).set_index("Ticker").sort_index()
    if results.empty:
        return results

    if do_rank:
        results = results.sort_values("Score", ascending=False).head(top_n)
        results.insert(0, "Rank", range(1, len(results) + 1))
    return results


# ---------------------------
# 3) UI — Sidebar controls
# ---------------------------

with st.sidebar:
    st.markdown("### Filters")

    period = st.selectbox(
        "Price history period",
        ["6mo", "1y", "2y", "5y", "10y", "max"],
        index=1
    )

    rsi_min, rsi_max = st.slider("RSI Range (14)", 0, 100, (30, 70), step=1)

    price_vs_50 = st.selectbox("Price vs 50-day MA", ["Any", "Above", "Below"], index=0)
    price_vs_200 = st.selectbox("Price vs 200-day MA", ["Any", "Above", "Below"], index=0)

    want_cross = st.checkbox("Require recent MA crossover (50 vs 200)?", value=False)
    cross_type = st.selectbox("Crossover Type", ["Any", "Golden", "Death"], disabled=not want_cross)
    cross_lookback = st.number_input("Lookback days for crossover", 1, 60, 10, step=1, disabled=not want_cross)

    st.markdown("### MACD")
    macd_on = st.checkbox("Enable MACD filter", value=False)
    macd_mode = st.selectbox(
        "MACD condition",
        ["Line > Signal", "Line < Signal", "Recent cross (lookback)"],
        disabled=not macd_on
    )
    macd_lookback = st.number_input("Lookback days for MACD cross", 1, 60, 10, step=1, disabled=not macd_on)

    st.markdown("### Scoring & Ranking")
    do_rank = st.checkbox("Enable ranking / composite score", value=True)

    w_rsi = st.slider("Weight: RSI sweet spot", 0.0, 2.0, 1.0, 0.1, disabled=not do_rank)
    w_trend = st.slider("Weight: Trend vs MAs", 0.0, 2.0, 1.0, 0.1, disabled=not do_rank)
    w_macd = st.slider("Weight: MACD momentum", 0.0, 2.0, 1.0, 0.1, disabled=not do_rank)
    w_cross = st.slider("Weight: Golden/Death boost", 0.0, 2.0, 0.5, 0.1, disabled=not do_rank)
    top_n = st.number_input("Show Top N", 5, 100, 25, step=1, disabled=not do_rank)

    st.divider()
    run_clicked = st.button("Run Scan", type="primary", use_container_width=True)
    clear_clicked = st.button("Clear Results", use_container_width=True)


# ---------------------------
# 4) Actions: run / clear
# ---------------------------

# Pack current settings so we can save them after a successful run
current_settings = dict(
    period=period,
    rsi_min=rsi_min, rsi_max=rsi_max,
    price_vs_50=price_vs_50, price_vs_200=price_vs_200,
    want_cross=want_cross, cross_type=cross_type, cross_lookback=cross_lookback,
    macd_on=macd_on, macd_mode=macd_mode, macd_lookback=macd_lookback,
    do_rank=do_rank, top_n=top_n,
    weights={"w_rsi": w_rsi, "w_trend": w_trend, "w_macd": w_macd, "w_cross": w_cross},
)

if clear_clicked:
    st.session_state.results = None
    st.session_state.last_settings = {}
    st.session_state.chart_ticker = None
    st.toast("Results cleared.", icon="🗑️")
    st.stop()

if run_clicked:
    with st.spinner("Running scan..."):
        universe = load_sp500_tickers()
        results_df = run_scan_once(
            universe=universe,
            period=current_settings["period"],
            rsi_range=(current_settings["rsi_min"], current_settings["rsi_max"]),
            price_vs_50=current_settings["price_vs_50"],
            price_vs_200=current_settings["price_vs_200"],
            want_cross=current_settings["want_cross"],
            cross_type=current_settings["cross_type"],
            cross_lookback=current_settings["cross_lookback"],
            macd_on=current_settings["macd_on"],
            macd_mode=current_settings["macd_mode"],
            macd_lookback=current_settings["macd_lookback"],
            do_rank=current_settings["do_rank"],
            weights=current_settings["weights"],
            top_n=current_settings["top_n"],
        )
    st.session_state.results = results_df
    st.session_state.last_settings = current_settings
    # Reset chart ticker to first row (if any)
    st.session_state.chart_ticker = results_df.index[0] if not results_df.empty else None


# ---------------------------
# 5) Results table
# ---------------------------

results = st.session_state.results

if results is None:
    st.info("Click **Run Scan** to generate results.")
    st.stop()

if results.empty:
    st.warning("No matches found. Relax filters or change period and try again.")
    st.stop()

st.success(f"Found {len(results)} match(es).")

# Nice formatting
fmt = {
    "Price": "${:,.2f}",
    "MA50": "${:,.2f}",
    "MA200": "${:,.2f}",
    "RSI(14)": "{:,.2f}",
    "MACD": "{:,.2f}",
    "Signal": "{:,.2f}",
    "Hist": "{:,.2f}",
    "Score": "{:,.3f}"
}

# If Rank exists, put it first
cols = list(results.columns)
if "Rank" in cols:
    cols = ["Rank"] + [c for c in cols if c != "Rank"]
results = results[cols]

st.dataframe(
    results.style.format(fmt),
    use_container_width=True,
    height=480
)


# ---------------------------
# 6) Chart section (session-safe)
# ---------------------------

st.markdown("### Full chart & debug")

left, right = st.columns([1, 3])

with left:
    tickers = results.index.tolist()
    default_idx = 0
    if st.session_state.chart_ticker in tickers:
        default_idx = tickers.index(st.session_state.chart_ticker)

    sel = st.selectbox(
        "Choose ticker",
        tickers,
        index=default_idx
    )
    # Remember last selection so reruns won't reset the table
    st.session_state.chart_ticker = sel

    chart_period = st.selectbox("Chart period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=1)
    mode = st.radio("View mode", ["Price (actual)", "% change (rebased)", "Log price"], index=0)

with right:
    # Fetch fresh series for chosen ticker period (only for the chart)
    df_c = fetch_prices(sel, period=chart_period)
    df_c = compute_indicators(df_c)

    if df_c.empty or df_c["AdjClose"].count() < 10:
        st.warning("Not enough data to chart this ticker & period.")
    else:
        # Build figure
        show_log = (mode == "Log price")
        show_rebased = (mode == "% change (rebased)")

        price = df_c["AdjClose"]
        if show_rebased:
            base = price.iloc[0]
            price_plot = 100 * (price / base - 1.0)
            ytitle = "Return (%)"
        else:
            price_plot = price
            ytitle = "Price"

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.58, 0.12, 0.15, 0.15],
            vertical_spacing=0.02,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]],
        )

        # Price line & MAs
        fig.add_trace(go.Scatter(x=price_plot.index, y=price_plot, name="Adj Close", line=dict(color="#1f77b4")), row=1, col=1)

        if not show_rebased:
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c["MA50"], name="MA50", line=dict(color="#2ca02c")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c["MA200"], name="MA200", line=dict(color="#d62728")), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(x=df_c.index, y=df_c["Volume"], name="Volume", marker_color="#94b2d6"), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c["RSI14"], name="RSI(14)", line=dict(color="#444")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="green", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="red", row=3, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c["MACD"], name="MACD", line=dict(color="#9467bd")), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_c.index, y=df_c["Signal"], name="Signal", line=dict(color="#ff7f0e")), row=4, col=1)
        fig.add_trace(go.Bar(x=df_c.index, y=df_c["Hist"], name="Hist", marker_color=np.where(df_c["Hist"]>=0, "#7fba7a", "#d77979")), row=4, col=1)

        fig.update_yaxes(title_text=ytitle, row=1, col=1, type="log" if show_log and not show_rebased else "linear")
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI(14)", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)

        fig.update_layout(
            height=820,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=60, r=20, t=10, b=20),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Debug (data sanity check)"):
        st.write(df_c.tail(3))
        if not df_c.empty:
            st.write({
                "points": int(df_c["AdjClose"].count()),
                "min": float(df_c["AdjClose"].min()),
                "max": float(df_c["AdjClose"].max()),
            })
        st.line_chart(df_c["AdjClose"] if "AdjClose" in df_c else pd.Series(dtype=float))
