# scanner.py
# Streamlit StockPeers Screener (S&P500) with robust chart + debug panel
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(
    page_title="StockPeers Screener",
    page_icon="📈",
    layout="wide",
)

# ---------------------------
# Utilities & Caching
# ---------------------------

@st.cache_data(ttl=60 * 60)
def get_sp500_tickers() -> list:
    """Fetch S&P 500 tickers. Fallback to a short static list if Wikipedia is blocked."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp = tables[0]
        tickers = sp["Symbol"].tolist()
        # Fix Berkshire tickers formatting if needed
        tickers = [t.replace(".", "-") for t in tickers]
        return tickers
    except Exception:
        # A minimal fallback set (so the app still runs without internet to Wikipedia)
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA",
            "META", "BRK-B", "XOM", "JNJ", "JPM", "TSLA", "V", "PG",
        ]


@st.cache_data(ttl=15 * 60)
def yf_download_daily(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download daily adjusted data from yfinance."""
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    # Ensure we always have a DataFrame with the expected columns
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    # yfinance sometimes returns "Adj Close" but we auto_adjust=True, so Close == Adj Close.
    # Keep a consistent column presence:
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df


def rsi_wilder(price: pd.Series, length: int = 14) -> pd.Series:
    """RSI (Wilder). Handles NaN gracefully."""
    price = pd.Series(price).dropna()
    if price.size < length + 1:
        return pd.Series(index=price.index, dtype=float)

    delta = price.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder's smoothing
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "RSI14"
    return rsi


def simple_ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def macd_lines(s: pd.Series, fast=12, slow=26, signal=9):
    """Return MACD line, signal, and histogram."""
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd.rename("MACD"), sig.rename("Signal"), hist.rename("Hist")


def recent_cross(slow: pd.Series, fast: pd.Series, lookback: int, want: str = "Any") -> bool:
    """
    Detect a recent crossover within `lookback` bars:
      - want="Golden": fast crosses up above slow (50 > 200)
      - want="Death" : fast crosses down below slow
      - want="Any"   : either
    Assumes series share the same index.
    """
    s1 = slow.copy().astype(float)
    s2 = fast.copy().astype(float)
    s1, s2 = s1.align(s2, join="inner")
    if s1.empty or s2.empty or len(s1) < 3:
        return False

    cross_up = (s2 > s1) & (s2.shift(1) <= s1.shift(1))
    cross_dn = (s2 < s1) & (s2.shift(1) >= s1.shift(1))

    if lookback <= 0:
        lookback = 1
    recent_slice = slice(-lookback, None)
    has_up = bool(cross_up.iloc[recent_slice].any())
    has_dn = bool(cross_dn.iloc[recent_slice].any())

    if want == "Golden":
        return has_up
    elif want == "Death":
        return has_dn
    else:
        return has_up or has_dn


# ---------------------------
# Sidebar
# ---------------------------

st.sidebar.title("Filters")

universe = st.sidebar.selectbox("Stock universe", ["S&P 500"], index=0)
tickers = get_sp500_tickers()

price_period = st.sidebar.selectbox("Price history period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=1)

rsi_min, rsi_max = st.sidebar.slider("RSI Range (14)", 0, 100, (30, 70))
ma50_sel = st.sidebar.selectbox("Price vs 50-day MA", ["Any", "Above", "Below"], index=0)
ma200_sel = st.sidebar.selectbox("Price vs 200-day MA", ["Any", "Above", "Below"], index=0)

need_cross = st.sidebar.checkbox("Require recent MA crossover (50 vs 200)?", value=False)
cross_type = st.sidebar.selectbox("Crossover type", ["Any", "Golden", "Death"], disabled=not need_cross)
cross_lookback = st.sidebar.number_input("Lookback days for crossover", min_value=1, max_value=60, value=10, step=1, disabled=not need_cross)

st.sidebar.markdown("---")

st.sidebar.subheader("MACD")
use_macd = st.sidebar.checkbox("Enable MACD filter", value=False)
macd_condition = st.sidebar.selectbox("MACD condition", ["Line > Signal", "Line < Signal", "Recent cross (lookback)"], index=0, disabled=not use_macd)
macd_lb = st.sidebar.number_input("Lookback days for MACD cross", 1, 60, 10, disabled=(not use_macd or macd_condition != "Recent cross (lookback)"))

st.sidebar.markdown("---")

st.sidebar.subheader("Scoring & Ranking")
use_score = st.sidebar.checkbox("Enable ranking / composite score", value=True)

w_rsi = st.sidebar.slider("Weight: RSI sweet spot", 0.0, 2.0, 1.0, 0.05)
w_trend = st.sidebar.slider("Weight: Trend vs MAs", 0.0, 2.0, 1.0, 0.05)
w_macd = st.sidebar.slider("Weight: MACD momentum", 0.0, 2.0, 1.0, 0.05)
w_cross = st.sidebar.slider("Weight: Golden/Death boost", 0.0, 2.0, 0.50, 0.05)

top_n = st.sidebar.number_input("Show Top N", min_value=5, max_value=100, value=25, step=1)

run = st.sidebar.button("Run Scan", use_container_width=True)


# ---------------------------
# Screener Engine
# ---------------------------

def passes_filters(row):
    """Apply sidebar filters to a row of computed techs."""
    rsi_val = row["RSI14"]
    if not (rsi_min <= rsi_val <= rsi_max):
        return False

    price = row["Price"]
    ma50 = row["MA50"]
    ma200 = row["MA200"]

    if ma50_sel == "Above" and not (price > ma50):
        return False
    if ma50_sel == "Below" and not (price < ma50):
        return False

    if ma200_sel == "Above" and not (price > ma200):
        return False
    if ma200_sel == "Below" and not (price < ma200):
        return False

    if need_cross:
        if not recent_cross(pd.Series(row["MA200_hist"]), pd.Series(row["MA50_hist"]), cross_lookback, cross_type):
            return False

    if use_macd:
        if macd_condition == "Line > Signal" and not (row["MACD"] > row["Signal"]):
            return False
        if macd_condition == "Line < Signal" and not (row["MACD"] < row["Signal"]):
            return False
        if macd_condition == "Recent cross (lookback)":
            macd_series = pd.Series(row["MACD_hist"], dtype=float)
            sig_series = pd.Series(row["Signal_hist"], dtype=float)
            macd_series, sig_series = macd_series.align(sig_series, join="inner")
            if macd_series.size < 3:
                return False
            up = (macd_series > sig_series) & (macd_series.shift(1) <= sig_series.shift(1))
            dn = (macd_series < sig_series) & (macd_series.shift(1) >= sig_series.shift(1))
            win = slice(-macd_lb, None)
            if not (bool(up.iloc[win].any()) or bool(dn.iloc[win].any())):
                return False

    return True


def compute_score(row):
    """Simple composite score from RSI sweet spot, trend, MACD, crossover boost."""
    # RSI sweet spot (closer to 50 is better)
    rsi_term = -abs(row["RSI14"] - 50.0)

    # Trend vs. MAs (price above MAs is good)
    trend_term = 0.0
    if row["Price"] > row["MA50"]:
        trend_term += 1.0
    if row["Price"] > row["MA200"]:
        trend_term += 1.0

    # MACD momentum
    macd_term = float(row["Hist"]) if np.isfinite(row["Hist"]) else 0.0

    # Crossover boost (based on most recent)
    cross_boost = 0.0
    try:
        ma50_hist = pd.Series(row["MA50_hist"], dtype=float)
        ma200_hist = pd.Series(row["MA200_hist"], dtype=float)
        ma50_hist, ma200_hist = ma50_hist.align(ma200_hist, join="inner")
        up = (ma50_hist > ma200_hist) & (ma50_hist.shift(1) <= ma200_hist.shift(1))
        dn = (ma50_hist < ma200_hist) & (ma50_hist.shift(1) >= ma200_hist.shift(1))
        if bool(up.tail(5).any()):
            cross_boost = 1.0
        elif bool(dn.tail(5).any()):
            cross_boost = -1.0
    except Exception:
        cross_boost = 0.0

    score = (w_rsi * rsi_term) + (w_trend * trend_term) + (w_macd * macd_term) + (w_cross * cross_boost)
    return float(score)


def scan():
    rows = []
    for tkr in tickers:
        try:
            df = yf_download_daily(tkr, period=price_period)
            if df.empty or "Adj Close" not in df.columns:
                continue

            s = df["Adj Close"].astype(float).dropna()
            if s.size < 60:  # avoid tiny series
                continue

            rsi14 = rsi_wilder(s, 14).iloc[-1] if not rsi_wilder(s, 14).empty else np.nan
            ma50 = simple_ma(s, 50).iloc[-1]
            ma200 = simple_ma(s, 200).iloc[-1]
            macd, sig, hist = macd_lines(s)

            row = dict(
                Ticker=tkr,
                Price=float(s.iloc[-1]),
                RSI14=float(rsi14) if np.isfinite(rsi14) else np.nan,
                MA50=float(ma50) if np.isfinite(ma50) else np.nan,
                MA200=float(ma200) if np.isfinite(ma200) else np.nan,
                MACD=float(macd.iloc[-1]) if macd.size else np.nan,
                Signal=float(sig.iloc[-1]) if sig.size else np.nan,
                Hist=float(hist.iloc[-1]) if hist.size else np.nan,
                MA50_hist=simple_ma(s, 50).tolist(),
                MA200_hist=simple_ma(s, 200).tolist(),
                MACD_hist=macd.tolist(),
                Signal_hist=sig.tolist(),
            )

            # Apply filters
            if passes_filters(row):
                if use_score:
                    row["Score"] = compute_score(row)
                else:
                    row["Score"] = 0.0
                rows.append(row)

        except Exception:
            # Skip bad tickers quietly
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Ticker")
    df = df.sort_values("Score", ascending=False)
    return df


# ---------------------------
# Main Area
# ---------------------------

st.title("StockPeers Screener — S&P 500")

if run:
    results = scan()
    if results.empty:
        st.warning("No matches found for your filters. Relax the constraints and try again.")
        st.stop()

    st.success(f"Found {len(results)} match(es).")

    # Show top N
    show = results.head(top_n).copy()

    # Display table with formatting
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=False,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "MA50": st.column_config.NumberColumn("MA50", format="$%.2f"),
            "MA200": st.column_config.NumberColumn("MA200", format="$%.2f"),
            "RSI14": st.column_config.NumberColumn("RSI(14)", format="%.2f"),
            "Score": st.column_config.NumberColumn("Score", format="%.3f"),
            "MACD": st.column_config.NumberColumn("MACD", format="%.4f"),
            "Signal": st.column_config.NumberColumn("Signal", format="%.4f"),
            "Hist": st.column_config.NumberColumn("Hist", format="%.4f"),
            # hide list columns
            "MA50_hist": None,
            "MA200_hist": None,
            "MACD_hist": None,
            "Signal_hist": None,
        },
    )

    # -------------- CHART --------------
    st.markdown("### Open full interactive chart")
    with st.expander("Full chart & debug", expanded=True):
        sel = st.selectbox("Choose ticker", show.index.tolist(), key="chart_ticker")

        # Chart period + view
        cp = st.selectbox("Chart period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=1)
        view_mode = st.radio("View mode", ["Price (actual)", "% change (rebased)", "Log price"], horizontal=True)

        df_c = yf_download_daily(sel, period=cp)
        s = df_c["Adj Close"].astype(float).dropna() if not df_c.empty and "Adj Close" in df_c.columns else pd.Series(dtype=float)
        vol = df_c["Volume"] if not df_c.empty and "Volume" in df_c.columns else pd.Series(dtype=float)

        # Technicals for chart period
        rsi_series = rsi_wilder(s, 14)
        ma50_c = simple_ma(s, 50)
        ma200_c = simple_ma(s, 200)
        macd_c, sig_c, hist_c = macd_lines(s)

        # ---- Debug panel ----
        with st.expander("Debug (data sanity check)"):
            st.write("Ticker:", sel, "| Period:", cp)
            st.write("Points:", int(s.size))
            if s.size > 0:
                st.write("Head:", s.head())
                st.write("Tail:", s.tail())
                try:
                    st.write("Min / Max:", float(np.nanmin(s)), float(np.nanmax(s)))
                except Exception:
                    pass
                st.write("Any NaN?", bool(s.isna().any()))
                st.write("Quick fallback line chart (Streamlit):")
                st.line_chart(s)
            else:
                st.info("Series is empty (no Adj Close data).")

        # ----- Build Plotly figure -----
        rows = 4
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.45, 0.15, 0.20, 0.20],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]],
        )

        # ----- PRICE (robust) -----
        if s.size > 0:
            if view_mode == "% change (rebased)":
                base = float(s.iloc[0])
                y = (s / base - 1.0) * 100.0
                fig.add_trace(go.Scatter(x=y.index, y=y, name="% change", mode="lines",
                                         line=dict(color="#1f77b4", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ma50_c.index, y=(ma50_c / base - 1.0) * 100.0, name="MA50",
                                         mode="lines", line=dict(color="#2ca02c")), row=1, col=1)
                fig.add_trace(go.Scatter(x=ma200_c.index, y=(ma200_c / base - 1.0) * 100.0, name="MA200",
                                         mode="lines", line=dict(color="#d62728")), row=1, col=1)

                ymin = float(np.nanmin(y))
                ymax = float(np.nanmax(y))
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                    pad = 0.05 * (ymax - ymin + 1e-9)
                    fig.update_yaxes(range=[ymin - pad, ymax + pad], row=1, col=1)

                fig.update_yaxes(title_text="Return (%)", tickformat=".1f", ticksuffix="%", row=1, col=1)

            else:
                # main price lines
                fig.add_trace(go.Scatter(x=s.index, y=s, name="Adj Close", mode="lines",
                                         line=dict(color="#1f77b4", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ma50_c.index, y=ma50_c, name="MA50",
                                         mode="lines", line=dict(color="#2ca02c")), row=1, col=1)
                fig.add_trace(go.Scatter(x=ma200_c.index, y=ma200_c, name="MA200",
                                         mode="lines", line=dict(color="#d62728")), row=1, col=1)

                smin = float(np.nanmin(s))
                smax = float(np.nanmax(s))
                if np.isfinite(smin) and np.isfinite(smax) and smin != smax:
                    pad = 0.05 * (smax - smin + 1e-9)
                    fig.update_yaxes(range=[smin - pad, smax + pad], row=1, col=1)

                fig.update_yaxes(
                    type="log" if view_mode == "Log price" else "linear",
                    title_text="Price",
                    tickformat=",.2f",
                    tickprefix="$",
                    row=1, col=1,
                )

                # ---- SAFETY NET: overlay axis ----
                if not (np.isfinite(smin) and np.isfinite(smax)) or smin == smax:
                    # allow a small range so something renders
                    if not np.isfinite(smin) or not np.isfinite(smax):
                        smin, smax = 0.0, float(s.iloc[-1]) if s.size else 1.0
                    if smin == smax:
                        smin -= 0.5
                        smax += 0.5
                fig.update_layout(
                    yaxis5=dict(
                        title="Price (overlay)",
                        overlaying="y",
                        side="right",
                        range=[smin, smax]
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=s.index, y=s,
                        name="Adj Close (overlay)",
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0.45)", width=1, dash="dot"),
                        yaxis="y5",
                    ),
                    row=1, col=1
                )

        # ----- VOLUME -----
        if vol.size > 0:
            fig.add_trace(
                go.Bar(x=vol.index, y=vol, name="Volume", marker_color="rgba(100,100,100,0.5)"),
                row=2, col=1
            )
            fig.update_yaxes(title_text="Volume", row=2, col=1)

        # ----- RSI -----
        if rsi_series.size > 0:
            fig.add_trace(
                go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI(14)",
                           line=dict(color="gray")),
                row=3, col=1
            )
            # RSI bands
            fig.add_hline(y=70, line_color="green", line_width=1, row=3, col=1)
            fig.add_hline(y=30, line_color="red", line_width=1, row=3, col=1)
            fig.update_yaxes(title_text="RSI(14)", range=[0, 100], row=3, col=1)

        # ----- MACD -----
        if macd_c.size > 0:
            fig.add_trace(go.Scatter(x=macd_c.index, y=macd_c, name="MACD", line=dict(color="#9467bd")),
                          row=4, col=1)
            fig.add_trace(go.Scatter(x=sig_c.index, y=sig_c, name="Signal", line=dict(color="#ff7f0e")),
                          row=4, col=1)
            # histogram bars
            colors = np.where(hist_c >= 0, "rgba(40,167,69,0.6)", "rgba(220,53,69,0.6)")
            fig.add_trace(go.Bar(x=hist_c.index, y=hist_c, name="Hist", marker_color=colors, opacity=0.6),
                          row=4, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)

        fig.update_layout(
            height=800,
            showlegend=True,
            margin=dict(l=40, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Configure your filters in the sidebar and click **Run Scan**.")
