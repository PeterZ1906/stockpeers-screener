import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.set_page_config(page_title="StockPeers Screener", layout="wide")

# ==============================
# Robust S&P 500 ticker fetcher
# ==============================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_sp500_tickers() -> list[str]:
    """
    Robust S&P 500 tickers fetcher, with multiple fallbacks:
      1) yfinance.tickers_sp500()
      2) pandas.read_html (if available)
      3) GitHub CSV (datasets/s-and-p-500-companies)
    Returns upper-cased symbols with '.' replaced by '-'.
    """
    def _clean(sym_list: list[str]) -> list[str]:
        return [s.strip().upper().replace(".", "-") for s in sym_list if s and isinstance(s, str)]

    # 1) yfinance built-in
    try:
        syms = yf.tickers_sp500()
        if syms:
            return _clean(syms)
    except Exception:
        pass

    # 2) Wikipedia via pandas.read_html (needs lxml or html5lib)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)  # fails if parser not installed
        if tables and "Symbol" in tables[0].columns:
            return _clean(tables[0]["Symbol"].tolist())
    except Exception:
        pass

    # 3) Fallback CSV (no parser required)
    try:
        csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(csv_url)
        if "Symbol" in df.columns:
            return _clean(df["Symbol"].tolist())
    except Exception:
        pass

    # Last-resort emergency fallback
    return _clean(["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"])

# ==============================
# Technical indicator functions
# ==============================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ==============================
# Sidebar controls
# ==============================
st.sidebar.header("Filters")

price_period = st.sidebar.selectbox("Price history period", ["6mo", "1y", "2y", "5y"], index=1)
rsi_min, rsi_max = st.sidebar.slider("RSI Range (14)", 0, 100, (30, 70))
require_crossover = st.sidebar.checkbox("Require recent MA crossover (50 vs 200)?")
crossover_type = st.sidebar.selectbox("Crossover Type", ["Golden", "Death"])

top_n = st.sidebar.number_input("Show Top N", min_value=5, max_value=50, value=25)

# ==============================
# Run scan
# ==============================
@st.cache_data(show_spinner=True, ttl=900)
def run_scan(universe, period):
    results = []
    end = datetime.today()
    start = end - pd.DateOffset(years=1)

    for ticker in universe:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                continue
            df["MA50"] = moving_average(df["Adj Close"], 50)
            df["MA200"] = moving_average(df["Adj Close"], 200)
            df["RSI14"] = rsi(df["Adj Close"], 14)
            macd_line, signal_line, hist = macd(df["Adj Close"])
            df["MACD"] = macd_line
            df["Signal"] = signal_line
            df["Hist"] = hist

            latest = df.iloc[-1]
            score = 0
            if rsi_min <= latest["RSI14"] <= rsi_max:
                score += 1
            if latest["MA50"] > latest["MA200"]:
                score += 1
            if latest["MACD"] > latest["Signal"]:
                score += 1
            if require_crossover:
                cross = df["MA50"].iloc[-2] < df["MA200"].iloc[-2] and latest["MA50"] > latest["MA200"]
                if crossover_type == "Golden" and cross:
                    score += 1
                elif crossover_type == "Death" and not cross:
                    score += 1

            results.append([ticker, score, latest["Adj Close"], latest["RSI14"], latest["MA50"], latest["MA200"]])
        except Exception:
            continue

    results_df = pd.DataFrame(results, columns=["Ticker", "Score", "Price", "RSI14", "MA50", "MA200"])
    results_df = results_df.sort_values("Score", ascending=False).head(top_n)
    return results_df

universe = get_sp500_tickers()
results_df = run_scan(universe, price_period)

st.success(f"Found {len(results_df)} match(es).")
st.dataframe(results_df, use_container_width=True)

# ==============================
# Chart viewer
# ==============================
st.subheader("Interactive Chart")
ticker = st.selectbox("Choose ticker", results_df["Ticker"] if not results_df.empty else [])

if ticker:
    df = yf.download(ticker, period=price_period, progress=False)
    df["MA50"] = moving_average(df["Adj Close"], 50)
    df["MA200"] = moving_average(df["Adj Close"], 200)
    df["RSI14"] = rsi(df["Adj Close"], 14)
    macd_line, signal_line, hist = macd(df["Adj Close"])
    df["MACD"] = macd_line
    df["Signal"] = signal_line
    df["Hist"] = hist

    fig = make_chart(df, ticker)
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Chart builder
# ==============================
def make_chart(df, ticker):
    fig = go.Figure()

    # Price & moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name="Adj Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI(14)", yaxis="y2"))

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", yaxis="y3"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], mode="lines", name="Signal", yaxis="y3"))
    fig.add_trace(go.Bar(x=df.index, y=df["Hist"], name="Hist", yaxis="y3"))

    fig.update_layout(
        title=f"{ticker} Indicators",
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="RSI", overlaying="y", side="right", position=0.95, range=[0, 100]),
        yaxis3=dict(title="MACD", overlaying="y", side="right", position=1.05),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=700
    )
    return fig
