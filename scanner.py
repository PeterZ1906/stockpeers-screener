# scanner.py
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from functools import lru_cache

# -------------------------
# Util / TA helpers
# -------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd_line(series: pd.Series, fast=12, slow=26) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def macd_signal(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd = macd_line(series, fast, slow)
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def crossed(series_a: pd.Series, series_b: pd.Series, lookback: int, direction: str = "up") -> bool:
    """
    direction: "up" (a crossed above b), "down" (a crossed below b), or "any"
    """
    if len(series_a) < lookback + 2 or len(series_b) < lookback + 2:
        return False
    a = series_a.dropna().tail(lookback + 2)
    b = series_b.dropna().tail(lookback + 2)
    if len(a) != len(b) or len(a) < 2:
        return False
    prev = a.iloc[-2] - b.iloc[-2]
    curr = a.iloc[-1] - b.iloc[-1]
    if direction == "up":
        return prev < 0 and curr > 0
    elif direction == "down":
        return prev > 0 and curr < 0
    else:
        return (prev < 0 and curr > 0) or (prev > 0 and curr < 0)

# -------------------------
# Universe
# -------------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers() -> List[str]:
    # Lightweight cached list to avoid web scraping dependencies
    # You can update/extend this list easily if you want.
    # (short list here as placeholder; replace with full S&P 500 list if you wish)
    tickers = [
        "AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","BRK-B","XOM","UNH",
        "JNJ","JPM","V","PG","HD","CVX","MA","LLY","ABBV","PEP","KO","PFE","MRK",
        "AVGO","COST","TMO","DIS","WMT","BAC","CSCO","ADBE","NFLX","ORCL","CRM",
        "LIN","ACN","TXN","ABT","DHR","QCOM","MCD"
    ]
    return tickers

# -------------------------
# Data
# -------------------------
@st.cache_data(show_spinner=False)
def get_prices(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.title)  # Close, Volume
    return df

# -------------------------
# Settings structure
# -------------------------
@dataclass
class Settings:
    period: str
    rsi_range: Tuple[int, int]

    price_vs_ma50: str
    price_vs_ma200: str

    require_cross: bool
    cross_type: str     # Golden / Death / Any
    cross_lookback: int

    macd_enable: bool
    macd_condition: str # Line > Signal / Line < Signal / Crossed up / Crossed down / Any
    macd_lookback: int

    fundamentals_enable: bool
    mc_min: float
    mc_max: float
    pe_max: float
    dy_min: float
    beta_max: float

    scoring_enable: bool
    w_rsi: float
    w_trend: float
    w_macd: float
    w_cross: float

    top_n: int
    universe: str       # "S&P 500"

# -------------------------
# Sidebar Filters (restores everything)
# -------------------------
def render_filters() -> Settings:
    with st.sidebar:
        st.header("Filters")

        with st.form("filters_form", clear_on_submit=False):
            period = st.selectbox(
                "Price history period",
                ["6mo","1y","2y","5y","10y","max"],
                index=1,
            )
            rsi_range = st.slider("RSI Range (14)", 0, 100, (30, 70))

            st.markdown("**Price vs 50-day MA**")
            price_vs_ma50 = st.selectbox("Price vs 50-day MA", ["Any","Above","Below"], index=0)

            st.markdown("**Price vs 200-day MA**")
            price_vs_ma200 = st.selectbox("Price vs 200-day MA", ["Any","Above","Below"], index=0)

            require_cross = st.checkbox("Require recent MA crossover (50 vs 200)?", value=False)
            cross_type = st.selectbox("Crossover Type", ["Golden","Death","Any"], index=0, disabled=not require_cross)
            cross_lookback = st.number_input("Lookback days for crossover", 1, 60, 10, disabled=not require_cross)

            st.markdown("---")
            st.subheader("MACD")
            macd_enable = st.checkbox("Enable MACD filter", value=False)
            macd_condition = st.selectbox(
                "MACD condition",
                ["Line > Signal","Line < Signal","Crossed up","Crossed down","Any"],
                index=0,
                disabled=not macd_enable
            )
            macd_lookback = st.number_input("Lookback days for MACD cross", 1, 60, 10, disabled=not macd_enable)

            st.markdown("---")
            st.subheader("Fundamental Filters")
            fundamentals_enable = st.checkbox("Enable fundamental filters", value=False, help="Requires Yahoo key stats; we use simple Yahoo fields.")
            mc_min = st.number_input("Market Cap min ($B)", 0.0, 5000.0, 0.0, 10.0, disabled=not fundamentals_enable)
            mc_max = st.number_input("Market Cap max ($B)", 0.0, 5000.0, 9999.0, 10.0, disabled=not fundamentals_enable)
            pe_max = st.number_input("P/E max", 0.0, 1000.0, 1000.0, 1.0, disabled=not fundamentals_enable)
            dy_min = st.number_input("Dividend Yield min (%)", 0.0, 50.0, 0.0, 0.1, disabled=not fundamentals_enable)
            beta_max = st.number_input("Beta max (5y monthly)", 0.0, 10.0, 10.0, 0.1, disabled=not fundamentals_enable)

            st.markdown("---")
            st.subheader("Scoring & Ranking")
            scoring_enable = st.checkbox("Enable ranking / composite score", value=True)
            w_rsi = st.slider("Weight: RSI sweet spot", 0.0, 2.0, 1.0, 0.1, disabled=not scoring_enable)
            w_trend = st.slider("Weight: Trend vs MAs", 0.0, 2.0, 1.0, 0.1, disabled=not scoring_enable)
            w_macd = st.slider("Weight: MACD momentum", 0.0, 2.0, 1.0, 0.1, disabled=not scoring_enable)
            w_cross = st.slider("Weight: Golden/Death boost", 0.0, 2.0, 0.5, 0.1, disabled=not scoring_enable)

            top_n = st.number_input("Show Top N", 5, 100, 25, 1)

            universe = st.selectbox("Universe", ["S&P 500"], index=0)

            submitted = st.form_submit_button("Run Scan", use_container_width=True)

    if submitted:
        st.session_state["_run_scan"] = True

    # keep persistent
    settings = Settings(
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
    return settings

# -------------------------
# (Optional) fundamentals — defensive defaults
# -------------------------
def get_simple_fundamentals(ticker: str) -> Dict[str, float]:
    """
    Pull a few simple fields via yfinance.info.
    If unavailable, return NaNs and let filters ignore.
    """
    out = {"marketCapB": np.nan, "trailingPE": np.nan, "dividendYieldPct": np.nan, "beta": np.nan}
    try:
        info = yf.Ticker(ticker).info
        if "marketCap" in info and info["marketCap"]:
            out["marketCapB"] = info["marketCap"] / 1e9
        if "trailingPE" in info and info["trailingPE"]:
            out["trailingPE"] = float(info["trailingPE"])
        if "dividendYield" in info and info["dividendYield"]:
            out["dividendYieldPct"] = float(info["dividendYield"]) * 100.0
        if "beta" in info and info["beta"]:
            out["beta"] = float(info["beta"])
    except Exception:
        pass
    return out

# -------------------------
# Scoring
# -------------------------
def compute_score(close: pd.Series, rsi14: pd.Series, ma50: pd.Series, ma200: pd.Series,
                  macd_hist: pd.Series, settings: Settings, did_cross_golden: bool, did_cross_death: bool) -> float:
    if not settings.scoring_enable:
        return 0.0

    score = 0.0

    # RSI sweet spot around 50: the closer, the higher
    if not rsi14.dropna().empty:
        last_rsi = rsi14.iloc[-1]
        score += settings.w_rsi * (1.0 - min(abs(last_rsi - 50) / 50, 1.0))

    # Trend vs MAs: + points for above 50/200, - for below both
    if not close.dropna().empty and not ma50.dropna().empty and not ma200.dropna().empty:
        c = close.iloc[-1]
        s50 = ma50.iloc[-1]
        s200 = ma200.iloc[-1]
        trend = 0
        trend += 1 if c > s50 else -1
        trend += 1 if c > s200 else -1
        score += settings.w_trend * (trend / 2.0)  # normalize to [-1,1]

    # MACD momentum: recent average histogram
    if not macd_hist.dropna().empty:
        last_hist = macd_hist.tail(5).mean()
        score += settings.w_macd * (np.tanh(last_hist))

    # Crossover boost
    if did_cross_golden:
        score += settings.w_cross * 1.0
    if did_cross_death:
        score -= settings.w_cross * 1.0

    return float(score)

# -------------------------
# Single-ticker evaluation
# -------------------------
def evaluate_ticker(t: str, settings: Settings) -> Dict:
    df = get_prices(t, settings.period)
    if df.empty or "Close" not in df:
        return {}

    close = df["Close"].copy()
    vol = df.get("Volume", pd.Series(index=close.index, dtype=float))

    ma50 = sma(close, 50)
    ma200 = sma(close, 200)
    rsi14 = rsi(close, 14)
    m_line, m_sig, m_hist = macd_signal(close)

    # --- FILTERS ---
    # RSI
    if not rsi14.dropna().empty:
        last_rsi = rsi14.iloc[-1]
        if not (settings.rsi_range[0] <= last_rsi <= settings.rsi_range[1]):
            return {}

    # Price vs MA50 / MA200
    if not ma50.dropna().empty and settings.price_vs_ma50 != "Any":
        c = close.iloc[-1]; s50 = ma50.iloc[-1]
        if settings.price_vs_ma50 == "Above" and not (c > s50):
            return {}
        if settings.price_vs_ma50 == "Below" and not (c < s50):
            return {}

    if not ma200.dropna().empty and settings.price_vs_ma200 != "Any":
        c = close.iloc[-1]; s200 = ma200.iloc[-1]
        if settings.price_vs_ma200 == "Above" and not (c > s200):
            return {}
        if settings.price_vs_ma200 == "Below" and not (c < s200):
            return {}

    # Require MA cross
    did_cross_golden = crossed(ma50, ma200, settings.cross_lookback, "up")
    did_cross_death = crossed(ma50, ma200, settings.cross_lookback, "down")
    if settings.require_cross:
        if settings.cross_type == "Golden" and not did_cross_golden:
            return {}
        if settings.cross_type == "Death" and not did_cross_death:
            return {}
        if settings.cross_type == "Any" and not (did_cross_golden or did_cross_death):
            return {}

    # MACD condition
    if settings.macd_enable:
        if settings.macd_condition == "Line > Signal" and not (m_line.iloc[-1] > m_sig.iloc[-1]):
            return {}
        if settings.macd_condition == "Line < Signal" and not (m_line.iloc[-1] < m_sig.iloc[-1]):
            return {}
        if settings.macd_condition == "Crossed up" and not crossed(m_line, m_sig, settings.macd_lookback, "up"):
            return {}
        if settings.macd_condition == "Crossed down" and not crossed(m_line, m_sig, settings.macd_lookback, "down"):
            return {}

    # Fundamentals (simple, optional)
    if settings.fundamentals_enable:
        f = get_simple_fundamentals(t)
        # Skip if any required bound violated; NaN passes
        if not math.isnan(f["marketCapB"]):
            if not (settings.mc_min <= f["marketCapB"] <= settings.mc_max):
                return {}
        if not math.isnan(f["trailingPE"]):
            if not (f["trailingPE"] <= settings.pe_max):
                return {}
        if not math.isnan(f["dividendYieldPct"]):
            if not (f["dividendYieldPct"] >= settings.dy_min):
                return {}
        if not math.isnan(f["beta"]):
            if not (f["beta"] <= settings.beta_max):
                return {}

    # --- SCORE ---
    score = compute_score(close, rsi14, ma50, ma200, m_hist, settings, did_cross_golden, did_cross_death)

    return {
        "Ticker": t,
        "Score": round(score, 4),
        "Price": round(float(close.iloc[-1]), 2),
        "RSI14": round(float(rsi14.iloc[-1]), 2) if not rsi14.dropna().empty else np.nan,
        "MA50": round(float(ma50.iloc[-1]), 2) if not ma50.dropna().empty else np.nan,
        "MA200": round(float(ma200.iloc[-1]), 2) if not ma200.dropna().empty else np.nan,
        "CloseSeries": close,
        "VolSeries": vol,
        "RSISeries": rsi14,
        "MA50Series": ma50,
        "MA200Series": ma200,
        "MACD": m_line,
        "MACDSig": m_sig,
        "MACDHist": m_hist
    }

# -------------------------
# Run Scan
# -------------------------
def run_scan(settings: Settings) -> pd.DataFrame:
    if settings.universe == "S&P 500":
        tickers = get_sp500_tickers()
    else:
        tickers = get_sp500_tickers()

    results: List[Dict] = []
    for t in tickers:
        try:
            row = evaluate_ticker(t, settings)
            if row:
                results.append(row)
        except Exception:
            continue

    if not results:
        st.info("No matches passed filters. Showing fallback (top tickers by last price).")
        fallback = []
        for t in tickers[:50]:
            df = get_prices(t, settings.period)
            if df.empty: 
                continue
            fallback.append([t, 0.0, round(float(df['Close'].iloc[-1]), 2), np.nan, np.nan, np.nan])
        return pd.DataFrame(fallback, columns=["Ticker","Score","Price","RSI14","MA50","MA200"]).head(settings.top_n)

    df = pd.DataFrame([{
        "Ticker": r["Ticker"], "Score": r["Score"], "Price": r["Price"],
        "RSI14": r["RSI14"], "MA50": r["MA50"], "MA200": r["MA200"]
    } for r in results])
    return df.sort_values("Score", ascending=False).head(settings.top_n)

# -------------------------
# Chart
# -------------------------
def plot_interactive(selected: str, record: Dict):
    import plotly.graph_objects as go
    if not record:
        st.warning("No chartable data.")
        return

    close = record["CloseSeries"]
    vol = record["VolSeries"]
    rsi14 = record["RSISeries"]
    ma50 = record["MA50Series"]
    ma200 = record["MA200Series"]
    macd = record["MACD"]
    sig = record["MACDSig"]
    hist = record["MACDHist"]

    # Price panel
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Adj Close", line=dict(color="#2c7be5")))
    fig.add_trace(go.Scatter(x=close.index, y=ma50, name="MA50", line=dict(color="#00b894")))
    fig.add_trace(go.Scatter(x=close.index, y=ma200, name="MA200", line=dict(color="#d63031")))

    fig.update_layout(
        height=640,
        margin=dict(l=10,r=10,t=30,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # RSI panel
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rsi14.index, y=rsi14, name="RSI(14)", line=dict(color="#636e72")))
    fig2.add_hline(y=70, line_width=1, line_dash="dash", line_color="#e17055")
    fig2.add_hline(y=30, line_width=1, line_dash="dash", line_color="#00b894")
    fig2.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # MACD panel
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=macd.index, y=macd, name="MACD", line=dict(color="#6c5ce7")))
    fig3.add_trace(go.Scatter(x=sig.index, y=sig, name="Signal", line=dict(color="#fd9644")))
    fig3.add_trace(go.Bar(x=hist.index, y=hist, name="Hist", marker_color=np.where(hist>=0,"#2ecc71","#e74c3c")))
    fig3.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10), barmode="relative")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Main
# -------------------------
def main():
    st.set_page_config(page_title="StockPeers Screener", layout="wide")
    st.title("StockPeers Screener")

    settings = render_filters()

    # Only run scan when user presses "Run Scan"
    if st.session_state.get("_run_scan", False):
        with st.spinner("Scanning..."):
            results_df = run_scan(settings)
        st.session_state["results_df"] = results_df
        st.session_state["_run_scan"] = False

    results_df = st.session_state.get("results_df", pd.DataFrame(columns=["Ticker","Score","Price","RSI14","MA50","MA200"]))
    st.success(f"Found {len(results_df)} match(es).")
    st.dataframe(results_df, use_container_width=True, height=360)

    # ---- Chart block (outside the filters form, does NOT trigger a re-scan) ----
    st.subheader("Interactive Chart")
    tickers = results_df["Ticker"].tolist()
    selected = st.selectbox("Choose ticker", tickers, index=0 if tickers else None)
    if selected:
        # Recompute this ticker once for chart (or keep from cache/previous eval)
        rec = evaluate_ticker(selected, settings)
        plot_interactive(selected, rec)

if __name__ == "__main__":
    main()
