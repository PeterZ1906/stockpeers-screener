import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from io import BytesIO

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Password protection (Streamlit Secrets)
# =========================
def check_password():
    def password_entered():
        expected = st.secrets.get("password", {}).get("app_password")
        if expected and st.session_state.get("password") == expected:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password" not in st.secrets or "app_password" not in st.secrets.get("password", {}):
        st.warning(
            "Password secret not set. In Streamlit Cloud go to **Settings → Secrets** and add:\n\n"
            "[password]\napp_password = \"StockPeers2024!\""
        )
        st.stop()

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        st.stop()

check_password()

# =========================
# App Title
# =========================
st.set_page_config(page_title="StockPeers Screener", layout="wide")
st.title("📊 StockPeers Screener")

# =========================
# Data helpers
# =========================
@st.cache_data(show_spinner=False)
def load_sp500_symbols():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    return sorted(df["Symbol"].dropna().unique().tolist())

@st.cache_data(show_spinner=False)
def download_prices(tickers, period="1y"):
    """Adjusted close prices in wide format (index=date, columns=tickers)."""
    px = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(px, pd.DataFrame) and "Close" in px.columns:
        px = px["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all", axis=1)

def rsi_wilder(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def moving_average(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.rolling(window).mean()

def recent_crosses(short_ma: pd.DataFrame, long_ma: pd.DataFrame, lookback: int) -> dict:
    """Golden/Death crosses in the last `lookback` rows (per ticker), index-aligned."""
    res = {}
    for t in short_ma.columns.intersection(long_ma.columns):
        s = short_ma[t].dropna()
        l = long_ma[t].dropna()
        if len(s) < 2 or len(l) < 2:
            res[t] = None
            continue
        s_aligned, l_aligned = s.align(l, join="inner")
        if s_aligned.empty or l_aligned.empty:
            res[t] = None
            continue
        window = max(2, int(lookback) + 1)
        s_recent = s_aligned.tail(window)
        l_recent = l_aligned.tail(window)
        if len(s_recent) < 2 or len(l_recent) < 2:
            res[t] = None
            continue
        cross_up = (s_recent > l_recent) & (s_recent.shift(1) <= l_recent.shift(1))
        cross_down = (s_recent < l_recent) & (s_recent.shift(1) >= l_recent.shift(1))
        if cross_up.any():
            res[t] = "Golden"
        elif cross_down.any():
            res[t] = "Death"
        else:
            res[t] = None
    return res

def macd(prices: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def billions(x):
    if pd.isna(x):
        return np.nan
    return float(x) / 1e9

@st.cache_data(show_spinner=False)
def fetch_fundamentals(tickers):
    yobjs = yf.Tickers(tickers)
    rows = []
    for t in tickers:
        try:
            info = yobjs.tickers[t].info
        except Exception:
            info = {}
        mc = info.get("marketCap", np.nan)
        pe = info.get("trailingPE", np.nan)
        dy = info.get("dividendYield", np.nan)
        if dy is not None and not pd.isna(dy):
            dy = float(dy) * 100.0
        beta = info.get("beta", np.nan)
        rows.append({"Ticker": t, "MarketCap": mc, "PE": pe, "DividendYield": dy, "Beta": beta})
    return pd.DataFrame(rows).set_index("Ticker")

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Filters")

sp500_tickers = load_sp500_symbols()
st.sidebar.caption(f"S&P 500 universe loaded: {len(sp500_tickers)} tickers")

period = st.sidebar.selectbox("Price history period", ["6mo", "1y", "2y"], index=1)
rsi_min, rsi_max = st.sidebar.slider("RSI Range (14)", 0, 100, (30, 70), step=1)
price_vs_ma50 = st.sidebar.selectbox("Price vs 50-day MA", ["Any", "Above", "Below"], index=0)
price_vs_ma200 = st.sidebar.selectbox("Price vs 200-day MA", ["Any", "Above", "Below"], index=0)

cross_filter_on = st.sidebar.checkbox("Require recent MA crossover (50 vs 200)?", value=False)
if cross_filter_on:
    cross_type = st.sidebar.selectbox("Crossover Type", ["Golden", "Death"], index=0)
    cross_lookback = st.sidebar.number_input("Lookback days for crossover", min_value=1, max_value=60, value=20, step=1)

st.sidebar.subheader("MACD")
macd_on = st.sidebar.checkbox("Enable MACD filter", value=False)
if macd_on:
    macd_mode = st.sidebar.selectbox(
        "MACD condition",
        ["Line > Signal", "Histogram > 0", "Recent Bullish Cross (cross up)"],
        index=0
    )
    macd_lookback = st.sidebar.number_input("Lookback days for MACD cross", 1, 60, 10, 1)

st.sidebar.subheader("Fundamental Filters")
use_fundamentals = st.sidebar.checkbox("Enable fundamental filters", value=False, help="Fetching fundamentals may add 10–30 seconds.")
if use_fundamentals:
    cap_min_b, cap_max_b = st.sidebar.slider("Market Cap (Billions $)", 0.0, 3000.0, (0.0, 3000.0), step=10.0)
    pe_min, pe_max = st.sidebar.slider("P/E (Trailing)", 0.0, 200.0, (0.0, 200.0), step=1.0)
    dy_min = st.sidebar.slider("Dividend Yield (%, min)", 0.0, 15.0, 0.0, step=0.1)
    beta_min, beta_max = st.sidebar.slider("Beta range", -1.0, 4.0, (0.0, 2.0), step=0.1)

run = st.sidebar.button("Run Scan")

# ============================================
# Run the scan only when button is pressed
# Persist outputs in st.session_state["scan"]
# ============================================
if run:
    with st.spinner("Fetching S&P 500 prices and calculating indicators..."):
        prices = download_prices(sp500_tickers, period=period)
        if prices is None or prices.empty:
            st.error("Could not download price data. Please try again.")
            st.stop()

        rsi14 = rsi_wilder(prices, period=14)
        ma50 = moving_average(prices, 50)
        ma200 = moving_average(prices, 200)
        macd_line, macd_signal, macd_hist = macd(prices)

        prices_filled = prices.ffill()
        latest = pd.DataFrame({
            "Price": prices_filled.iloc[-1],
            "RSI14": rsi14.iloc[-1],
            "MA50": ma50.iloc[-1],
            "MA200": ma200.iloc[-1],
        }).dropna()

        filt = latest["RSI14"].between(rsi_min, rsi_max)

        if price_vs_ma50 != "Any":
            filt &= (latest["Price"] > latest["MA50"]) if price_vs_ma50 == "Above" else (latest["Price"] < latest["MA50"])
        if price_vs_ma200 != "Any":
            filt &= (latest["Price"] > latest["MA200"]) if price_vs_ma200 == "Above" else (latest["Price"] < latest["MA200"])

        if cross_filter_on:
            crosses = recent_crosses(ma50, ma200, lookback=int(cross_lookback))
            cross_mask = latest.index.to_series().map(lambda t: crosses.get(t) == cross_type)
            filt &= cross_mask.fillna(False)

        if macd_on:
            if macd_mode == "Line > Signal":
                macd_mask = (macd_line.iloc[-1] > macd_signal.iloc[-1])
            elif macd_mode == "Histogram > 0":
                macd_mask = (macd_hist.iloc[-1] > 0)
            else:
                cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
                macd_mask = cross.tail(int(macd_lookback)).any(axis=0)
            macd_mask = macd_mask.reindex(latest.index).fillna(False)
            filt &= macd_mask

        results = latest[filt].sort_index()

        # Fundamentals (optional)
        if use_fundamentals and len(results) > 0:
            fundamentals_df = fetch_fundamentals(results.index.tolist())
            fund_mask = pd.Series(True, index=fundamentals_df.index)
            mc_b = fundamentals_df["MarketCap"].map(billions)
            fund_mask &= mc_b.between(cap_min_b, cap_max_b)
            fund_mask &= fundamentals_df["PE"].between(pe_min, pe_max).fillna(False)
            fund_mask &= fundamentals_df["DividendYield"].fillna(0.0) >= dy_min
            fund_mask &= fundamentals_df["Beta"].between(beta_min, beta_max).fillna(False)
            results = results.join(fundamentals_df[fund_mask], how="inner")

        # Score (simple composite)
        def z(x): return (x - x.mean()) / (x.std(ddof=0) + 1e-9)
        if len(results) > 0:
            rsi_sweet = - (results["RSI14"] - 50).abs()
            trend_pts = (results["Price"] > results["MA50"]).astype(int) + (results["Price"] > results["MA200"]).astype(int)
            macd_boost = pd.Series(0.0, index=results.index)
            try:
                macd_boost = macd_hist.iloc[-1].reindex(results.index).fillna(0.0)
            except Exception:
                pass
            results["Score"] = z(rsi_sweet) + z(trend_pts) + z(macd_boost)
            results = results.sort_values("Score", ascending=False)

        # >>> Persist all needed objects so UI changes don't wipe results
        st.session_state["scan"] = {
            "results": results,
            "prices": prices,              # for sparkline/extra plots if needed
            "period": period,
        }

# =========================
# Render results (from state if available)
# =========================
scan = st.session_state.get("scan")

if scan is None or scan.get("results") is None:
    st.info("Set your filters in the sidebar and click **Run Scan**.")
else:
    results = scan["results"]
    period = scan["period"]

    st.success(f"Found {len(results)} match(es).")
    if len(results) == 0:
        st.info("No matches. Loosen filters and try again (wider RSI, MA = Any, longer lookbacks).")
    else:
        if "MarketCap" in results.columns:
            results = results.copy()
            if "MarketCap ($B)" not in results.columns:
                results["MarketCap ($B)"] = results["MarketCap"].map(billions).round(2)
            base_cols = ["Score"] if "Score" in results.columns else []
            first_cols = base_cols + ["Price", "RSI14", "MA50", "MA200"]
            extras = [c for c in ["MarketCap ($B)", "PE", "DividendYield", "Beta"] if c in results.columns]
            results = results[first_cols + extras]
        st.dataframe(results, use_container_width=True)

        # ---------- Full interactive chart ----------
        with st.expander("Open full interactive chart"):
            sel = st.selectbox("Choose ticker", results.index.tolist(), key="chart_ticker")
            if sel:
                # Use raw OHLC (not auto_adjust) so candlesticks are valid
                ohlc = yf.download(sel, period=period, auto_adjust=False, progress=False)
                needed = ["Open", "High", "Low", "Close"]
                if not set(needed).issubset(ohlc.columns):
                    st.info("Not enough OHLC data to chart this ticker right now.")
                else:
                    ohlc = ohlc[needed].dropna()
                    if len(ohlc) < 10:
                        st.info("Too few data points to draw a meaningful chart.")
                    else:
                        close = ohlc["Close"]
                        ma50_s = close.rolling(50).mean()
                        ma200_s = close.rolling(200).mean()
                        # MACD (recomputed on the same series)
                        ema_fast = close.ewm(span=12, adjust=False).mean()
                        ema_slow = close.ewm(span=26, adjust=False).mean()
                        macd_line_s = ema_fast - ema_slow
                        macd_signal_s = macd_line_s.ewm(span=9, adjust=False).mean()
                        macd_hist_s = macd_line_s - macd_signal_s

                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            row_heights=[0.7, 0.3], vertical_spacing=0.05)

                        fig.add_trace(go.Candlestick(
                            x=ohlc.index, open=ohlc["Open"], high=ohlc["High"],
                            low=ohlc["Low"], close=ohlc["Close"], name="OHLC"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=ma50_s.index, y=ma50_s, name="MA50"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=ma200_s.index, y=ma200_s, name="MA200"), row=1, col=1)

                        fig.add_trace(go.Scatter(x=macd_line_s.index, y=macd_line_s, name="MACD"), row=2, col=1)
                        fig.add_trace(go.Scatter(x=macd_signal_s.index, y=macd_signal_s, name="Signal"), row=2, col=1)
                        fig.add_trace(go.Bar(x=macd_hist_s.index, y=macd_hist_s, name="Hist"), row=2, col=1)

                        # Hide weekend gaps to make the chart dense
                        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

                        fig.update_layout(height=700, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
                        st.plotly_chart(fig, use_container_width=True)

        # ---------- Export ----------
        st.subheader("Download Results")
        fmt = st.selectbox("Choose format", ["CSV", "Excel (.xlsx)"], index=0)
        if fmt == "CSV":
            csv = results.to_csv().encode("utf-8")
            st.download_button("Download CSV", csv, file_name="stock_scan_results.csv", mime="text/csv")
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                results.to_excel(writer, sheet_name="Results")
            st.download_button(
                "Download Excel",
                data=output.getvalue(),
                file_name="stock_scan_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
