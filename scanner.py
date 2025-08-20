import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from io import BytesIO

# =========================
# Password protection (uses Streamlit Secrets)
# =========================
def check_password():
    def password_entered():
        if st.session_state.get("password") == st.secrets["password"]["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

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
    return sorted(df["Symbol"].unique().tolist())

@st.cache_data(show_spinner=False)
def download_prices(tickers, period="1y"):
    """Download adjusted close prices (wide DataFrame: date index x tickers columns)."""
    px = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(px, pd.DataFrame) and "Close" in px.columns:
        px = px["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")

def rsi_wilder(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def moving_average(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.rolling(window).mean()

def recent_crosses(short_ma: pd.DataFrame, long_ma: pd.DataFrame, lookback: int) -> dict:
    res = {}
    common_cols = [c for c in short_ma.columns if c in long_ma.columns]
    for t in common_cols:
        s = short_ma[t].dropna()
        l = long_ma[t].dropna()
        n = min(len(s), len(l))
        if n < 2:
            res[t] = None
            continue
        s_recent = s.iloc[-(lookback+1):]
        l_recent = l.iloc[-(lookback+1):]
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

@st.cache_data(show_spinner=False)
def fetch_fundamentals(tickers):
    """
    Fetch a small set of fundamentals using yfinance.
    Returns DataFrame indexed by ticker with columns:
    MarketCap (USD), PE (trailingPE), DividendYield (%), Beta
    """
    data = []
    yobjs = yf.Tickers(tickers)
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
        data.append({"Ticker": t, "MarketCap": mc, "PE": pe, "DividendYield": dy, "Beta": beta})
    df = pd.DataFrame(data).set_index("Ticker")
    return df

def billions(x):
    if pd.isna(x):
        return np.nan
    return float(x) / 1e9

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Filters")

# Universe: S&P 500 only (auto)
sp500_tickers = load_sp500_symbols()
st.sidebar.caption(f"S&P 500 universe loaded: {len(sp500_tickers)} tickers")

# Technical Parameters
period = st.sidebar.selectbox("Price history period", ["6mo", "1y", "2y"], index=1)
rsi_min, rsi_max = st.sidebar.slider("RSI Range (14)", 0, 100, (30, 70), step=1)
price_vs_ma50 = st.sidebar.selectbox("Price vs 50-day MA", ["Any", "Above", "Below"], index=0)
price_vs_ma200 = st.sidebar.selectbox("Price vs 200-day MA", ["Any", "Above", "Below"], index=0)

cross_filter_on = st.sidebar.checkbox("Require recent MA crossover (50 vs 200)?", value=False)
if cross_filter_on:
    cross_type = st.sidebar.selectbox("Crossover Type", ["Golden", "Death"], index=0)
    cross_lookback = st.sidebar.number_input("Lookback days for crossover", min_value=1, max_value=60, value=5, step=1)

# Fundamental Filters
st.sidebar.subheader("Fundamental Filters")
use_fundamentals = st.sidebar.checkbox("Enable fundamental filters", value=False, help="Fetching fundamentals may add 10–30 seconds.")
if use_fundamentals:
    cap_min_b, cap_max_b = st.sidebar.slider("Market Cap (Billions $)", 0.0, 3000.0, (0.0, 3000.0), step=10.0)
    pe_min, pe_max = st.sidebar.slider("P/E (Trailing)", 0.0, 200.0, (0.0, 200.0), step=1.0)
    dy_min = st.sidebar.slider("Dividend Yield (%, min)", 0.0, 15.0, 0.0, step=0.1)
    beta_min, beta_max = st.sidebar.slider("Beta range", -1.0, 4.0, (0.0, 2.0), step=0.1)

run = st.sidebar.button("Run Scan")

# =========================
# Main Scan
# =========================
if run:
    with st.spinner("Fetching S&P 500 prices and calculating indicators..."):
        prices = download_prices(sp500_tickers, period=period)
        rsi14 = rsi_wilder(prices, period=14)
        ma50 = moving_average(prices, 50)
        ma200 = moving_average(prices, 200)

        latest_idx = prices.dropna().index.max()
        latest = pd.DataFrame({
            "Price": prices.loc[latest_idx],
            "RSI14": rsi14.loc[latest_idx],
            "MA50": ma50.loc[latest_idx],
            "MA200": ma200.loc[latest_idx],
        }).dropna()

        # Technical Filters
        filt = (latest["RSI14"].between(rsi_min, rsi_max))

        if price_vs_ma50 != "Any":
            if price_vs_ma50 == "Above":
                filt &= latest["Price"] > latest["MA50"]
            else:
                filt &= latest["Price"] < latest["MA50"]

        if price_vs_ma200 != "Any":
            if price_vs_ma200 == "Above":
                filt &= latest["Price"] > latest["MA200"]
            else:
                filt &= latest["Price"] < latest["MA200"]

        if cross_filter_on:
            crosses = recent_crosses(ma50, ma200, lookback=cross_lookback)
            cross_mask = latest.index.to_series().map(lambda t: crosses.get(t) == cross_type)
            filt &= cross_mask.fillna(False)

        results = latest[filt].sort_index()

    # Fundamentals (optional)
    if use_fundamentals and len(results) > 0:
        with st.spinner("Fetching fundamentals (market cap, P/E, dividend yield, beta)..."):
            fundamentals_df = fetch_fundamentals(results.index.tolist())
            # Apply fundamental filters
            fund_mask = pd.Series(True, index=fundamentals_df.index)

            mc_b = fundamentals_df["MarketCap"].map(billions)
            fund_mask &= mc_b.between(cap_min_b, cap_max_b)

            pe_vals = fundamentals_df["PE"]
            pe_ok = pe_vals.between(pe_min, pe_max).fillna(False)
            fund_mask &= pe_ok

            dy_vals = fundamentals_df["DividendYield"]
            dy_ok = dy_vals.fillna(0.0) >= dy_min
            fund_mask &= dy_ok

            beta_vals = fundamentals_df["Beta"]
            beta_ok = beta_vals.between(beta_min, beta_max).fillna(False)
            fund_mask &= beta_ok

            results = results.join(fundamentals_df[fund_mask], how="inner")

    # Display
    st.success(f"Found {len(results)} match(es).")
    if len(results) == 0:
        st.info("No matches. Loosen filters and try again.")
    else:
        if "MarketCap" in results.columns:
            results = results.copy()
            results["MarketCap ($B)"] = results["MarketCap"].map(billions).round(2)
            first_cols = ["Price", "RSI14", "MA50", "MA200"]
            extra = [c for c in ["MarketCap ($B)", "PE", "DividendYield", "Beta"] if c in results.columns]
            results = results[first_cols + extra]

        st.dataframe(results, use_container_width=True)

        # Sparkline
        with st.expander("Show mini price chart (sparkline) for a ticker"):
            sel = st.selectbox("Choose ticker", results.index.tolist())
            if sel:
                hist = download_prices([sel], period="6mo")
                st.line_chart(hist, height=180)

        # Export options
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

else:
    st.info("Set your filters in the sidebar and click **Run Scan**.")
