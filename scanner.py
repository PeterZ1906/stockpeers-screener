# scanner.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from io import BytesIO

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# App & Password
# =========================
st.set_page_config(page_title="StockPeers Screener", layout="wide")

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
st.title("📊 StockPeers Screener")

# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_sp500_symbols():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    return sorted(df["Symbol"].dropna().unique().tolist())

@st.cache_data(show_spinner=False)
def download_prices(tickers, period="1y"):
    """
    Returns a wide DataFrame indexed by date with columns=tickers, values=Adjusted Close.
    Handles both single and multiple tickers and unwraps MultiIndex columns.
    """
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    # Single ticker → simple frame with "Close" column (already adjusted)
    if isinstance(df, pd.DataFrame) and not isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns:
            out = df["Close"].to_frame()
        elif "Adj Close" in df.columns:
            out = df["Adj Close"].to_frame()
        else:
            out = df.select_dtypes(include=[np.number]).iloc[:, :1]
        out.columns = [tickers] if isinstance(tickers, str) else out.columns
        return out

    # MultiIndex (multiple tickers) → unwrap to Close/Adj Close level
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in set(lvl0):
            out = df["Close"]
        elif "Adj Close" in set(lvl0):
            out = df["Adj Close"]
        else:
            first = lvl0.unique()[0]
            out = df[first]
        out.columns = [str(c) for c in out.columns]  # plain ticker names
        out = out.dropna(how="all", axis=1)
        return out

    return pd.DataFrame()

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
    """Golden/Death crosses in last `lookback` rows (per ticker)."""
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
        w = max(2, int(lookback) + 1)
        s_recent = s_aligned.tail(w)
        l_recent = l_aligned.tail(w)
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
# Sidebar
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

# ---- Scoring & Ranking ----
st.sidebar.subheader("Scoring & Ranking")
enable_rank = st.sidebar.checkbox("Enable ranking / composite score", value=True)
w_rsi   = st.sidebar.slider("Weight: RSI sweet spot", 0.0, 3.0, 1.0, 0.1)
w_ma    = st.sidebar.slider("Weight: Trend vs MAs",   0.0, 3.0, 1.0, 0.1)
w_macd  = st.sidebar.slider("Weight: MACD momentum",  0.0, 3.0, 1.0, 0.1)
w_cross = st.sidebar.slider("Weight: Golden/Death boost", 0.0, 3.0, 0.5, 0.1)
top_n   = st.sidebar.number_input("Show Top N", min_value=1, max_value=100, value=25, step=1)

run = st.sidebar.button("Run Scan")

# =========================
# Run Scan
# =========================
if run:
    with st.spinner("Fetching S&P 500 prices and calculating indicators..."):
        prices = download_prices(sp500_tickers, period=period)
        if prices is None or prices.empty:
            st.error("Could not download price data. Please try again.")
            st.stop()

        rsi14 = rsi_wilder(prices, period=14)
        ma50  = moving_average(prices, 50)
        ma200 = moving_average(prices, 200)
        macd_line, macd_signal, macd_hist = macd(prices)

        prices_filled = prices.ffill()
        latest = pd.DataFrame({
            "Price": prices_filled.iloc[-1],
            "RSI14": rsi14.iloc[-1],
            "MA50":  ma50.iloc[-1],
            "MA200": ma200.iloc[-1],
        }).dropna()

        # --- base filter mask ---
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

        # ---------- Ranked composite score ----------
        def z(s: pd.Series) -> pd.Series:
            return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

        if enable_rank and len(latest) > 0:
            price_last = prices_filled.iloc[-1].reindex(latest.index)
            rsi_last   = rsi14.iloc[-1].reindex(latest.index)
            ma50_last  = ma50.iloc[-1].reindex(latest.index)
            ma200_last = ma200.iloc[-1].reindex(latest.index)
            macd_hist_last = macd_hist.iloc[-1].reindex(latest.index).fillna(0.0)

            # 1) RSI sweet spot (closer to 50 is better)
            rsi_sweet = -(rsi_last - 50.0).abs()

            # 2) Trend vs MAs: % above MA50 + % above MA200
            trend50  = (price_last - ma50_last)  / ma50_last.replace(0, np.nan)
            trend200 = (price_last - ma200_last) / ma200_last.replace(0, np.nan)
            trend_vs_mas = trend50.fillna(0.0) + trend200.fillna(0.0)

            # 3) MACD momentum (histogram)
            macd_momo = macd_hist_last

            # 4) Golden/Death boost
            cross_for_score = recent_crosses(ma50, ma200, lookback=60)
            cross_boost = pd.Series(0.0, index=latest.index)
            for t in cross_boost.index:
                v = cross_for_score.get(t)
                if v == "Golden":
                    cross_boost.loc[t] = 1.0
                elif v == "Death":
                    cross_boost.loc[t] = -1.0

            score = (
                w_rsi   * z(rsi_sweet)   +
                w_ma    * z(trend_vs_mas) +
                w_macd  * z(macd_momo)   +
                w_cross * z(cross_boost)
            )

            latest_scored = latest.copy()
            latest_scored["Score"] = score
            results = latest_scored[filt].sort_values("Score", ascending=False)
            if len(results) > top_n:
                results = results.head(int(top_n))
        else:
            results = latest[filt].sort_index()

        # Fundamentals (optional) AFTER scoring/filtering
        if use_fundamentals and len(results) > 0:
            fundamentals_df = fetch_fundamentals(results.index.tolist())
            fund_mask = pd.Series(True, index=fundamentals_df.index)
            mc_b = fundamentals_df["MarketCap"].map(billions)
            fund_mask &= mc_b.between(cap_min_b, cap_max_b)
            fund_mask &= fundamentals_df["PE"].between(pe_min, pe_max).fillna(False)
            fund_mask &= fundamentals_df["DividendYield"].fillna(0.0) >= dy_min
            fund_mask &= fundamentals_df["Beta"].between(beta_min, beta_max).fillna(False)
            results = results.join(fundamentals_df[fund_mask], how="inner")

        st.session_state["scan"] = {
            "results": results,
            "period": period,
        }

# =========================
# Render results (safe)
# =========================
scan = st.session_state.get("scan")

if scan is None or scan.get("results") is None:
    st.info("Set your filters in the sidebar and click **Run Scan**.")

    with st.expander("Open full interactive chart"):
        st.info("Run a scan first to enable charting.")

else:
    results = scan["results"]
    period  = scan["period"]

    st.success(f"Found {len(results)} match(es).")
    if len(results) == 0:
        st.info("No matches. Loosen filters and try again (wider RSI, MA = Any, longer lookbacks).")
    else:
        # Pretty table (+ Rank)
        tbl = results.copy()
        if "MarketCap" in tbl.columns:
            tbl["MarketCap ($B)"] = tbl["MarketCap"].map(billions).round(2)
        if "Score" in tbl.columns:
            tbl.insert(0, "Rank", tbl["Score"].rank(ascending=False, method="dense").astype(int))
        cols = [c for c in ["Rank", "Score", "Price", "RSI14", "MA50", "MA200",
                            "MarketCap ($B)", "PE", "DividendYield", "Beta"] if c in tbl.columns]
        st.dataframe(tbl[cols] if cols else tbl, use_container_width=True)

        # ----------------- CHART EXPANDER (Price + Volume + RSI + MACD + Cross markers) -----------------
        with st.expander("Open full interactive chart"):
            sel = st.selectbox("Choose ticker", results.index.tolist(), key="chart_ticker")
            chart_period = st.selectbox("Chart period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=1, key="chart_period")

            view_mode = st.radio(
                "View mode",
                ["Price (actual)", "% change (rebased)", "Log price"],
                index=0,
                horizontal=True,
                help="Switch to % or Log for long horizons to make moves visible."
            )
            focus = st.selectbox(
                "Focus window",
                ["Full", "Last 6m", "Last 1y", "Last 2y", "Last 5y"],
                index=1
            )

            def fetch_series(ticker, per):
                """Adjusted close series. 1d for <=2y; 1wk otherwise."""
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
                return s

            def fetch_ohlcv(ticker, per):
                """Optional OHLCV for candles & volume (NOT auto-adjusted)."""
                interval = "1d" if per in ["6mo", "1y", "2y"] else "1wk"
                df = yf.download(ticker, period=per, interval=interval, auto_adjust=False, progress=False)
                if isinstance(df, pd.DataFrame):
                    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
                    if cols:
                        df = df[cols].dropna()
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception:
                            pass
                        return df
                return pd.DataFrame()

            def finish_layout(fig, ytype="linear", ytitle_price="Price"):
                # Price axes
                fig.update_yaxes(type=ytype, title_text=ytitle_price,
                                 tickformat="$,.0f" if ytype != "percent" else ",.1f%",
                                 row=1, col=1, showspikes=True)
                # Volume axes
                fig.update_yaxes(title_text="Volume", row=2, col=1, tickformat=",.0f")
                # RSI axes
                fig.update_yaxes(title_text="RSI(14)", row=3, col=1, range=[0, 100])
                # MACD axes
                fig.update_yaxes(title_text="MACD", tickformat=".2f", row=4, col=1,
                                 zeroline=True, zerolinecolor="gray")

                fig.update_xaxes(
                    tickformat="%b %Y",
                    rangebreaks=[dict(bounds=["sat","mon"])],
                    showspikes=True
                )
                fig.update_layout(
                    height=880, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_rangeslider_visible=False,
                )

            if not sel:
                st.info("Choose a ticker above to draw the chart.")
            else:
                # 1) Base series
                widen = {"6mo":"1y","1y":"2y","2y":"5y","5y":"10y","10y":"max"}
                p = chart_period
                s = fetch_series(sel, p)
                while len(s) < 120 and p in widen:
                    p = widen[p]
                    s = fetch_series(sel, p)

                if s.empty:
                    st.info("No chartable data returned. Try a longer chart period.")
                else:
                    # 2) Focus window without refetching
                    if focus != "Full":
                        nmap = {"Last 6m": 180, "Last 1y": 365, "Last 2y": 730, "Last 5y": 1825}
                        n = nmap.get(focus, None)
                        if n and len(s) > n:
                            s = s.tail(n)

                    # Indicators from current slice
                    ma50s  = s.rolling(50,  min_periods=1).mean()
                    ma200s = s.rolling(200, min_periods=1).mean()
                    rsi14s = rsi_wilder(s.to_frame(), 14).iloc[:, 0]  # series
                    ema_fast = s.ewm(span=12, adjust=False).mean()
                    ema_slow = s.ewm(span=26, adjust=False).mean()
                    macd_line_s   = ema_fast - ema_slow
                    macd_signal_s = macd_line_s.ewm(span=9, adjust=False).mean()
                    macd_hist_s   = (macd_line_s - macd_signal_s)

                    # Cross markers (within current window)
                    crosses_local = (ma50s > ma200s) & (ma50s.shift(1) <= ma200s.shift(1))
                    golden_dates = crosses_local[crosses_local].index
                    crosses_local_dn = (ma50s < ma200s) & (ma50s.shift(1) >= ma200s.shift(1))
                    death_dates = crosses_local_dn[crosses_local_dn].index

                    # OHLCV for candles & volume (trim to focus)
                    ohlcv = fetch_ohlcv(sel, p)
                    if not ohlcv.empty and focus != "Full":
                        ohlcv = ohlcv.loc[s.index.min(): s.index.max()]
                    have_candles = (not ohlcv.empty) and {"Open","High","Low","Close"}.issubset(ohlcv.columns)
                    have_volume  = (not ohlcv.empty) and ("Volume" in ohlcv.columns)

                    # 3) Build figure (4 rows: Price, Volume, RSI, MACD)
                    fig = make_subplots(
                        rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.15, 0.15, 0.15], vertical_spacing=0.03
                    )

                    # --- Row 1: Price ---
                    if view_mode == "% change (rebased)":
                        base = s.iloc[0]
                        y = (s / base - 1.0) * 100.0
                        ytitle = "Return (%)"
                        fig.add_trace(go.Scatter(x=y.index, y=y, name="% change",
                                                 mode="lines", connectgaps=True), row=1, col=1)
                        # Rebased MAs
                        m50  = (ma50s / base - 1.0) * 100.0
                        m200 = (ma200s / base - 1.0) * 100.0
                        fig.add_trace(go.Scatter(x=m50.index,  y=m50,  name="MA50",
                                                 mode="lines", connectgaps=True), row=1, col=1)
                        fig.add_trace(go.Scatter(x=m200.index, y=m200, name="MA200",
                                                 mode="lines", connectgaps=True), row=1, col=1)
                        # Cross markers (use s for y even in rebased → they still show relative locations)
                        if len(golden_dates):
                            fig.add_trace(go.Scatter(
                                x=golden_dates, y=s.reindex(golden_dates),
                                mode="markers", name="Golden Cross",
                                marker_symbol="triangle-up", marker_color="limegreen", marker_size=10
                            ), row=1, col=1)
                        if len(death_dates):
                            fig.add_trace(go.Scatter(
                                x=death_dates, y=s.reindex(death_dates),
                                mode="markers", name="Death Cross",
                                marker_symbol="triangle-down", marker_color="crimson", marker_size=10
                            ), row=1, col=1)
                        ytype = "linear"
                    else:
                        if have_candles:
                            fig.add_trace(go.Candlestick(
                                x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
                                low=ohlcv["Low"], close=ohlcv["Close"], name="OHLC"
                            ), row=1, col=1)
                            price_for_markers = ohlcv["Close"]
                        else:
                            fig.add_trace(go.Scatter(x=s.index, y=s, name="Adj Close",
                                                     mode="lines", connectgaps=True), row=1, col=1)
                            price_for_markers = s
                        fig.add_trace(go.Scatter(x=ma50s.index,  y=ma50s,  name="MA50",
                                                 mode="lines", connectgaps=True), row=1, col=1)
                        fig.add_trace(go.Scatter(x=ma200s.index, y=ma200s, name="MA200",
                                                 mode="lines", connectgaps=True), row=1, col=1)
                        # Cross markers
                        if len(golden_dates):
                            fig.add_trace(go.Scatter(
                                x=golden_dates, y=price_for_markers.reindex(golden_dates),
                                mode="markers", name="Golden Cross",
                                marker_symbol="triangle-up", marker_color="limegreen", marker_size=10
                            ), row=1, col=1)
                        if len(death_dates):
                            fig.add_trace(go.Scatter(
                                x=death_dates, y=price_for_markers.reindex(death_dates),
                                mode="markers", name="Death Cross",
                                marker_symbol="triangle-down", marker_color="crimson", marker_size=10
                            ), row=1, col=1)
                        ytitle = "Price"
                        ytype  = "log" if view_mode == "Log price" else "linear"

                    # --- Row 2: Volume ---
                    if have_volume:
                        v = ohlcv["Volume"]
                        # Color bars green if close >= open; else red
                        up = (ohlcv["Close"] >= ohlcv["Open"]).reindex(v.index).fillna(False)
                        vol_colors = np.where(up, "rgba(0,150,0,0.6)", "rgba(200,0,0,0.6)")
                        fig.add_trace(go.Bar(x=v.index, y=v, name="Volume", marker_color=vol_colors), row=2, col=1)
                    else:
                        # no volume available → show a faint baseline
                        fig.add_trace(go.Bar(x=s.index, y=np.zeros(len(s)), name="Volume"), row=2, col=1)

                    # --- Row 3: RSI ---
                    fig.add_trace(go.Scatter(x=rsi14s.index, y=rsi14s, name="RSI(14)",
                                             mode="lines", line=dict(color="#888")), row=3, col=1)
                    # Shaded 30–70 zone + lines
                    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="rgba(200,200,200,0.15)", row=3, col=1)
                    fig.add_hline(y=30, line_width=1, line_color="crimson", row=3, col=1)
                    fig.add_hline(y=70, line_width=1, line_color="seagreen", row=3, col=1)

                    # --- Row 4: MACD with safe, colored histogram (NaN-proof) ---
                    hist_plot = macd_hist_s.fillna(0.0)
                    colors = np.where(hist_plot >= 0, "rgba(0,160,0,0.6)", "rgba(200,0,0,0.6)")
                    fig.add_trace(go.Bar(x=hist_plot.index, y=hist_plot,
                                         name="Hist", marker_color=colors), row=4, col=1)
                    fig.add_trace(go.Scatter(x=macd_line_s.index,   y=macd_line_s,
                                             name="MACD",  mode="lines", connectgaps=True), row=4, col=1)
                    fig.add_trace(go.Scatter(x=macd_signal_s.index, y=macd_signal_s,
                                             name="Signal",mode="lines", connectgaps=True), row=4, col=1)
                    fig.add_hline(y=0, line_width=1, line_color="gray", row=4, col=1)

                    # MACD autoscale safely (Series or DataFrame)
                    vals = macd_hist_s
                    if isinstance(vals, pd.DataFrame):
                        vals = vals.stack()
                    vals = np.asarray(vals, dtype="float64")
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        max_abs = float(np.nanmax(np.abs(vals)))
                        if max_abs == 0:
                            max_abs = 1.0
                        fig.update_yaxes(range=[-1.2 * max_abs, 1.2 * max_abs], row=4, col=1)

                    finish_layout(fig, ytype=("percent" if view_mode == "% change (rebased)" else ytype),
                                  ytitle_price=ytitle)
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
