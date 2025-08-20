# StockPeers Screener (v2: Fundamentals + Sparkline)

A password-protected Streamlit app that screens the **S&P 500** by technicals (RSI/MA rules + recent Golden/Death cross) **and optional fundamental filters** (Market Cap, P/E, Dividend Yield, Beta). Exports results to **CSV or Excel** and shows an on-demand sparkline.

## Deploy (Streamlit Cloud)
1. Create a GitHub repo (e.g., `stockpeers-screener`) and upload:
   - `scanner.py`
   - `requirements.txt`
   - `.streamlit/secrets.template.toml`
2. In Streamlit Cloud → App → **Settings → Secrets**, paste:
   ```toml
   [password]
   app_password = "StockPeers2024!"
   ```
3. New App → Choose repo/branch → Main file path: `scanner.py` → **Deploy**.

Your app will be live at a URL like: `https://stockpeers-screener.streamlit.app`

## Local run (optional)
```bash
pip install -r requirements.txt
streamlit run scanner.py
```
