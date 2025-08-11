import streamlit as st
import pandas as pd
import yfinance as yf
import time
from datetime import date, timedelta

# Optional NYSE calendar (falls back to weekdays if not available)
try:
    import pandas_market_calendars as mcal
    HAS_MCAL = True
except Exception:
    HAS_MCAL = False

st.set_page_config(page_title="S&P 500 Outliers", layout="wide")
st.title("📊 S&P 500 — Outliers, Sector & Industry Performance")

# ---------------- Your working logic, preserved ----------------
@st.cache_data(ttl=12*60*60)
def get_sp500_metadata():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    df = df.rename(columns={'Symbol': 'Ticker', 'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Industry Group'})
    df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)  # BRK.B -> BRK-B
    return df[['Ticker', 'Security', 'Sector', 'Industry Group']]

def _trading_days(start_date, end_date_exclusive):
    if HAS_MCAL:
        nyse = mcal.get_calendar('NYSE')
        sched = nyse.schedule(start_date=start_date, end_date=end_date_exclusive - pd.Timedelta(days=1))
        return pd.DatetimeIndex(sched.index, name="Date")
    # fallback to business days
    return pd.bdate_range(start=start_date, end=end_date_exclusive - pd.Timedelta(days=1), name="Date")

@st.cache_data(ttl=2*60*60, show_spinner=False)
def get_price_data_with_trading_days(tickers, start_date, end_date, min_data_fraction=0.95, sleep_sec=0.02):
    # yfinance end is EXCLUSIVE → add +1 day
    start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_excl = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    td = set(_trading_days(pd.to_datetime(start_str), pd.to_datetime(end_excl)))
    if not td:
        return pd.DataFrame()

    adj_close = {}
    for t in tickers:
        d = yf.download(t, start=start_str, end=end_excl, auto_adjust=True, progress=False, threads=True)
        if d.empty:
            continue
        # choose column
        if 'Adj Close' in d.columns:
            sub = d['Adj Close']
        elif 'Close' in d.columns:
            sub = d['Close']
        else:
            continue
        overlap = td & set(sub.index)
        if len(overlap) == 0:
            continue
        # require enough coverage (like your Jupyter version)
        if len(overlap) / len(td) >= min_data_fraction:
            adj_close[t] = sub.loc[sorted(overlap)]
        time.sleep(sleep_sec)
    if not adj_close:
        return pd.DataFrame()
    return pd.DataFrame(adj_close)

def calculate_sum_daily_return(prices):
    if prices.empty or prices.shape[0] < 2:
        return pd.Series(dtype=float)
    return (prices.pct_change().sum() * 100).round(4)

def merge_metadata_returns(metadata, returns):
    merged = metadata.copy()
    merged['Sum Daily Return %'] = merged['Ticker'].map(returns)
    return merged.dropna(subset=['Sum Daily Return %'])

def get_top_bottom(merged, n=10):
    s = merged.sort_values('Sum Daily Return %', ascending=False)
    return s.head(n), s.tail(n)

def sector_performance(merged):
    return merged.groupby('Sector')['Sum Daily Return %'].mean().sort_values(ascending=False).round(4)

def industry_group_performance(merged):
    return merged.groupby('Industry Group')['Sum Daily Return %'].mean().sort_values(ascending=False).round(4)
# ----------------------------------------------------------------

# -------------------------- UI --------------------------
st.write("Default view is **Year-To-Date (YTD)**. You can also select a **custom range** or a **single day**.")

today = date.today()
ytd_start = date(today.year, 1, 1)

mode = st.radio(
    "Mode",
    ["Year-to-date", "Custom range", "Single day"],
    index=0,
    horizontal=True
)

if mode == "Year-to-date":
    start_date = ytd_start
    end_date = today
elif mode == "Custom range":
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=ytd_start)
    with c2:
        end_date = st.date_input("End date", value=today)
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()
else:  # Single day
    single_day = st.date_input("Date", value=today)
    start_date = single_day
    end_date = single_day

top_n = st.slider("Top/Bottom N", 5, 25, 10)

with st.status("Fetching constituents & prices…", expanded=False):
    meta = get_sp500_metadata()
    tickers = meta["Ticker"].tolist()
    prices = get_price_data_with_trading_days(tickers, str(start_date), str(end_date), min_data_fraction=0.90)

# Diagnostics
with st.expander("🔎 Diagnostics"):
    st.write("Window:", str(start_date), "→", str(end_date), "(end+1 sent to Yahoo)")
    st.write("Prices shape:", prices.shape)

if prices.empty:
    st.error("No price data returned. Try a different date range, then use the ‘Rerun’ button in the top-right.")
    st.stop()

# Keep only tickers with data
meta = meta[meta['Ticker'].isin(prices.columns)]

rets = calculate_sum_daily_return(prices)
merged = merge_metadata_returns(meta, rets)

if merged.empty:
    st.error("No tickers with valid returns in this window.")
    st.stop()

top, bottom = get_top_bottom(merged, top_n)
sec = sector_performance(merged)
grp = industry_group_performance(merged)

c1, c2 = st.columns(2)
with c1:
    st.subheader(f"🏆 Top {top_n} Gainers")
    st.dataframe(top[['Ticker','Security','Sum Daily Return %']], use_container_width=True, hide_index=True)
with c2:
    st.subheader(f"📉 Top {top_n} Losers")
    st.dataframe(bottom[['Ticker','Security','Sum Daily Return %']], use_container_width=True, hide_index=True)

st.subheader("📊 Sector Performance (Avg of ticker sum daily %)")
st.dataframe(sec.to_frame('Sum Daily Return %'), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("🏭 Industry Groups — Top 10")
    st.dataframe(grp.head(10).to_frame('Sum Daily Return %'), use_container_width=True)
with c4:
    st.subheader("🏭 Industry Groups — Bottom 10")
    st.dataframe(grp.tail(10).to_frame('Sum Daily Return %'), use_container_width=True)

st.caption("Notes: Yahoo end date is exclusive (we query end+1). S&P 500 list from Wikipedia; prices from Yahoo Finance (auto-adjusted).")
