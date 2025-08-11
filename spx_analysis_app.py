import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import time

# -------------------- UI SETUP --------------------
st.set_page_config(page_title="S&P 500 Outliers", layout="wide")
st.title("📊 S&P 500 — Outliers, Sector & Industry Performance")
st.caption("Prices: Yahoo Finance • Constituents: Wikipedia • Note: Yahoo end date is exclusive (+1 day).")

# -------------------- HELPERS (same logic as notebook, with robust guards) --------------------
def get_sp500_metadata():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    df = df.rename(columns={'Symbol': 'Ticker', 'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Industry Group'})
    df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)  # BRK.B -> BRK-B
    return df[['Ticker', 'Security', 'Sector', 'Industry Group']]

def _trading_days(start_date, end_date_exclusive):
    # tz-naive, normalized trading-day index; use NYSE if available, else business days
    start_dt = pd.to_datetime(start_date)
    end_exc_dt = pd.to_datetime(end_date_exclusive)
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')
        sched = nyse.schedule(start_date=start_dt, end_date=end_exc_dt - pd.Timedelta(days=1))
        idx = pd.DatetimeIndex(sched.index)
    except Exception:
        idx = pd.bdate_range(start=start_dt, end=end_exc_dt - pd.Timedelta(days=1))
    idx = pd.to_datetime(idx).tz_localize(None).normalize()
    return idx

def get_price_data_with_trading_days(tickers, start_date, end_date, min_data_fraction=0.90, sleep_sec=0.0):
    """
    Per-ticker fetch (stable on cloud). Ensures each dict entry is a 1-D Series
    aligned to a common trading-day index. Skips tickers with poor coverage.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    end_excl = end_dt + pd.Timedelta(days=1)  # yfinance 'end' is exclusive

    td_index = _trading_days(start_dt, end_excl)
    if len(td_index) == 0:
        return pd.DataFrame()

    out = {}
    for t in tickers:
        d = yf.download(
            t, start=start_dt, end=end_excl,
            auto_adjust=True, progress=False,
            threads=False, interval="1d", prepost=False
        )
        if d is None or d.empty:
            continue

        # Pick price column (prefer Adj Close)
        if 'Adj Close' in d.columns:
            sub = d['Adj Close']
        elif 'Close' in d.columns:
            sub = d['Close']
        else:
            # Squeeze any weird shape
            try:
                sub = d.squeeze()
                if isinstance(sub, pd.DataFrame):
                    sub = sub.iloc[:, 0]
            except Exception:
                continue

        if not isinstance(sub, pd.Series):
            try:
                sub = pd.Series(sub)
            except Exception:
                continue

        # Normalize index to tz-naive dates & deduplicate
        idx = pd.to_datetime(sub.index).tz_localize(None).normalize()
        sub.index = idx
        sub = sub[~sub.index.duplicated(keep='last')]

        # Align to common trading-day index
        aligned = sub.reindex(td_index)

        # Require coverage
        if aligned.notna().sum() / len(td_index) >= min_data_fraction:
            out[t] = aligned.astype('float64')

        if sleep_sec:
            time.sleep(sleep_sec)

    if not out:
        return pd.DataFrame()

    return pd.DataFrame(out, index=td_index)

def calculate_sum_daily_return(prices):
    """Sum of daily % changes (matches Excel 'sum of 1-day returns' style)."""
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

# -------------------- UI CONTROLS --------------------
today = date.today()
ytd_start = date(today.year, 1, 1)

mode = st.radio("Mode", ["Year-to-date", "Custom range", "Single day"], index=0, horizontal=True)

if mode == "Year-to-date":
    start_date = ytd_start
    end_date = today
elif mode == "Custom range":
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=ytd_start)
    with c2:
        end_date = st.date_input("End date", value= today)
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()
else:
    # Single day: fetch a short window ending that day so pct_change has a prior day
    single_day = st.date_input("Date", value=today)
    start_date = single_day - timedelta(days=7)
    end_date = single_day

top_n = st.slider("Top/Bottom N", 5, 25, 10)

st.divider()

# -------------------- RUN --------------------
with st.status("Fetching constituents & prices…", expanded=False):
    meta = get_sp500_metadata()
    tickers = meta["Ticker"].tolist()
    prices = get_price_data_with_trading_days(tickers, str(start_date), str(end_date), min_data_fraction=0.90, sleep_sec=0.0)

with st.expander("🔎 Diagnostics"):
    st.write("Window:", str(start_date), "→", str(end_date), "(Yahoo sends end+1)")
    st.write("Prices shape:", prices.shape)

if prices.empty:
    st.error("No price data returned (date window too narrow, market holidays, or Yahoo hiccup). Try a different range.")
    st.stop()

# Keep only tickers that have data
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
    st.dataframe(top[['Ticker','Security','Sum Daily Return %']].reset_index(drop=True), use_container_width=True)
with c2:
    st.subheader(f"📉 Top {top_n} Losers")
    st.dataframe(bottom[['Ticker','Security','Sum Daily Return %']].reset_index(drop=True), use_container_width=True)

st.subheader("📊 Sector Performance (Average of ticker sum of daily %)")
st.dataframe(sec.to_frame('Sum Daily Return %'), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("🏭 Industry Groups — Top 10")
    st.dataframe(grp.head(10).to_frame('Sum Daily Return %'), use_container_width=True)
with c4:
    st.subheader("🏭 Industry Groups — Bottom 10")
    st.dataframe(grp.tail(10).to_frame('Sum Daily Return %'), use_container_width=True)

st.caption("If a few tickers are missing, it’s due to incomplete Yahoo coverage in the chosen window. The analysis still runs on available symbols.")
