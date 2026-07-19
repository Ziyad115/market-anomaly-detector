mport streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Market Anomaly Detector", layout="wide", page_icon="📊")

st.markdown("""
<style>
.main {background-color: #0e1117;}
.stMetric {background-color: #1c1f26; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Market Anomaly & Crisis Detector")
st.caption("Live statistical monitoring of market stress using rolling z-scores across major asset classes")

NEWSDATA_API_KEY = st.secrets.get("NEWSDATA_API_KEY", "")

HISTORICAL_EVENTS = {
    "2008-09-15": "Lehman Brothers files for bankruptcy, triggering global financial crisis.",
    "2008-10-13": "Global stock markets rally after coordinated bank bailout announcements.",
    "2008-11-20": "S&P 500 hits multi-year lows amid deepening recession fears.",
    "2009-03-09": "S&P 500 bottoms out during the Global Financial Crisis.",
    "2010-05-06": "Flash Crash: Dow Jones drops ~1000 points in minutes.",
    "2011-08-08": "US credit rating downgraded by S&P, sparking global selloff.",
    "2015-08-24": "China devaluation fears trigger global market selloff ('Black Monday').",
}

@st.cache_data(ttl=3600)
def load_data():
    tickers = {
        'S&P500': '^GSPC', 'VIX': '^VIX', 'Gold': 'GC=F',
        'Oil_WTI': 'CL=F', 'USD_Index': 'DX-Y.NYB'
    }
    data = {}
    for name, t in tickers.items():
        df = yf.download(t, start='2005-01-01', progress=False)
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        data[name] = close
    prices = pd.DataFrame(data).dropna()
    return prices

def compute_anomaly(prices, window=63):
    df = prices.copy()
    assets = ['S&P500', 'Gold', 'Oil_WTI', 'USD_Index']
    for col in assets:
        df[f'{col}_Return'] = df[col].pct_change()
        df[f'{col}_RollMean'] = df[f'{col}_Return'].rolling(window).mean()
        df[f'{col}_RollStd'] = df[f'{col}_Return'].rolling(window).std()
        df[f'{col}_Zscore'] = (df[f'{col}_Return'] - df[f'{col}_RollMean']) / df[f'{col}_RollStd']
    df['VIX_RollMean'] = df['VIX'].rolling(window).mean()
    df['VIX_RollStd'] = df['VIX'].rolling(window).std()
    df['VIX_Zscore'] = (df['VIX'] - df['VIX_RollMean']) / df['VIX_RollStd']
    zcols = [f'{a}_Zscore' for a in assets]
    df['Avg_Abs_Zscore'] = df[zcols].abs().mean(axis=1)
    df['Anomaly_Score'] = df['Avg_Abs_Zscore'] + df['VIX_Zscore'].abs()/2
    df['Threshold'] = df['Anomaly_Score'].mean() + 2*df['Anomaly_Score'].std()
    df['Flagged'] = df['Anomaly_Score'] > df['Threshold']
    return df

def get_news_for_date(date_str, api_key):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    if date_obj < datetime(2016, 1, 1):
        return None
    if not api_key:
        return "no_key"
    try:
        url = "https://newsdata.io/api/1/archive"
        params = {
            "apikey": api_key,
            "q": "stock market",
            "from_date": date_str,
            "to_date": date_str,
            "language": "en"
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return data.get("results", [])[:5]
    except Exception as e:
        return []

with st.spinner("Loading live market data..."):
    prices = load_data()
    df = compute_anomaly(prices)

latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Anomaly Score", f"{latest['Anomaly_Score']:.2f}")
col2.metric("Threshold", f"{latest['Threshold']:.2f}")
status = "🔴 ANOMALY" if latest['Flagged'] else "🟢 NORMAL"
col3.metric("Current Status", status)
col4.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))

st.divider()

view = st.selectbox("Select time range", ["Last 6 Months", "Last 2 Years", "Full History (2005-Present)"])
if view == "Last 6 Months":
    plot_df = df.tail(126)
elif view == "Last 2 Years":
    plot_df = df.tail(504)
else:
    plot_df = df

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines',
                          name='Anomaly Score', line=dict(color='#E4572E', width=2),
                          fill='tozeroy', fillcolor='rgba(228,87,46,0.15)'))
fig.add_hline(y=latest['Threshold'], line_dash='dash', line_color='gray', annotation_text='Threshold')
flagged = plot_df[plot_df['Flagged']]
fig.add_trace(go.Scatter(x=flagged.index, y=flagged['Anomaly_Score'], mode='markers',
                          name='Flagged Day', marker=dict(color='red', size=6)))
fig.update_layout(title="Market Anomaly Score Over Time", plot_bgcolor='white',
                   legend=dict(orientation='h', y=1.1))
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Anomaly Score")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Flagged Anomaly Days — With News Context")
recent_flags = df[df['Flagged']].tail(15)

for date_idx, row in recent_flags[::-1].iterrows():
    date_str = date_idx.strftime("%Y-%m-%d")
    with st.expander(f"{date_str} — Score: {row['Anomaly_Score']:.2f} | S&P500: {row['S&P500']:.1f} | VIX: {row['VIX']:.1f}"):
        if date_str in HISTORICAL_EVENTS:
            st.info(f"📌 Known historical event: {HISTORICAL_EVENTS[date_str]}")
        else:
            news = get_news_for_date(date_str, NEWSDATA_API_KEY)
            if news == "no_key":
                st.warning("News API key not configured yet.")
            elif news is None:
                st.info("This date is before 2016 — no live news archive available. Check historical records manually.")
            elif len(news) == 0:
                st.info("No news articles found for this date.")
            else:
                for article in news:
                    st.markdown(f"**[{article.get('title','No title')}]({article.get('link','#')})**")
                    st.caption(article.get('pubDate',''))

st.subheader("Raw Data Explorer")
st.dataframe(df.tail(100), use_container_width=True)

st.caption("Data source: Yahoo Finance + NewsData.io | Model: Rolling 63-day z-score composite anomaly detection")
