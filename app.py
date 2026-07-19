import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
from datetime import datetime, timedelta
import urllib.parse

st.set_page_config(page_title="Market Anomaly Detector", layout="wide", page_icon="📊")

st.markdown("""
<style>
.main {background-color: #0e1117;}
.stMetric {background-color: #1c1f26; padding: 15px; border-radius: 10px;}
.anomaly-card {
    background: linear-gradient(135deg, #1c1f26 0%, #23272f 100%);
    border-left: 4px solid #E4572E;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 6px;
    cursor: pointer;
}
.anomaly-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 16px;
    font-weight: 600;
    color: #f0f0f0;
}
.anomaly-badge {
    background-color: #E4572E;
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
}
.anomaly-meta {
    color: #9aa0aa;
    font-size: 13px;
    margin-top: 4px;
}
.news-pill {
    background-color: #2a2f3a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Market Anomaly & Crisis Detector")
st.caption("Live statistical monitoring of market stress using rolling z-scores across major asset classes")

HISTORICAL_EVENTS = {
    "2008-09-15": "Lehman Brothers files for bankruptcy, triggering global financial crisis.",
    "2008-10-13": "Global stock markets rally after coordinated bank bailout announcements.",
    "2008-11-20": "S&P 500 hits multi-year lows amid deepening recession fears.",
    "2009-03-09": "S&P 500 bottoms out during the Global Financial Crisis.",
    "2010-05-06": "Flash Crash: Dow Jones drops ~1000 points in minutes.",
    "2011-08-08": "US credit rating downgraded by S&P, sparking global selloff.",
    "2015-08-24": "China devaluation fears trigger global market selloff (\'Black Monday\').",
    "2020-02-24": "COVID-19 fears trigger global market selloff as cases spread outside China.",
    "2020-03-16": "Circuit breakers halt trading as COVID-19 panic selling accelerates.",
    "2022-06-13": "S&P 500 enters bear market amid rate hike and inflation fears.",
}

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=86400, show_spinner=False)
def get_news_for_date(date_str, days_window=1):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        after = d.strftime("%Y-%m-%d")
        before = (d + timedelta(days=days_window)).strftime("%Y-%m-%d")
        query = f"stock market after:{after} before:{before}"
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:5]
        return [{"title": i.find("title").text, "link": i.find("link").text, "pubDate": i.find("pubDate").text} for i in items]
    except Exception:
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
    plot_df = df.tail(504).resample("W").last()
else:
    plot_df = df.resample("W").last()

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines',
                          name='Anomaly Score', line=dict(color='#E4572E', width=2),
                          fill='tozeroy', fillcolor='rgba(228,87,46,0.15)'))
fig.add_hline(y=latest['Threshold'], line_dash='dash', line_color='gray', annotation_text='Threshold')
flagged_plot = plot_df[plot_df['Flagged']]
fig.add_trace(go.Scatter(x=flagged_plot.index, y=flagged_plot['Anomaly_Score'], mode='markers',
                          name='Flagged Day', marker=dict(color='red', size=6)))
fig.update_layout(title="Market Anomaly Score Over Time", plot_bgcolor='white',
                   legend=dict(orientation='h', y=1.1))
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Anomaly Score")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Flagged Anomaly Days")
st.caption("Click any anomaly below to load real news headlines from that exact date.")

recent_flags = df[df['Flagged']].tail(40).sort_index(ascending=False)

search_term = st.text_input("🔎 Filter anomalies by date (YYYY-MM-DD) or leave blank to see all", "")
if search_term:
    recent_flags = recent_flags[recent_flags.index.strftime("%Y-%m-%d").str.contains(search_term)]

for date_idx, row in recent_flags.iterrows():
    date_str = date_idx.strftime("%Y-%m-%d")
    date_pretty = date_idx.strftime("%B %d, %Y")
    days_ago = (datetime.now() - date_idx.to_pydatetime().replace(tzinfo=None)).days
    severity = "🔴 Severe" if row['Anomaly_Score'] > row['Threshold']*1.3 else "🟠 Moderate"

    st.markdown(f"""
    <div class="anomaly-card">
        <div class="anomaly-header">
            <span>{date_pretty} &nbsp; <span class="anomaly-badge">{severity}</span></span>
            <span>Score: {row['Anomaly_Score']:.2f}</span>
        </div>
        <div class="anomaly-meta">S&P 500: {row['S&P500']:.1f} &nbsp;|&nbsp; VIX: {row['VIX']:.1f} &nbsp;|&nbsp; {days_ago} days ago</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📰 View news from {date_pretty}"):
        if date_str in HISTORICAL_EVENTS:
            st.info(f"📌 {HISTORICAL_EVENTS[date_str]}")
        with st.spinner("Fetching headlines..."):
            news = get_news_for_date(date_str)
        if news:
            for article in news:
                st.markdown(f"""
                <div class="news-pill">
                <b>{article['title']}</b><br>
                <span style="color:#9aa0aa;font-size:12px;">{article['pubDate']}</span><br>
                <a href="{article['link']}" target="_blank">Read more →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No headlines found via automated search for this date.")

    st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

st.subheader("Raw Data Explorer")
st.dataframe(df.tail(100), use_container_width=True)

st.caption("Data source: Yahoo Finance + Google News | Model: Rolling 63-day z-score composite anomaly detection")
