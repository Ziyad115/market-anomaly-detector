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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0a0d12 0%, #0d1117 100%);
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 2rem !important;
    max-width: 1200px;
}

.top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 4px;
    margin-bottom: 8px;
}
.brand-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.8px;
    background: linear-gradient(100deg, #ffb08a 0%, #E4572E 55%, #c23f1e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.brand-sub {
    color: #6b7280;
    font-size: 14px;
    font-weight: 500;
    margin-top: 2px;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #9ca3af;
    letter-spacing: 0.3px;
}
.dot-live {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.4; }
    100% { opacity: 1; }
}

[data-testid="stMetric"] {
    background: linear-gradient(160deg, #12151b 0%, #171b22 100%);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 20px 20px 16px 20px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    min-height: 96px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(228,87,46,0.35);
}
div[data-testid="column"] { display: flex; }
div[data-testid="column"] > div { width: 100%; }

[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 26px !important;
    white-space: nowrap;
    color: #f5f6f7 !important;
    font-family: 'JetBrains Mono', monospace;
}

.section-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 700;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 32px 0 14px 0;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.12), transparent);
}

.anomaly-card {
    background: linear-gradient(160deg, #12151b 0%, #181c24 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #E4572E;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 8px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}
.anomaly-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 15px;
    font-weight: 700;
    color: #f3f4f6;
    letter-spacing: -0.1px;
}
.anomaly-score-val {
    font-family: 'JetBrains Mono', monospace;
    color: #ff8a5c;
    font-weight: 700;
}
.anomaly-badge {
    background: linear-gradient(90deg, #E4572E, #ff7a45);
    color: white;
    padding: 4px 11px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-left: 10px;
}
.anomaly-meta {
    color: #6b7280;
    font-size: 12.5px;
    margin-top: 6px;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}
.news-pill {
    background: linear-gradient(150deg, #171a21 0%, #1d2129 100%);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
}
.news-pill b { color: #e5e7eb; font-size: 14px; }
.news-pill a { color: #60a5fa !important; font-size: 13px; text-decoration: none; }
.news-pill a:hover { text-decoration: underline; }

.context-box {
    background: linear-gradient(150deg, #12151b 0%, #171b22 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    color: #9ca3af;
    font-size: 13.5px;
    line-height: 1.7;
}
.context-box b { color: #e5e7eb; }

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.015);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSelectbox"] label, .stTextInput label {
    color: #9ca3af !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

hr, [data-testid="stDivider"] {
    border-color: rgba(255,255,255,0.07) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-header">
    <div>
        <div class="brand-title">Market Anomaly &amp; Crisis Detector</div>
        <div class="brand-sub">Live statistical monitoring of market stress across five major asset classes</div>
    </div>
    <div class="status-chip"><span class="dot-live"></span> LIVE DATA</div>
</div>
""", unsafe_allow_html=True)

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
col4.metric("Last Updated", datetime.now().strftime("%d %b, %H:%M"))

st.divider()

view = st.selectbox("Select time range", ["Last 6 Months", "Last 2 Years", "Full History (2005-Present)"])
if view == "Last 6 Months":
    plot_df = df.tail(126)
elif view == "Last 2 Years":
    plot_df = df.tail(504).resample("W").last()
else:
    plot_df = df.resample("W").last()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines',
    name='Anomaly Score',
    line=dict(color='#ff7a45', width=2.5, shape='spline', smoothing=0.3),
    fill='tozeroy', fillcolor='rgba(228,87,46,0.18)',
    hovertemplate='%{x|%b %d, %Y}<br>Score: %{y:.2f}<extra></extra>'
))

fig.add_hline(
    y=latest['Threshold'], line_dash='dot', line_color='rgba(255,255,255,0.35)', line_width=1.5,
    annotation_text='Anomaly Threshold', annotation_font_color='#9aa0aa',
    annotation_font_size=12, annotation_position='top left'
)

flagged_plot = plot_df[plot_df['Flagged']]
fig.add_trace(go.Scatter(
    x=flagged_plot.index, y=flagged_plot['Anomaly_Score'], mode='markers',
    name='Flagged Day',
    marker=dict(color='#ff3b30', size=8, line=dict(color='#0e1117', width=1.5), symbol='circle'),
    hovertemplate='⚠️ %{x|%b %d, %Y}<br>Score: %{y:.2f}<extra></extra>'
))

fig.update_layout(
    title=dict(text="Market Anomaly Score Over Time", font=dict(size=18, color='#f0f0f0', family='Inter')),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c8ccd4', family='Inter'),
    legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center', bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=10, r=10, t=60, b=10),
    hovermode='x unified'
)
fig.update_xaxes(title_text="Date", showgrid=False, showline=True, linecolor='rgba(255,255,255,0.1)', zeroline=False)
fig.update_yaxes(title_text="Anomaly Score", showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
fig.update_traces(cliponaxis=False)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown('<div class="section-label">🔍&nbsp; Flagged Anomaly Days</div>', unsafe_allow_html=True)
st.caption("Select a year (and optionally a month) to browse anomalies, then click any card to load real news from that exact date.")

all_flags = df[df['Flagged']].sort_index(ascending=False)

available_years = sorted(all_flags.index.year.unique(), reverse=True)
year_options = ["All Years"] + [str(y) for y in available_years]

col_a, col_b = st.columns(2)
with col_a:
    selected_year = st.selectbox("📅 Year", year_options)
with col_b:
    month_options = ["All Months"]
    if selected_year != "All Years":
        months_in_year = sorted(all_flags[all_flags.index.year == int(selected_year)].index.month.unique())
        month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        month_options += [month_names[m-1] for m in months_in_year]
    selected_month = st.selectbox("🗓️ Month (optional)", month_options)

recent_flags = all_flags.copy()
if selected_year != "All Years":
    recent_flags = recent_flags[recent_flags.index.year == int(selected_year)]
if selected_month != "All Months":
    month_num = ["January","February","March","April","May","June","July","August","September","October","November","December"].index(selected_month) + 1
    recent_flags = recent_flags[recent_flags.index.month == month_num]

st.caption(f"Showing {len(recent_flags)} anomaly day(s)" + (f" in {selected_year}" if selected_year != "All Years" else " across all history"))

if len(recent_flags) > 60:
    recent_flags = recent_flags.head(60)
    st.info("Showing most recent 60 matches. Narrow down by month for more precision.")

for date_idx, row in recent_flags.iterrows():
    date_str = date_idx.strftime("%Y-%m-%d")
    date_pretty = date_idx.strftime("%B %d, %Y")
    days_ago = (datetime.now() - date_idx.to_pydatetime().replace(tzinfo=None)).days
    severity = "🔴 Severe" if row['Anomaly_Score'] > row['Threshold']*1.3 else "🟠 Moderate"

    st.markdown(f"""
    <div class="anomaly-card">
        <div class="anomaly-header">
            <span>{date_pretty}<span class="anomaly-badge">{severity}</span></span>
            <span class="anomaly-score-val">{row['Anomaly_Score']:.2f}</span>
        </div>
        <div class="anomaly-meta">S&amp;P 500 {row['S&P500']:.1f}  ·  VIX {row['VIX']:.1f}  ·  {days_ago}d ago</div>
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

st.markdown('<div class="section-label">📁&nbsp; Raw Data Explorer</div>', unsafe_allow_html=True)
st.markdown("""
<div class="context-box">
<b>What am I looking at?</b><br>
This table shows the last 100 trading days of raw and calculated data feeding the model above.
Each asset (S&P 500, Gold, Oil, USD Index, VIX) has its <b>daily return</b>, <b>63-day rolling mean/std</b>,
and resulting <b>z-score</b> — the number of standard deviations that day's move was from its recent norm.
The final <b>Anomaly_Score</b> combines all z-scores into one composite reading, and <b>Flagged</b> marks
days where that score crossed the statistical threshold (mean + 2 standard deviations).
</div>
""", unsafe_allow_html=True)
st.dataframe(df.tail(100), use_container_width=True)

st.caption("Data source: Yahoo Finance + Google News | Model: Rolling 63-day z-score composite anomaly detection")
