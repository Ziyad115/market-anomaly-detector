import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
from datetime import datetime, timedelta
import urllib.parse

st.set_page_config(page_title="Market Anomaly Detector", layout="wide", page_icon="📈")

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN SYSTEM
#  Dark-slate base · glass surfaces · cyan/indigo neon · red/amber for stress
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root{
    --bg-0:#070A0F;
    --bg-1:#0B111A;
    --glass:rgba(19,26,37,0.60);
    --glass-hi:rgba(26,34,48,0.60);
    --stroke:rgba(148,163,184,0.12);
    --stroke-strong:rgba(148,163,184,0.24);
    --txt:#E8EEF5;
    --txt-dim:#8C97A8;
    --txt-faint:#5A6576;
    --neon:#22D3EE;      /* cyan  */
    --neon-2:#818CF8;    /* indigo*/
    --pos:#34D399;       /* green */
    --warn:#FBBF24;      /* amber */
    --danger:#FB4B57;    /* red   */
    --mono:'JetBrains Mono', ui-monospace, monospace;
}

html, body, [class*="css"]{
    font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp{
    background:
        radial-gradient(1100px 560px at 12% -12%, rgba(34,211,238,0.10), transparent 60%),
        radial-gradient(1000px 520px at 102% -6%, rgba(129,140,248,0.09), transparent 58%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
    background-attachment:fixed;
    color:var(--txt);
}
#MainMenu, footer, header {visibility:hidden;}
.block-container{ padding-top:1.6rem !important; padding-bottom:3rem !important; max-width:1220px; }

/* selection + scrollbar polish */
::selection{ background:rgba(34,211,238,0.28); }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:rgba(148,163,184,0.18); border-radius:8px; }
::-webkit-scrollbar-thumb:hover{ background:rgba(148,163,184,0.30); }

/* ── HEADER ─────────────────────────────────────────────── */
.top-header{
    display:flex; align-items:flex-end; justify-content:space-between;
    padding:6px 2px 2px; margin-bottom:22px;
}
.brand-title{
    font-size:32px; font-weight:900; letter-spacing:-1px; line-height:1.05; margin:0;
    background:linear-gradient(100deg,#A5F3FC 0%, #22D3EE 42%, #818CF8 100%);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.brand-sub{ color:var(--txt-dim); font-size:13.5px; font-weight:500; margin-top:5px; letter-spacing:0.1px; }
.status-chip{
    display:inline-flex; align-items:center; gap:8px;
    background:var(--glass); border:1px solid var(--stroke); border-radius:999px;
    padding:8px 16px; font-size:11.5px; font-weight:700; color:#B7C0CE; letter-spacing:0.8px;
    backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
    box-shadow:0 6px 20px rgba(0,0,0,0.35);
}
.dot-live{ width:8px; height:8px; border-radius:50%; background:var(--pos);
    box-shadow:0 0 0 0 rgba(52,211,153,0.6); animation:pulse 2s infinite; }
@keyframes pulse{
    0%{ box-shadow:0 0 0 0 rgba(52,211,153,0.55);}
    70%{ box-shadow:0 0 0 7px rgba(52,211,153,0);}
    100%{ box-shadow:0 0 0 0 rgba(52,211,153,0);}
}

/* ── KPI CARDS ──────────────────────────────────────────── */
.kpi-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:2px 0 6px; }
@media(max-width:880px){ .kpi-grid{ grid-template-columns:repeat(2,1fr);} }
.kpi{
    position:relative; overflow:hidden; padding:18px 18px 17px; border-radius:18px;
    background:var(--glass); border:1px solid var(--stroke);
    box-shadow:0 12px 30px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter:blur(14px) saturate(120%); -webkit-backdrop-filter:blur(14px) saturate(120%);
    transition:transform .18s ease, border-color .18s ease;
}
.kpi:hover{ transform:translateY(-2px); border-color:var(--stroke-strong); }
.kpi::after{ content:""; position:absolute; left:0; right:0; top:0; height:2px;
    background:linear-gradient(90deg, transparent, var(--acc, var(--neon)), transparent); opacity:.85; }
.kpi-top{ display:flex; align-items:center; justify-content:space-between; }
.kpi-ico{ width:36px; height:36px; border-radius:11px; display:grid; place-items:center;
    color:var(--acc, var(--neon));
    background:color-mix(in srgb, var(--acc, var(--neon)) 12%, transparent);
    border:1px solid color-mix(in srgb, var(--acc, var(--neon)) 30%, transparent); }
.kpi-ico svg{ width:19px; height:19px; }
.kpi-label{ font-size:10.5px; font-weight:800; letter-spacing:0.9px; text-transform:uppercase; color:var(--txt-faint); }
.kpi-value{ font-family:var(--mono); font-size:27px; font-weight:700; color:var(--txt);
    margin-top:12px; line-height:1; letter-spacing:-0.6px; white-space:nowrap; }
.kpi-sub{ font-size:11.5px; color:var(--txt-dim); margin-top:8px; font-weight:600; letter-spacing:0.2px; }
.kpi-sub .up{ color:var(--danger); } .kpi-sub .down{ color:var(--pos); }

/* ── SECTION LABELS ─────────────────────────────────────── */
.section-label{
    display:flex; align-items:center; gap:12px; font-size:12px; font-weight:800;
    color:#AEB7C4; text-transform:uppercase; letter-spacing:1.4px; margin:34px 0 15px;
}
.section-label .sq{ width:7px; height:7px; border-radius:2px; background:var(--neon);
    box-shadow:0 0 10px var(--neon); }
.section-label::after{ content:""; flex:1; height:1px;
    background:linear-gradient(90deg, rgba(148,163,184,0.22), transparent); }

/* ── ALERT / ANOMALY CARDS ──────────────────────────────── */
.alert{
    position:relative; display:flex; border-radius:16px; overflow:hidden; margin-bottom:6px;
    background:var(--glass); border:1px solid var(--stroke);
    box-shadow:0 10px 26px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.04);
    backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
    transition:border-color .18s ease, transform .18s ease;
}
.alert:hover{ transform:translateX(2px); border-color:var(--stroke-strong); }
.alert-rail{ flex:0 0 4px; background:var(--sev); box-shadow:0 0 16px var(--sev); }
.alert-body{ flex:1; padding:15px 20px 16px; }
.alert-row1{ display:flex; align-items:center; justify-content:space-between; gap:14px; }
.alert-left{ display:flex; align-items:center; gap:11px; }
.alert-date{ font-size:15.5px; font-weight:700; color:var(--txt); letter-spacing:-0.2px; }
.sev-pill{ font-size:9.5px; font-weight:800; text-transform:uppercase; letter-spacing:0.7px;
    padding:4px 10px; border-radius:999px; color:var(--sev);
    background:color-mix(in srgb, var(--sev) 13%, transparent);
    border:1px solid color-mix(in srgb, var(--sev) 34%, transparent); }
.alert-score{ font-family:var(--mono); font-weight:700; font-size:17px; color:var(--sev); }
.alert-stats{ display:flex; gap:9px; margin-top:13px; flex-wrap:wrap; }
.stat{ background:rgba(255,255,255,0.028); border:1px solid var(--stroke); border-radius:9px; padding:6px 12px; }
.stat-k{ font-size:9px; text-transform:uppercase; letter-spacing:0.6px; color:var(--txt-faint); font-weight:800; }
.stat-v{ font-family:var(--mono); font-size:13px; color:#D5DCE6; font-weight:600; margin-top:2px; }

/* historical event note */
.event-note{
    display:flex; gap:10px; align-items:flex-start; margin:2px 0 10px;
    background:linear-gradient(150deg, rgba(129,140,248,0.10), rgba(34,211,238,0.06));
    border:1px solid rgba(129,140,248,0.28); border-radius:12px; padding:12px 15px;
    color:#D7DCEC; font-size:13px; line-height:1.55; font-weight:500;
}
.event-note .pin{ color:var(--neon-2); font-size:14px; line-height:1.4; }

/* news pills */
.news{ background:rgba(255,255,255,0.026); border:1px solid var(--stroke); border-radius:12px;
    padding:12px 15px; margin-top:9px; }
.news-title{ color:#E6EBF2; font-size:13.5px; font-weight:600; line-height:1.45; }
.news-date{ color:var(--txt-faint); font-size:11px; margin-top:5px; font-family:var(--mono); }
.news-link{ display:inline-block; margin-top:9px; color:var(--neon); font-size:12px; font-weight:700;
    text-decoration:none; letter-spacing:0.2px; }
.news-link:hover{ text-decoration:underline; }

/* context / raw-data explainer */
.context-box{
    background:var(--glass); border:1px solid var(--stroke); border-radius:16px;
    padding:18px 22px; margin-bottom:16px; color:var(--txt-dim); font-size:13.5px; line-height:1.75;
    backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
}
.context-box b{ color:#DCE3EC; }

/* ── STREAMLIT WIDGET OVERRIDES ─────────────────────────── */
[data-testid="stExpander"]{
    background:rgba(255,255,255,0.02); border-radius:12px; border:1px solid var(--stroke); margin-top:2px;
}
[data-testid="stExpander"] summary{ font-size:13px; font-weight:600; color:#AEB7C4; }
[data-baseweb="select"] > div{
    background:var(--glass) !important; border:1px solid var(--stroke) !important;
    border-radius:12px !important; color:var(--txt) !important;
}
[data-baseweb="select"] > div:hover{ border-color:var(--stroke-strong) !important; }
[data-testid="stSelectbox"] label, .stTextInput label{
    color:#AEB7C4 !important; font-size:11.5px !important; font-weight:700 !important;
    letter-spacing:0.5px !important; text-transform:uppercase;
}
[data-testid="stCaptionContainer"]{ color:var(--txt-faint) !important; }
[data-testid="stDataFrame"]{ border:1px solid var(--stroke); border-radius:12px; overflow:hidden; }
hr, [data-testid="stDivider"]{ border-color:var(--stroke) !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
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
    "2015-08-24": "China devaluation fears trigger global market selloff ('Black Monday').",
    "2020-02-24": "COVID-19 fears trigger global market selloff as cases spread outside China.",
    "2020-03-16": "Circuit breakers halt trading as COVID-19 panic selling accelerates.",
    "2022-06-13": "S&P 500 enters bear market amid rate hike and inflation fears.",
}

# ── DATA LOGIC (unchanged) ───────────────────────────────────────────────────
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

# ── KPI CARDS ────────────────────────────────────────────────────────────────
ICO = {
    "activity": '<path d="M22 12h-4l-3 8L9 4l-3 8H2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    "target":   '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "shield":   '<path d="M12 21s7-3.4 7-9V5.5L12 3 5 5.5V12c0 5.6 7 9 7 9z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 8.5v3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="15" r="0.6" fill="currentColor" stroke="currentColor" stroke-width="1"/>',
    "clock":    '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 7.5v5l3.2 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
}

def kpi(icon, label, value, sub, accent):
    return f"""
    <div class="kpi" style="--acc:{accent}">
        <div class="kpi-top">
            <div class="kpi-label">{label}</div>
            <div class="kpi-ico"><svg viewBox="0 0 24 24">{ICO[icon]}</svg></div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

score = latest['Anomaly_Score']
thresh = latest['Threshold']
gap = score - thresh
if gap >= 0:
    gap_sub = f'<span class="up">▲ {gap:.2f} above threshold</span>'
else:
    gap_sub = f'<span class="down">▼ {abs(gap):.2f} below threshold</span>'

flagged_now = bool(latest['Flagged'])
status_txt = "ANOMALY" if flagged_now else "NORMAL"
status_acc = "var(--danger)" if flagged_now else "var(--pos)"
status_sub = "Market stress detected" if flagged_now else "Within normal range"

cards = "".join([
    kpi("activity", "Latest Anomaly Score", f"{score:.2f}", gap_sub, "var(--neon)"),
    kpi("target",   "Threshold",            f"{thresh:.2f}", "mean + 2σ baseline", "var(--neon-2)"),
    kpi("shield",   "Current Status",        status_txt, status_sub, status_acc),
    kpi("clock",    "Last Updated",          datetime.now().strftime("%d %b · %H:%M"), "Auto-refresh · 60 min", "#94A3B8"),
])
st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

# ── CHART ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label"><span class="sq"></span> Anomaly Score Timeline</div>', unsafe_allow_html=True)

view = st.selectbox("Select time range", ["Last 6 Months", "Last 2 Years", "Full History (2005-Present)"])
if view == "Last 6 Months":
    plot_df = df.tail(126)
elif view == "Last 2 Years":
    plot_df = df.tail(504).resample("W").last()
else:
    plot_df = df.resample("W").last()

y_top = max(plot_df['Anomaly_Score'].max(), thresh) * 1.10

fig = go.Figure()

# danger zone above the threshold
fig.add_hrect(y0=thresh, y1=y_top, line_width=0, fillcolor="rgba(251,75,87,0.055)", layer="below")

# soft neon glow underlay for the score line
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines',
    line=dict(color='rgba(34,211,238,0.28)', width=9, shape='spline', smoothing=0.35),
    hoverinfo='skip', showlegend=False
))
# main score line + fill
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df['Anomaly_Score'], mode='lines',
    name='Anomaly Score',
    line=dict(color='#22D3EE', width=2.4, shape='spline', smoothing=0.35),
    fill='tozeroy', fillcolor='rgba(34,211,238,0.10)',
    hovertemplate='Score  <b>%{y:.2f}</b><extra></extra>'
))

fig.add_hline(
    y=thresh, line_dash='dot', line_color='rgba(226,232,240,0.35)', line_width=1.4,
    annotation_text='THRESHOLD', annotation_font_color='#94A3B8',
    annotation_font_size=10, annotation_position='top left'
)

flagged_plot = plot_df[plot_df['Flagged']]
# glow halo under flagged markers
fig.add_trace(go.Scatter(
    x=flagged_plot.index, y=flagged_plot['Anomaly_Score'], mode='markers',
    marker=dict(color='rgba(251,75,87,0.28)', size=18, symbol='circle'),
    hoverinfo='skip', showlegend=False
))
fig.add_trace(go.Scatter(
    x=flagged_plot.index, y=flagged_plot['Anomaly_Score'], mode='markers',
    name='Flagged Day',
    marker=dict(color='#FB4B57', size=8, line=dict(color='#0B111A', width=1.6), symbol='circle'),
    hovertemplate='⚠ Flagged  ·  <b>%{y:.2f}</b><extra></extra>'
))

fig.update_layout(
    height=430,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#AEB7C4', family='Inter', size=12),
    legend=dict(orientation='h', y=1.10, x=1, xanchor='right', bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color='#8C97A8')),
    margin=dict(l=8, r=8, t=34, b=8),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='rgba(13,19,28,0.94)', bordercolor='rgba(148,163,184,0.25)',
                    font=dict(family='JetBrains Mono', size=12, color='#E8EEF5')),
)
fig.update_xaxes(
    showgrid=False, showline=True, linecolor='rgba(148,163,184,0.14)', zeroline=False,
    showspikes=True, spikemode='across', spikecolor='rgba(148,163,184,0.28)',
    spikethickness=1, spikedash='dot', ticks='outside', tickcolor='rgba(148,163,184,0.14)',
    tickfont=dict(size=11)
)
fig.update_yaxes(
    range=[0, y_top], showgrid=True, gridcolor='rgba(148,163,184,0.07)', zeroline=False,
    tickfont=dict(size=11), ticksuffix='  '
)
fig.update_traces(cliponaxis=False)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ── FLAGGED ANOMALY DAYS ─────────────────────────────────────────────────────
st.markdown('<div class="section-label"><span class="sq"></span> Flagged Anomaly Days</div>', unsafe_allow_html=True)
st.caption("Select a year (and optionally a month) to browse anomalies, then expand any card to load real news from that exact date.")

all_flags = df[df['Flagged']].sort_index(ascending=False)

available_years = sorted(all_flags.index.year.unique(), reverse=True)
year_options = ["All Years"] + [str(y) for y in available_years]

col_a, col_b = st.columns(2)
with col_a:
    selected_year = st.selectbox("Year", year_options)
with col_b:
    month_options = ["All Months"]
    if selected_year != "All Years":
        months_in_year = sorted(all_flags[all_flags.index.year == int(selected_year)].index.month.unique())
        month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        month_options += [month_names[m-1] for m in months_in_year]
    selected_month = st.selectbox("Month (optional)", month_options)

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
    is_severe = row['Anomaly_Score'] > row['Threshold'] * 1.3
    sev_label = "Severe" if is_severe else "Moderate"
    sev_color = "var(--danger)" if is_severe else "var(--warn)"

    st.markdown(f"""
    <div class="alert" style="--sev:{sev_color}">
        <div class="alert-rail"></div>
        <div class="alert-body">
            <div class="alert-row1">
                <div class="alert-left">
                    <span class="alert-date">{date_pretty}</span>
                    <span class="sev-pill">{sev_label}</span>
                </div>
                <span class="alert-score">{row['Anomaly_Score']:.2f}</span>
            </div>
            <div class="alert-stats">
                <div class="stat"><div class="stat-k">S&amp;P 500</div><div class="stat-v">{row['S&P500']:,.1f}</div></div>
                <div class="stat"><div class="stat-k">VIX</div><div class="stat-v">{row['VIX']:.1f}</div></div>
                <div class="stat"><div class="stat-k">Threshold</div><div class="stat-v">{row['Threshold']:.2f}</div></div>
                <div class="stat"><div class="stat-k">When</div><div class="stat-v">{days_ago}d ago</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📰  View news from {date_pretty}"):
        if date_str in HISTORICAL_EVENTS:
            st.markdown(f'<div class="event-note"><span class="pin">📌</span><span>{HISTORICAL_EVENTS[date_str]}</span></div>', unsafe_allow_html=True)
        with st.spinner("Fetching headlines..."):
            news = get_news_for_date(date_str)
        if news:
            for article in news:
                st.markdown(f"""
                <div class="news">
                    <div class="news-title">{article['title']}</div>
                    <div class="news-date">{article['pubDate']}</div>
                    <a class="news-link" href="{article['link']}" target="_blank">Read more →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No headlines found via automated search for this date.")

    st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

# ── RAW DATA EXPLORER ────────────────────────────────────────────────────────
st.markdown('<div class="section-label"><span class="sq"></span> Raw Data Explorer</div>', unsafe_allow_html=True)
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

st.caption("Data source: Yahoo Finance + Google News  ·  Model: Rolling 63-day z-score composite anomaly detection")
