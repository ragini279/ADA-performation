import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AcadPredict · AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CSS  –  Dark theme, every text token explicit
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

:root{
  --bg:      #060e1a;
  --surface: #0c1828;
  --card:    #111f32;
  --border:  #1c3454;
  --a1:#00d4ff; --a2:#7b61ff; --a3:#ff6b6b;
  --a4:#00ff9d; --gold:#ffd166;
  --txt:#e8f0fa;
  --txt2:#b8cfe8;
  --muted:#6a90b8;
}

html,body,[class*="css"]{
  font-family:'Syne',sans-serif !important;
  background-color:var(--bg) !important;
  color:var(--txt) !important;
}
.stApp {
  background-color: var(--bg) !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebar"],
footer,header{display:none !important;}

.main .block-container{
  padding:1.5rem 2.5rem 4rem !important;
  max-width:1440px;
}

/* HERO */
.hero{
  background:linear-gradient(135deg,#07101f 0%,#0a1a30 50%,#06111f 100%);
  border:1px solid var(--border);border-radius:20px;
  padding:3rem 3.5rem;margin-bottom:2.5rem;
  position:relative;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;top:-60px;right:-60px;
  width:320px;height:320px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,212,255,.13) 0%,transparent 70%);
}
.hero-tag{
  font-family:'JetBrains Mono',monospace;
  font-size:.72rem;letter-spacing:.25em;text-transform:uppercase;
  color:var(--a1);margin-bottom:.8rem;
}
.hero h1{
  font-size:2.6rem;font-weight:800;line-height:1.15;margin:0 0 1rem;
  background:linear-gradient(90deg,#ffffff 0%,var(--a1) 55%,var(--a2) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-sub{font-size:1rem;color:var(--txt2);max-width:700px;line-height:1.75;}
.hero-badges{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1.5rem;}
.badge{
  font-family:'JetBrains Mono',monospace;font-size:.68rem;
  padding:.28rem .8rem;border-radius:100px;border:1px solid;
  letter-spacing:.1em;text-transform:uppercase;font-weight:600;
}
.b-blue  {color:var(--a1);border-color:var(--a1);background:rgba(0,212,255,.08);}
.b-purple{color:var(--a2);border-color:var(--a2);background:rgba(123,97,255,.08);}
.b-green {color:var(--a4);border-color:var(--a4);background:rgba(0,255,157,.08);}
.b-red   {color:var(--a3);border-color:var(--a3);background:rgba(255,107,107,.08);}
.b-gold  {color:var(--gold);border-color:var(--gold);background:rgba(255,209,102,.08);}

/* SECTION HEADER */
.sh{
  display:flex;align-items:center;gap:.8rem;
  margin:2.5rem 0 1.2rem;padding-bottom:.75rem;
  border-bottom:1px solid var(--border);
}
.sh-icon{
  width:36px;height:36px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;
}
.ic-blue  {background:rgba(0,212,255,.15);}
.ic-purple{background:rgba(123,97,255,.15);}
.ic-green {background:rgba(0,255,157,.15);}
.ic-red   {background:rgba(255,107,107,.15);}
.ic-gold  {background:rgba(255,209,102,.15);}
.sh h2{font-size:1.25rem;font-weight:700;color:var(--txt);margin:0;}
.sh .sub{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--muted);margin-left:.4rem;}

/* KPI CARDS */
.kpi-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
  gap:1rem;margin-bottom:.5rem;
}
.kpi{
  background:var(--card);border:1px solid var(--border);
  border-radius:16px;padding:1.4rem 1.2rem;position:relative;overflow:hidden;
}
.kpi::before{
  content:'';position:absolute;top:0;left:0;right:0;
  height:3px;border-radius:16px 16px 0 0;
}
.kpi-b::before{background:linear-gradient(90deg,var(--a1),transparent);}
.kpi-p::before{background:linear-gradient(90deg,var(--a2),transparent);}
.kpi-g::before{background:linear-gradient(90deg,var(--a4),transparent);}
.kpi-r::before{background:linear-gradient(90deg,var(--a3),transparent);}
.kpi-o::before{background:linear-gradient(90deg,var(--gold),transparent);}
.kpi-lbl{
  font-family:'JetBrains Mono',monospace;font-size:.64rem;
  letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:.45rem;
}
.kpi-val{font-size:2rem;font-weight:800;line-height:1;margin-bottom:.15rem;}
.kpi-b .kpi-val{color:var(--a1);}
.kpi-p .kpi-val{color:var(--a2);}
.kpi-g .kpi-val{color:var(--a4);}
.kpi-r .kpi-val{color:var(--a3);}
.kpi-o .kpi-val{color:var(--gold);}
.kpi-desc{font-size:.72rem;color:var(--muted);}

/* INFO CARD */
.ic{
  background:var(--card);border:1px solid var(--border);
  border-radius:16px;padding:1.5rem 1.8rem;margin-bottom:1rem;
}
.ic h3{font-size:1.02rem;font-weight:700;color:var(--txt);margin:0 0 .6rem;}
.ic p {font-size:.84rem;color:var(--txt2);line-height:1.75;margin:0;}

/* SOURCE CHIPS */
.src-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:.8rem;margin-top:.5rem;
}
.src{
  background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:1rem 1.2rem;display:flex;align-items:flex-start;gap:.8rem;
}
.src-ico{font-size:1.4rem;line-height:1;flex-shrink:0;}
.src-title{font-size:.8rem;font-weight:700;color:var(--txt);margin-bottom:.2rem;}
.src-desc {font-size:.7rem;color:var(--txt2);line-height:1.5;}

/* ARCH FLOW */
.arch-flow{
  display:flex;flex-wrap:wrap;gap:0;
  align-items:center;justify-content:center;margin:1.5rem 0;
}
.arch-step{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:1rem 1.2rem;text-align:center;min-width:110px;
}
.a-ico{font-size:1.5rem;}
.a-num{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:var(--muted);letter-spacing:.15em;margin-top:.2rem;}
.a-lbl{font-size:.75rem;font-weight:700;color:var(--txt);margin-top:.2rem;}
.arch-arr{font-size:1.2rem;color:var(--a1);padding:0 .3rem;opacity:.6;}

/* METRIC TABLE */
.mtbl{width:100%;border-collapse:collapse;font-size:.85rem;}
.mtbl th{
  font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--muted);
  padding:.6rem 1rem;text-align:left;border-bottom:1px solid var(--border);
}
.mtbl td{padding:.7rem 1rem;border-bottom:1px solid rgba(28,52,84,.5);color:var(--txt);}
.mtbl tr:hover td{background:rgba(0,212,255,.03);}
.best-row td{color:var(--a4) !important;font-weight:700;}

/* WARN CARD */
.warn{
  background:rgba(255,107,107,.07);border:1px solid rgba(255,107,107,.3);
  border-radius:12px;padding:.9rem 1.3rem;display:flex;align-items:center;
  gap:.8rem;margin-bottom:.55rem;font-size:.82rem;color:#ffb8b8;
}
.warn-ico{font-size:1.1rem;flex-shrink:0;}
.warn strong{color:#ffcccc;}

/* PREDICT BOX */
.pred-box{
  background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(123,97,255,.08));
  border:1px solid rgba(0,212,255,.3);border-radius:18px;
  padding:2rem;text-align:center;
}
.pred-lbl{
  font-family:'JetBrains Mono',monospace;font-size:.7rem;
  letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;
}
.pred-val{
  font-size:3.5rem;font-weight:800;line-height:1;
  background:linear-gradient(90deg,var(--a1),var(--a2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.pred-cat{font-size:1rem;font-weight:700;margin-top:.5rem;}
.pred-meta{margin-top:.7rem;font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--muted);}

/* TABS */
div[data-baseweb="tab-list"]{
  background:var(--surface) !important;border-radius:12px !important;
  padding:4px !important;border:1px solid var(--border) !important;gap:2px !important;
}
div[data-baseweb="tab"]{
  background:transparent !important;border-radius:8px !important;
  color:var(--txt2) !important;font-family:'Syne',sans-serif !important;
  font-weight:600 !important;font-size:.82rem !important;padding:.5rem 1.2rem !important;
}
div[data-baseweb="tab"]:hover{color:var(--txt) !important;}
div[aria-selected="true"][data-baseweb="tab"]{
  background:var(--card) !important;color:var(--a1) !important;
}
div[data-baseweb="tab-highlight"],div[data-baseweb="tab-border"]{display:none !important;}

/* ALL FORM LABELS — force light colour */
label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
.stSlider label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label{
  color:var(--txt2) !important;
  font-size:.78rem !important;
  font-family:'JetBrains Mono',monospace !important;
  letter-spacing:.06em !important;
  text-transform:uppercase !important;
}

/* Slider current-value text */
[data-testid="stMarkdownContainer"] p{color:var(--txt) !important;}

/* Selectbox / input backgrounds */
div[data-baseweb="select"]>div,
.stTextInput>div>div>input,
.stNumberInput>div>div>input{
  background:var(--surface) !important;border-color:var(--border) !important;
  color:var(--txt) !important;border-radius:10px !important;
  font-family:'Syne',sans-serif !important;
}
[data-baseweb="popover"] li{color:var(--txt) !important;background:var(--surface) !important;}
[data-baseweb="popover"] li:hover{background:var(--card) !important;}

/* BUTTON */
.stButton>button{
  background:linear-gradient(135deg,var(--a1),var(--a2)) !important;
  color:#000 !important;font-family:'Syne',sans-serif !important;
  font-weight:700 !important;border:none !important;
  border-radius:10px !important;padding:.6rem 2rem !important;
  font-size:.9rem !important;letter-spacing:.05em !important;
}
.stButton>button:hover{opacity:.86 !important;}

/* EXPANDER */
details{
  background:var(--card) !important;border:1px solid var(--border) !important;
  border-radius:12px !important;overflow:hidden !important;
}
summary{color:var(--txt) !important;font-family:'Syne',sans-serif !important;font-weight:600 !important;padding:.8rem 1.2rem !important;}

/* DATAFRAME */
[data-testid="stDataFrame"] *{color:var(--txt) !important;}

/* MARKDOWN containers inside st.markdown */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3{color:var(--txt) !important;}

/* PARAGRAPH sub-labels rendered via st.markdown */
.sub-label{color:var(--txt) !important;font-weight:700;font-size:.9rem;margin-bottom:.4rem;}

.footer-note{
  text-align:center;padding:2rem;color:var(--border);
  font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.15em;margin-top:2rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PLOTLY HELPERS
# ──────────────────────────────────────────────
PBG  = "#0c1828"
GRID = "#1c3454"
TCOL = "#e8f0fa"
COLS = ["#00d4ff","#7b61ff","#ff6b6b","#00ff9d","#ffd166","#ff9f43"]

def sf(fig, h=None):
    upd = dict(
        paper_bgcolor=PBG, plot_bgcolor=PBG,
        font=dict(family="Syne, sans-serif", color=TCOL, size=12),
        margin=dict(l=24, r=24, t=52, b=24),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TCOL)),
    )
    if h:
        upd["height"] = h
    fig.update_layout(**upd)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=TCOL, title_font_color=TCOL)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=TCOL, title_font_color=TCOL)
    return fig

def section(icon, icon_cls, title, sub=""):
    sub_html = f'<span class="sub">· {sub}</span>' if sub else ""
    st.markdown(
        f'<div class="sh"><div class="sh-icon {icon_cls}">{icon}</div>'
        f'<h2>{title}</h2>{sub_html}</div>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────
# DATA + FEATURES
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("C:/Users/varma/Desktop/academic_performance_dataset_3000.csv")
    except Exception:
        np.random.seed(42); n = 3000
        df = pd.DataFrame({
            "student_id": range(1,n+1),
            "study_time_hours": np.random.randint(1,11,n),
            "attendance_percentage": np.random.randint(40,100,n),
            "assignments_completed": np.random.randint(1,11,n),
            "lms_clicks": np.random.randint(20,500,n),
            "forum_participation": np.random.randint(0,25,n),
            "sleep_hours": np.round(np.random.uniform(4,9,n),1),
            "internet_usage_hours": np.round(np.random.uniform(1,8,n),1),
            "previous_grade": np.random.randint(30,100,n),
            "final_grade": np.random.randint(35,100,n).astype(float),
        })
    df = df.dropna(subset=["final_grade"])
    df["final_grade"] = pd.to_numeric(df["final_grade"], errors="coerce")
    df = df.dropna(subset=["final_grade"])
    mx = df.max(numeric_only=True)
    df["engagement_score"] = (
        df["lms_clicks"]/mx["lms_clicks"]*0.4 +
        df["forum_participation"]/mx["forum_participation"]*0.3 +
        df["assignments_completed"]/mx["assignments_completed"]*0.3
    ).round(4)
    df["consistency_index"] = (
        df["attendance_percentage"]/100*0.5 +
        df["study_time_hours"]/mx["study_time_hours"]*0.5
    ).round(4)
    df["lifestyle_score"] = (
        df["sleep_hours"]/mx["sleep_hours"]*0.5 -
        df["internet_usage_hours"]/mx["internet_usage_hours"]*0.5
    ).round(4)
    df["performance_label"] = df["final_grade"].apply(
        lambda g: "Distinction" if g>=75 else ("Pass" if g>=50 else "Fail"))
    df["risk_level"] = df["final_grade"].apply(
        lambda g: "High Risk" if g<40 else ("Medium Risk" if g<55 else "Low Risk"))
    return df

FEATS = [
    "study_time_hours","attendance_percentage","assignments_completed",
    "lms_clicks","forum_participation","sleep_hours","internet_usage_hours",
    "previous_grade","engagement_score","consistency_index","lifestyle_score"
]

@st.cache_resource
def train_models(_df):
    X  = _df[FEATS]
    yr = _df["final_grade"]
    le = LabelEncoder()
    yc = le.fit_transform(_df["performance_label"])
    Xtr,Xte,yr_tr,yr_te,yc_tr,yc_te = train_test_split(X,yr,yc,test_size=0.2,random_state=42)
    sc = StandardScaler(); Xs_tr=sc.fit_transform(Xtr); Xs_te=sc.transform(Xte)
    res = {}

    lr = LinearRegression().fit(Xs_tr,yr_tr); lp=lr.predict(Xs_te)
    res["Linear Regression"]=dict(model=lr,scaler=sc,
        rmse=float(np.sqrt(mean_squared_error(yr_te,lp))),r2=float(r2_score(yr_te,lp)),
        y_te=yr_te.values,y_pred=lp,importances=None)

    rf = RandomForestRegressor(n_estimators=120,random_state=42,n_jobs=-1).fit(Xtr,yr_tr); rp=rf.predict(Xte)
    res["Random Forest"]=dict(model=rf,scaler=None,
        rmse=float(np.sqrt(mean_squared_error(yr_te,rp))),r2=float(r2_score(yr_te,rp)),
        y_te=yr_te.values,y_pred=rp,importances=rf.feature_importances_)

    if XGB_AVAILABLE:
        xgb=XGBRegressor(n_estimators=120,learning_rate=0.1,random_state=42,verbosity=0).fit(Xtr,yr_tr); xp=xgb.predict(Xte)
        res["XGBoost"]=dict(model=xgb,scaler=None,
            rmse=float(np.sqrt(mean_squared_error(yr_te,xp))),r2=float(r2_score(yr_te,xp)),
            y_te=yr_te.values,y_pred=xp,importances=xgb.feature_importances_)

    rfc=RandomForestClassifier(n_estimators=120,random_state=42,n_jobs=-1).fit(Xtr,yc_tr)
    clf_acc=float(accuracy_score(yc_te,rfc.predict(Xte)))

    km=KMeans(n_clusters=3,random_state=42,n_init=10).fit(X)
    df2=_df.copy(); df2["segment"]=km.labels_
    means=df2.groupby("segment")["final_grade"].mean()
    s_map={means.idxmax():"High Performer",means.idxmin():"Low Performer"}
    s_map[[s for s in means.index if s not in s_map][0]]="Average"
    df2["segment_label"]=df2["segment"].map(s_map)
    return res,rfc,clf_acc,le,df2

df                                = load_data()
model_res,clf_model,clf_acc,le,df_seg = train_models(df)
best_name = min(model_res, key=lambda k: model_res[k]["rmse"])
best      = model_res[best_name]

# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">&#127891; Machine Learning &nbsp;&middot;&nbsp; Academic Intelligence &nbsp;&middot;&nbsp; v2.1</div>
  <h1>Academic Performance Prediction<br>Using Multisource Behavioral Data</h1>
  <div class="hero-sub">
    A comprehensive AI-powered dashboard integrating academic records, LMS activity,
    behavioral patterns, and lifestyle data to predict student performance &mdash;
    identify at-risk students early and reveal key success factors.
  </div>
  <div class="hero-badges">
    <span class="badge b-blue">&#128202; Multisource Data</span>
    <span class="badge b-purple">&#129302; ML Models</span>
    <span class="badge b-green">&#128293; Feature Engineering</span>
    <span class="badge b-red">&#9888;&#65039; Early Warning</span>
    <span class="badge b-gold">&#128200; 3 000 Students</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview","🔬 Data Explorer","🤖 Models",
    "🔥 Features","⚠️ Early Warning","👥 Segmentation",
    "🎯 Predict","📋 About"
])

# ════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ════════════════════════════════════════════════
with tabs[0]:
    total=len(df); avg_g=df["final_grade"].mean()
    at_risk=(df["risk_level"]=="High Risk").sum()
    distinction=(df["performance_label"]=="Distinction").sum()
    pass_rate=(df["performance_label"]!="Fail").mean()*100

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi kpi-b"><div class="kpi-lbl">Total Students</div><div class="kpi-val">{total:,}</div><div class="kpi-desc">In dataset</div></div>
      <div class="kpi kpi-p"><div class="kpi-lbl">Avg Final Grade</div><div class="kpi-val">{avg_g:.1f}</div><div class="kpi-desc">Out of 100</div></div>
      <div class="kpi kpi-g"><div class="kpi-lbl">Pass Rate</div><div class="kpi-val">{pass_rate:.0f}%</div><div class="kpi-desc">Pass + Distinction</div></div>
      <div class="kpi kpi-o"><div class="kpi-lbl">Distinctions</div><div class="kpi-val">{distinction:,}</div><div class="kpi-desc">Grade &ge; 75</div></div>
      <div class="kpi kpi-r"><div class="kpi-lbl">At-Risk</div><div class="kpi-val">{at_risk:,}</div><div class="kpi-desc">Grade &lt; 40</div></div>
    </div>
    """, unsafe_allow_html=True)

    section("🌐","ic-blue","Multisource Data Architecture")
    st.markdown("""
    <div class="src-grid">
      <div class="src"><div class="src-ico">&#128202;</div><div><div class="src-title">Academic Records</div><div class="src-desc">previous_grade, final_grade &mdash; historical performance baseline</div></div></div>
      <div class="src"><div class="src-ico">&#128187;</div><div><div class="src-title">LMS Activity</div><div class="src-desc">lms_clicks, forum_participation &mdash; digital engagement metrics</div></div></div>
      <div class="src"><div class="src-ico">&#128218;</div><div><div class="src-title">Behavioral Data</div><div class="src-desc">study_time_hours, assignments_completed &mdash; learning habits</div></div></div>
      <div class="src"><div class="src-ico">&#129504;</div><div><div class="src-title">Lifestyle Patterns</div><div class="src-desc">sleep_hours, internet_usage_hours &mdash; wellbeing indicators</div></div></div>
    </div>
    """, unsafe_allow_html=True)

    section("⚙️","ic-purple","System Architecture Pipeline")
    st.markdown("""
    <div class="arch-flow">
      <div class="arch-step"><div class="a-ico">&#128229;</div><div class="a-num">STEP 1</div><div class="a-lbl">Data Collection</div></div>
      <div class="arch-arr">&rarr;</div>
      <div class="arch-step"><div class="a-ico">&#129529;</div><div class="a-num">STEP 2</div><div class="a-lbl">Preprocessing</div></div>
      <div class="arch-arr">&rarr;</div>
      <div class="arch-step"><div class="a-ico">&#9879;&#65039;</div><div class="a-num">STEP 3</div><div class="a-lbl">Feature Eng.</div></div>
      <div class="arch-arr">&rarr;</div>
      <div class="arch-step"><div class="a-ico">&#129302;</div><div class="a-num">STEP 4</div><div class="a-lbl">Model Training</div></div>
      <div class="arch-arr">&rarr;</div>
      <div class="arch-step"><div class="a-ico">&#127919;</div><div class="a-num">STEP 5</div><div class="a-lbl">Prediction</div></div>
      <div class="arch-arr">&rarr;</div>
      <div class="arch-step"><div class="a-ico">&#128200;</div><div class="a-num">STEP 6</div><div class="a-lbl">Evaluation</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        section("📈","ic-green","Grade Distribution")
        fig=go.Figure(go.Histogram(x=df["final_grade"],nbinsx=30,
            marker=dict(color=df["final_grade"],
                colorscale=[[0,"#ff6b6b"],[.5,"#ffd166"],[1,"#00d4ff"]],
                line=dict(color=PBG,width=.5))))
        fig.update_layout(title_text="Final Grade Frequency",showlegend=False)
        st.plotly_chart(sf(fig,360),use_container_width=True)
    with c2:
        section("🥧","ic-gold","Performance Labels")
        vc=df["performance_label"].value_counts()
        fig2=go.Figure(go.Pie(labels=vc.index,values=vc.values,hole=.55,
            marker=dict(colors=["#00ff9d","#00d4ff","#ff6b6b"],line=dict(color=PBG,width=2)),
            textfont=dict(family="Syne",size=12,color="#fff")))
        fig2.update_layout(title_text="Pass / Fail / Distinction")
        st.plotly_chart(sf(fig2,360),use_container_width=True)

    section("🔗","ic-blue","Feature Correlation Heatmap")
    cc=["study_time_hours","attendance_percentage","assignments_completed",
        "lms_clicks","forum_participation","sleep_hours",
        "internet_usage_hours","previous_grade","final_grade",
        "engagement_score","consistency_index"]
    corr=df[cc].corr()
    fig3=go.Figure(go.Heatmap(
        z=corr.values,x=corr.columns,y=corr.index,
        colorscale=[[0,"#ff6b6b"],[.5,"#0c1828"],[1,"#00d4ff"]],
        zmin=-1,zmax=1,
        text=np.round(corr.values,2),texttemplate="%{text}",
        textfont=dict(size=9,family="JetBrains Mono",color="#e8f0fa")))
    fig3.update_layout(title_text="Feature Correlation Matrix",height=440)
    st.plotly_chart(sf(fig3),use_container_width=True)

# ════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ════════════════════════════════════════════════
with tabs[1]:
    section("🔬","ic-blue","Dataset Explorer","3 000 students · 9 raw + 3 engineered features")
    c1,c2=st.columns([3,1])
    with c2:
        n_show=st.slider("Rows to display",5,100,20)
        filt_risk=st.selectbox("Filter by Risk",["All","High Risk","Medium Risk","Low Risk"])
    disp=df.copy()
    if filt_risk!="All":
        disp=disp[disp["risk_level"]==filt_risk]
    show_cols=["student_id","study_time_hours","attendance_percentage",
               "assignments_completed","lms_clicks","sleep_hours",
               "previous_grade","final_grade","engagement_score",
               "consistency_index","performance_label","risk_level"]
    st.dataframe(
        disp[show_cols].head(n_show).style
            .format({"engagement_score":"{:.3f}","consistency_index":"{:.3f}","final_grade":"{:.1f}"}),
        use_container_width=True,height=370)

    section("📊","ic-purple","Feature Distribution & Scatter")
    feat_sel=st.selectbox("Select feature to explore",FEATS,index=0)
    fig4=make_subplots(rows=1,cols=2,subplot_titles=["Distribution","vs Final Grade"])
    fig4.add_trace(go.Histogram(x=df[feat_sel],marker_color=COLS[0],nbinsx=25,name=feat_sel),row=1,col=1)
    fig4.add_trace(go.Scatter(x=df[feat_sel],y=df["final_grade"],mode="markers",
        marker=dict(color=df["final_grade"],colorscale=[[0,"#ff6b6b"],[1,"#00d4ff"]],size=3,opacity=.5),
        name="vs Grade"),row=1,col=2)
    fig4.update_layout(paper_bgcolor=PBG,plot_bgcolor=PBG,font=dict(family="Syne",color=TCOL),
        margin=dict(l=20,r=20,t=50,b=20),showlegend=False,height=340)
    fig4.update_xaxes(gridcolor=GRID,color=TCOL); fig4.update_yaxes(gridcolor=GRID,color=TCOL)
    st.plotly_chart(fig4,use_container_width=True)

    section("🧮","ic-green","Descriptive Statistics")
    st.dataframe(df[FEATS+["final_grade"]].describe().T.style.format("{:.2f}"),use_container_width=True)

# ════════════════════════════════════════════════
# TAB 2 — MODELS
# ════════════════════════════════════════════════
with tabs[2]:
    section("🤖","ic-blue","Model Comparison","Regression · Classification")
    rows=[]
    for name,r in model_res.items():
        rows.append({"Model":name,"RMSE":f"{r['rmse']:.4f}","R² Score":f"{r['r2']:.4f}",
                     "Best":"✅" if name==best_name else ""})
    tdf=pd.DataFrame(rows)
    tbl='<table class="mtbl"><thead><tr>'
    for c in tdf.columns:
        tbl+=f"<th>{c}</th>"
    tbl+="</tr></thead><tbody>"
    for _,row in tdf.iterrows():
        cls="best-row" if row["Best"]=="✅" else ""
        tbl+=f'<tr class="{cls}">'
        for v in row: tbl+=f"<td>{v}</td>"
        tbl+="</tr>"
    tbl+="</tbody></table>"
    st.markdown(f'<div class="ic">{tbl}</div>',unsafe_allow_html=True)

    st.markdown(
        f'<div class="ic" style="border-color:rgba(0,255,157,.35);background:rgba(0,255,157,.05);">'
        f'<h3 style="color:#00ff9d">&#127942; Best Model: {best_name}</h3>'
        f'<p>RMSE = {best["rmse"]:.4f} &nbsp;|&nbsp; R&sup2; = {best["r2"]:.4f}<br>'
        f'Lower RMSE and higher R&sup2; indicate a better fit to the data.</p></div>',
        unsafe_allow_html=True)

    section("📈","ic-purple","Actual vs Predicted")
    mc=st.selectbox("Select model to visualise",list(model_res.keys()))
    r=model_res[mc]
    c1,c2=st.columns(2)
    with c1:
        mn,mx=float(r["y_te"].min()),float(r["y_te"].max())
        fig5=go.Figure()
        fig5.add_trace(go.Scatter(x=r["y_te"],y=r["y_pred"],mode="markers",
            marker=dict(color=COLS[0],size=4,opacity=.5),name="Predictions"))
        fig5.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],
            line=dict(color=COLS[2],dash="dash"),name="Perfect Fit"))
        fig5.update_layout(title_text="Actual vs Predicted",xaxis_title="Actual",yaxis_title="Predicted")
        st.plotly_chart(sf(fig5,360),use_container_width=True)
    with c2:
        res2=r["y_te"]-r["y_pred"]
        fig6=go.Figure(go.Histogram(x=res2,marker_color=COLS[1],nbinsx=30))
        fig6.update_layout(title_text="Residual Distribution",xaxis_title="Residual")
        st.plotly_chart(sf(fig6,360),use_container_width=True)

    section("🏷️","ic-green","Classification Results","Pass / Fail / Distinction")
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
      <div class="kpi kpi-g"><div class="kpi-lbl">Classifier Accuracy</div><div class="kpi-val">{clf_acc*100:.1f}%</div><div class="kpi-desc">Random Forest Classifier</div></div>
      <div class="kpi kpi-b"><div class="kpi-lbl">Classes</div><div class="kpi-val">3</div><div class="kpi-desc">Fail &middot; Pass &middot; Distinction</div></div>
      <div class="kpi kpi-p"><div class="kpi-lbl">Train / Test Split</div><div class="kpi-val">80/20</div><div class="kpi-desc">80% train &middot; 20% test</div></div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 3 — FEATURES
# ════════════════════════════════════════════════
with tabs[3]:
    section("🔥","ic-gold","Feature Importance","What drives academic performance?")
    imp_name="XGBoost" if ("XGBoost" in model_res and
        model_res["XGBoost"]["r2"]>model_res["Random Forest"]["r2"]) else "Random Forest"
    imps=model_res[imp_name]["importances"]
    fimp=pd.DataFrame({"Feature":FEATS,"Importance":imps}).sort_values("Importance",ascending=True)
    fig7=go.Figure(go.Bar(x=fimp["Importance"],y=fimp["Feature"],orientation="h",
        marker=dict(color=fimp["Importance"],colorscale=[[0,"#1c3454"],[.5,"#7b61ff"],[1,"#00d4ff"]],
            line=dict(color=PBG,width=.5)),
        text=[f"{v:.4f}" for v in fimp["Importance"]],textposition="outside",
        textfont=dict(color=TCOL,family="JetBrains Mono",size=10)))
    fig7.update_layout(title_text=f"Feature Importance — {imp_name}",height=430,xaxis_title="Importance Score")
    st.plotly_chart(sf(fig7),use_container_width=True)

    section("⚗️","ic-purple","Engineered Features Explained")
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('<div class="ic"><h3 style="color:#00d4ff">&#9889; Engagement Score</h3>'
            '<p>Composite of LMS clicks (40%), forum participation (30%), and assignments completed (30%). '
            'Captures how actively a student interacts with the learning platform.</p></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ic"><h3 style="color:#7b61ff">&#127919; Consistency Index</h3>'
            '<p>Weighted average of attendance percentage (50%) and study time hours (50%). '
            'Measures disciplined, regular learning behaviour sustained over time.</p></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="ic"><h3 style="color:#00ff9d">&#127769; Lifestyle Score</h3>'
            '<p>Positive weight for adequate sleep (50%) minus a penalty for excessive internet usage (50%). '
            'Reflects how wellbeing patterns impact academic outcomes.</p></div>',
            unsafe_allow_html=True)

    section("📉","ic-green","Engagement Score vs Final Grade")
    fig8=px.scatter(df,x="engagement_score",y="final_grade",color="performance_label",
        color_discrete_map={"Distinction":"#00d4ff","Pass":"#00ff9d","Fail":"#ff6b6b"},opacity=.55,
        labels={"engagement_score":"Engagement Score","final_grade":"Final Grade"})
    fig8.update_traces(marker=dict(size=4))
    st.plotly_chart(sf(fig8),use_container_width=True)

    section("📦","ic-red","Grade by Behavioral Groups")
    box_feat=st.selectbox("Group by feature",
        ["study_time_hours","sleep_hours","attendance_percentage","assignments_completed"])
    df_box=df.copy()
    df_box["group"]=pd.cut(df_box[box_feat],bins=4).astype(str)
    fig9=px.box(df_box,x="group",y="final_grade",color="group",color_discrete_sequence=COLS,
        labels={"group":box_feat.replace("_"," ").title(),"final_grade":"Final Grade"})
    fig9.update_layout(showlegend=False)
    st.plotly_chart(sf(fig9),use_container_width=True)

# ════════════════════════════════════════════════
# TAB 4 — EARLY WARNING
# ════════════════════════════════════════════════
with tabs[4]:
    # FIXED: apostrophe removed — "it is too late" instead of "it's too late"
    section("⚠️","ic-red","Early Warning System","Identify at-risk students before it is too late")
    rc=df["risk_level"].value_counts()
    c1,c2,c3=st.columns(3)
    for col,rk,cls in zip([c1,c2,c3],
                           ["High Risk","Medium Risk","Low Risk"],
                           ["kpi-r","kpi-o","kpi-g"]):
        cnt=int(rc.get(rk,0)); pct=cnt/len(df)*100
        with col:
            st.markdown(
                f'<div class="kpi {cls}"><div class="kpi-lbl">{rk}</div>'
                f'<div class="kpi-val">{cnt:,}</div><div class="kpi-desc">{pct:.1f}% of students</div></div>',
                unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        fig10=go.Figure(go.Pie(labels=rc.index,values=rc.values,hole=.58,
            marker=dict(colors=["#ff6b6b","#ffd166","#00ff9d"],line=dict(color=PBG,width=2)),
            textfont=dict(family="Syne",size=11,color="#fff")))
        fig10.update_layout(title_text="Risk Distribution",legend=dict(orientation="h"))
        st.plotly_chart(sf(fig10,360),use_container_width=True)
    with c2:
        fig11=px.scatter(df,x="consistency_index",y="engagement_score",color="risk_level",
            color_discrete_map={"High Risk":"#ff6b6b","Medium Risk":"#ffd166","Low Risk":"#00ff9d"},
            opacity=.55,labels={"consistency_index":"Consistency Index","engagement_score":"Engagement Score"})
        fig11.update_traces(marker=dict(size=4))
        fig11.update_layout(title_text="Risk by Engagement & Consistency")
        st.plotly_chart(sf(fig11,360),use_container_width=True)

    section("🚨","ic-red","High-Risk Student List","Showing up to 30 students with grade < 40")
    high_risk=df[df["risk_level"]=="High Risk"][
        ["student_id","final_grade","previous_grade",
         "attendance_percentage","study_time_hours","engagement_score"]
    ].sort_values("final_grade").head(30)

    for _,row in high_risk.iterrows():
        st.markdown(
            f'<div class="warn"><div class="warn-ico">&#9888;&#65039;</div>'
            f'<div><strong>Student #{int(row["student_id"])}</strong>'
            f' &nbsp;|&nbsp; Grade: <strong>{row["final_grade"]:.1f}</strong>'
            f' &nbsp;|&nbsp; Attendance: {row["attendance_percentage"]:.0f}%'
            f' &nbsp;|&nbsp; Study Time: {row["study_time_hours"]}h/day'
            f' &nbsp;|&nbsp; Engagement: {row["engagement_score"]:.3f}</div></div>',
            unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 5 — SEGMENTATION
# ════════════════════════════════════════════════
with tabs[5]:
    section("👥","ic-purple","Student Segmentation","K-Means Clustering · 3 Groups")
    sc=df_seg["segment_label"].value_counts()
    c1,c2,c3=st.columns(3)
    for col,seg,cls in zip([c1,c2,c3],
                            ["High Performer","Average","Low Performer"],
                            ["kpi-b","kpi-o","kpi-r"]):
        cnt=int(sc.get(seg,0)); avg2=df_seg[df_seg["segment_label"]==seg]["final_grade"].mean()
        with col:
            st.markdown(
                f'<div class="kpi {cls}"><div class="kpi-lbl">{seg}</div>'
                f'<div class="kpi-val">{cnt:,}</div><div class="kpi-desc">Avg Grade: {avg2:.1f}</div></div>',
                unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        fig12=px.scatter(df_seg,x="engagement_score",y="consistency_index",color="segment_label",
            color_discrete_map={"High Performer":"#00d4ff","Average":"#ffd166","Low Performer":"#ff6b6b"},
            opacity=.55)
        fig12.update_traces(marker=dict(size=4))
        fig12.update_layout(title_text="Clusters — Engagement vs Consistency")
        st.plotly_chart(sf(fig12,370),use_container_width=True)
    with c2:
        cats=["study_time_hours","attendance_percentage","sleep_hours","engagement_score","consistency_index"]
        seg_means=df_seg.groupby("segment_label")[cats].mean().reset_index()
        mx_vals=df[cats].max()
        fig13=go.Figure()
        seg_colors={"High Performer":"#00d4ff","Average":"#ffd166","Low Performer":"#ff6b6b"}
        for _,row in seg_means.iterrows():
            vals=[row[c]/mx_vals[c] for c in cats]+[row[cats[0]]/mx_vals[cats[0]]]
            fig13.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],fill="toself",
                name=row["segment_label"],line=dict(color=seg_colors[row["segment_label"]])))
        fig13.update_layout(
            polar=dict(bgcolor=PBG,
                radialaxis=dict(visible=True,range=[0,1],gridcolor=GRID,linecolor=GRID,color=TCOL),
                angularaxis=dict(color=TCOL)),
            paper_bgcolor=PBG,font=dict(color=TCOL,family="Syne"),
            title_text="Segment Profiles (Radar)",
            margin=dict(l=40,r=40,t=60,b=20),height=370)
        st.plotly_chart(fig13,use_container_width=True)

    fig14=px.violin(df_seg,x="segment_label",y="final_grade",color="segment_label",
        box=True,points="outliers",
        color_discrete_map={"High Performer":"#00d4ff","Average":"#ffd166","Low Performer":"#ff6b6b"})
    fig14.update_layout(title_text="Grade Distribution per Segment",showlegend=False)
    st.plotly_chart(sf(fig14),use_container_width=True)

# ════════════════════════════════════════════════
# TAB 6 — PREDICT
# ════════════════════════════════════════════════
with tabs[6]:
    section("🎯","ic-green","Predict a Student Performance","Enter values and get an instant AI prediction")
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('<p class="sub-label" style="color:#00d4ff;">&#128202; Academic Features</p>',unsafe_allow_html=True)
        prev_grade =st.slider("Previous Grade",30,100,70)
        attend_pct =st.slider("Attendance Percentage",40,100,80)
        assignments=st.slider("Assignments Done",1,10,7)
    with c2:
        st.markdown('<p class="sub-label" style="color:#7b61ff;">&#128187; LMS / Digital Features</p>',unsafe_allow_html=True)
        lms_clicks =st.slider("LMS Clicks",20,500,250)
        forum_part =st.slider("Forum Posts",0,25,10)
        internet   =st.slider("Internet Usage Hours",1.0,8.0,3.0,0.1)
    with c3:
        st.markdown('<p class="sub-label" style="color:#00ff9d;">&#129504; Behavioral / Lifestyle</p>',unsafe_allow_html=True)
        study_time =st.slider("Study Time Hours Per Day",1,10,5)
        sleep_hrs  =st.slider("Sleep Hours",4.0,9.0,7.0,0.1)

    if st.button("&#128640;  Generate Prediction",use_container_width=True):
        mx=df.max(numeric_only=True)
        eng=(lms_clicks/mx["lms_clicks"]*0.4+forum_part/mx["forum_participation"]*0.3+
             assignments/mx["assignments_completed"]*0.3)
        con=(attend_pct/100*0.5+study_time/mx["study_time_hours"]*0.5)
        lif=(sleep_hrs/mx["sleep_hours"]*0.5-internet/mx["internet_usage_hours"]*0.5)
        inp=np.array([[study_time,attend_pct,assignments,lms_clicks,
                       forum_part,sleep_hrs,internet,prev_grade,eng,con,lif]])
        bm=model_res[best_name]
        inp_t=bm["scaler"].transform(inp) if bm["scaler"] else inp
        pred=float(np.clip(bm["model"].predict(inp_t)[0],0,100))
        if pred>=75:   label,lc="Distinction",  "#00d4ff"
        elif pred>=50: label,lc="Pass",          "#00ff9d"
        else:          label,lc="Fail",          "#ff6b6b"
        risk_str=("High Risk" if pred<40 else "Medium Risk" if pred<55 else "Low Risk")
        _,mid,_=st.columns([1,2,1])
        with mid:
            st.markdown(
                f'<div class="pred-box">'
                f'<div class="pred-lbl">Predicted Final Grade</div>'
                f'<div class="pred-val">{pred:.1f}</div>'
                f'<div class="pred-cat" style="color:{lc};">{label}</div>'
                f'<div class="pred-meta">{risk_str}</div>'
                f'<div class="pred-meta">Model: {best_name} &nbsp;|&nbsp; '
                f'Engagement: {eng:.3f} &nbsp;|&nbsp; Consistency: {con:.3f}</div></div>',
                unsafe_allow_html=True)
        fig15=go.Figure(go.Indicator(
            mode="gauge+number",value=pred,domain={"x":[0,1],"y":[0,1]},
            title={"text":"Predicted Grade","font":{"color":TCOL,"family":"Syne","size":14}},
            number={"font":{"color":TCOL,"family":"Syne","size":52}},
            gauge={"axis":{"range":[0,100],"tickcolor":TCOL,"tickfont":{"color":TCOL}},
                   "bar":{"color":"#00d4ff"},"bgcolor":PBG,"bordercolor":GRID,
                   "steps":[{"range":[0,40],"color":"rgba(255,107,107,.18)"},
                             {"range":[40,75],"color":"rgba(255,209,102,.12)"},
                             {"range":[75,100],"color":"rgba(0,212,255,.12)"}],
                   "threshold":{"line":{"color":"#7b61ff","width":3},"thickness":.8,"value":pred}}))
        fig15.update_layout(paper_bgcolor=PBG,font=dict(color=TCOL,family="Syne"),
            height=290,margin=dict(l=30,r=30,t=30,b=10))
        st.plotly_chart(fig15,use_container_width=True)

# ════════════════════════════════════════════════
# TAB 7 — ABOUT
# ════════════════════════════════════════════════
with tabs[7]:
    section("📋","ic-gold","Project Documentation")
    c1,c2=st.columns(2)
    with c1:
        st.markdown(
            '<div class="ic"><h3>&#129504; Problem Statement</h3>'
            '<p>Traditional academic systems rely only on marks. This project improves prediction accuracy '
            'by incorporating learning behavior, online activity, and lifestyle patterns &mdash; '
            'creating a holistic view of each student\'s academic trajectory.</p></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ic"><h3>&#129302; ML Objective</h3>'
            '<p><strong style="color:#00d4ff">Regression:</strong> Predict exact final grade (e.g., 75.4, 82.1)<br>'
            '<strong style="color:#7b61ff">Classification:</strong> Predict category &mdash; Pass / Fail / Distinction<br>'
            '<strong style="color:#00ff9d">Clustering:</strong> Segment students into High / Average / Low performer groups</p></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ic"><h3>&#9888;&#65039; Limitations</h3>'
            '<p>&bull; Synthetic dataset &mdash; does not reflect real-world noise<br>'
            '&bull; Limited features &mdash; no demographic or socioeconomic factors<br>'
            '&bull; No real-time data integration<br>'
            '&bull; Clustering is unsupervised &mdash; segment labels are post-hoc</p></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="ic"><h3>&#128202; Dataset Summary</h3>'
            '<p>3 000 student records &nbsp;&middot;&nbsp; 9 raw features &nbsp;&middot;&nbsp; 3 engineered features<br><br>'
            '<strong style="color:#00d4ff">Academic:</strong> previous_grade, final_grade<br>'
            '<strong style="color:#7b61ff">Behavioral:</strong> study_time_hours, assignments_completed, forum_participation<br>'
            '<strong style="color:#00ff9d">LMS / Digital:</strong> lms_clicks, internet_usage_hours<br>'
            '<strong style="color:#ffd166">Lifestyle:</strong> sleep_hours, attendance_percentage<br>'
            '<strong style="color:#ff6b6b">Engineered:</strong> engagement_score, consistency_index, lifestyle_score</p></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ic"><h3>&#127807; Future Scope</h3>'
            '<p>&bull; Real-time LMS integration via API hooks<br>'
            '&bull; AI-based personalised learning recommendations<br>'
            '&bull; SHAP explainability: Why did this student score low?<br>'
            '&bull; Longitudinal tracking across semesters<br>'
            '&bull; Mobile push alerts for at-risk students</p></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ic"><h3>&#128736;&#65039; Tech Stack</h3>'
            '<p>Python &nbsp;&middot;&nbsp; Streamlit &nbsp;&middot;&nbsp; Scikit-learn &nbsp;&middot;&nbsp; XGBoost<br>'
            'Plotly &nbsp;&middot;&nbsp; Pandas &nbsp;&middot;&nbsp; NumPy &nbsp;&middot;&nbsp; KMeans Clustering<br>'
            'Random Forest &nbsp;&middot;&nbsp; Linear Regression &nbsp;&middot;&nbsp; Logistic Regression</p></div>',
            unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">ACADEMIC PERFORMANCE PREDICTION'
        ' &nbsp;&middot;&nbsp; MULTISOURCE BEHAVIORAL DATA'
        ' &nbsp;&middot;&nbsp; AI DASHBOARD v2.1</div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 8 — ABOUT
# ═══════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header"><div class="icon icon-gold">📋</div><h2>Project Documentation</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
          <h3>🧠 Problem Statement</h3>
          <p>Traditional academic systems rely only on marks. This project improves prediction accuracy
          by incorporating learning behavior, online activity, and lifestyle patterns — creating a
          holistic view of each student's academic trajectory.</p>
        </div>
        <div class="info-card">
          <h3>🤖 ML Objective</h3>
          <p><strong>Regression:</strong> Predict exact final grade (e.g., 75.4, 82.1)<br>
          <strong>Classification:</strong> Predict category — Pass / Fail / Distinction<br>
          <strong>Clustering:</strong> Segment students into High / Average / Low performer groups</p>
        </div>
        <div class="info-card">
          <h3>⚠️ Limitations</h3>
          <p>• Synthetic dataset — does not reflect real-world noise<br>
          • Limited features — no demographic or socioeconomic factors<br>
          • No real-time data integration<br>
          • Clustering is unsupervised — segment labels are post-hoc</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
          <h3>📊 Dataset Summary</h3>
          <p>3,000 student records · 9 raw features · 3 engineered features<br>
          <strong>Academic:</strong> previous_grade, final_grade<br>
          <strong>Behavioral:</strong> study_time_hours, assignments_completed, forum_participation<br>
          <strong>LMS/Digital:</strong> lms_clicks, internet_usage_hours<br>
          <strong>Lifestyle:</strong> sleep_hours, attendance_percentage<br>
          <strong>Engineered:</strong> engagement_score, consistency_index, lifestyle_score</p>
        </div>
        <div class="info-card">
          <h3>🌱 Future Scope</h3>
          <p>• Real-time LMS integration via API hooks<br>
          • AI-based personalized learning recommendations<br>
          • SHAP explainability: "Why did this student score low?"<br>
          • Longitudinal tracking across semesters<br>
          • Mobile push alerts for at-risk students</p>
        </div>
        <div class="info-card">
          <h3>🛠️ Tech Stack</h3>
          <p>Python · Streamlit · Scikit-learn · XGBoost<br>
          Plotly · Pandas · NumPy · KMeans Clustering<br>
          Random Forest · Linear Regression · Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 2rem; color: #1a3050;
                font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
                letter-spacing: 0.15em; margin-top: 2rem;">
      ACADEMIC PERFORMANCE PREDICTION · MULTISOURCE BEHAVIORAL DATA · AI DASHBOARD
    </div>
    """, unsafe_allow_html=True)
