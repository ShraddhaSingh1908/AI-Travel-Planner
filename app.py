import streamlit as st
import json
import datetime
import requests
import pandas as pd
import pydeck as pdk
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wanderlust Agentic AI | IBM Granite Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(160deg, #f0f4ff 0%, #e8f0fe 50%, #f5f7ff 100%);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5ff 100%);
        border-right: 1px solid #dde3f0;
    }
    [data-testid="stSidebar"] * { color: #1e293b !important; }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] hr { border-color: #dde3f0 !important; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f3460 0%, #0f62fe 50%, #1192e8 100%);
        padding: 3.5rem 2.5rem 3rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(15,98,254,0.35), 0 0 0 1px rgba(15,98,254,0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -80px; right: -80px;
        width: 320px; height: 320px;
        background: rgba(255,255,255,0.07);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -100px; left: -50px;
        width: 280px; height: 280px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
        letter-spacing: -1px;
        text-shadow: 0 2px 12px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    .hero-tagline {
        font-size: 1.1rem;
        opacity: 0.92;
        font-weight: 400;
        letter-spacing: 0.3px;
        position: relative;
        z-index: 1;
    }
    .hero-badges {
        margin-top: 1.4rem;
        display: flex;
        justify-content: center;
        gap: 0.65rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.30);
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        backdrop-filter: blur(4px);
        transition: background 0.2s;
    }

    /* ── Glass card ── */
    .glass-card {
        background: #ffffff;
        border: 1px solid #dde3f0;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        box-shadow: 0 4px 20px rgba(15,98,254,0.07);
        margin-bottom: 1.2rem;
    }

    /* ── Section headers ── */
    .section-header {
        color: #0f172a;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-sub {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 1.2rem;
    }

    /* ── Quick-pick chip buttons (scoped to stHorizontalBlock to avoid overriding generate btn) ── */
    [data-testid="stHorizontalBlock"] [data-testid="stButton"] > button {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #1e40af !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 0.7rem !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
        box-shadow: 0 1px 4px rgba(15,98,254,0.07) !important;
    }
    [data-testid="stHorizontalBlock"] [data-testid="stButton"] > button:hover {
        background: #eff6ff !important;
        border-color: #0f62fe !important;
        color: #0f62fe !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(15,98,254,0.18) !important;
    }

    /* ── Form inputs ── */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 10px !important;
        caret-color: #0f62fe !important;
        font-size: 0.93rem !important;
    }
    input::placeholder, textarea::placeholder { color: #94a3b8 !important; }
    input:focus, textarea:focus {
        border-color: #0f62fe !important;
        box-shadow: 0 0 0 3px rgba(15,98,254,0.12) !important;
        outline: none !important;
        background: #f8faff !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #1e293b !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }
    [data-testid="stNumberInput"] div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }
    [data-testid="stNumberInput"] input {
        color: #1e293b !important;
        background: transparent !important;
        border: none !important;
    }
    label, .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #475569 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
    }

    /* ── Generate button ── */
    div.stButton > button[kind="primary"],
    div.stButton > button:only-child {
        background: linear-gradient(90deg, #0f3460 0%, #0f62fe 60%, #1192e8 100%) !important;
        color: white !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 2rem !important;
        border-radius: 14px !important;
        border: none !important;
        letter-spacing: 0.4px !important;
        box-shadow: 0 8px 24px rgba(15,98,254,0.35) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button:only-child:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 14px 32px rgba(15,98,254,0.45) !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1.5px solid #dde3f0 !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.3rem !important;
        box-shadow: 0 2px 12px rgba(15,98,254,0.06) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important; font-size: 0.78rem !important;
        font-weight: 600 !important; text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important; font-size: 1.25rem !important; font-weight: 700 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9 !important;
        border-radius: 14px !important;
        padding: 5px !important;
        gap: 4px !important;
        border: 1px solid #dde3f0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.87rem !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f62fe, #1192e8) !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(15,98,254,0.3) !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

    /* ── Itinerary result card ── */
    .result-container {
        background: #ffffff;
        padding: 2rem 2.4rem;
        border-radius: 18px;
        border: 1.5px solid #dde3f0;
        border-left: 5px solid #0f62fe;
        box-shadow: 0 4px 24px rgba(15,98,254,0.08);
        margin-top: 1rem;
        color: #1e293b;
        line-height: 1.75;
    }
    .result-container h1, .result-container h2,
    .result-container h3, .result-container h4 { color: #0f62fe !important; }
    .result-container p, .result-container li { color: #334155 !important; }
    .result-container strong { color: #1e293b !important; }

    /* ── Destination hero image ── */
    .dest-hero {
        position: relative;
        border-radius: 22px;
        overflow: hidden;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    .dest-hero img {
        width: 100%; height: 340px;
        object-fit: cover; display: block;
        transition: transform 0.4s ease;
    }
    .dest-hero:hover img { transform: scale(1.02); }
    .dest-hero-overlay {
        position: absolute; inset: 0;
        background: linear-gradient(to top, rgba(5,10,40,0.88) 0%, rgba(5,10,40,0.2) 45%, transparent 70%);
    }
    .dest-hero-text {
        position: absolute; bottom: 1.8rem; left: 2rem; right: 2rem;
    }
    .dest-hero-title {
        font-size: 2.2rem; font-weight: 800; color: white;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5); margin-bottom: 0.3rem;
    }
    .dest-hero-sub {
        font-size: 0.98rem; color: rgba(255,255,255,0.78);
        display: flex; gap: 0.8rem; flex-wrap: wrap; align-items: center;
    }
    .dest-hero-tag {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.2rem 0.75rem; border-radius: 20px;
        font-size: 0.82rem; font-weight: 600;
        backdrop-filter: blur(4px);
    }

    /* ── Weather cards ── */
    .weather-card {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1.5px solid #bfdbfe;
        border-radius: 16px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .weather-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(15,98,254,0.15);
    }
    .weather-card .wc-label {
        font-size: 0.75rem; color: #64748b; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 0.5rem;
    }
    .weather-card .wc-value { font-size: 1.7rem; font-weight: 800; color: #1d4ed8; }
    .weather-card .wc-unit { font-size: 0.88rem; color: #64748b; }

    /* ── Info box ── */
    .info-box {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #bae6fd;
        border-left: 4px solid #0ea5e9;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #0c4a6e;
        font-size: 0.9rem;
        margin: 0.8rem 0;
    }

    /* ── Packing checklist ── */
    .stCheckbox label { color: #334155 !important; font-size: 0.92rem !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

    /* ── Text ── */
    h1, h2, h3, h4 { color: #0f172a !important; }
    p, .stMarkdown p { color: #334155 !important; }
    .stCaption { color: #94a3b8 !important; font-size: 0.82rem !important; }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] button {
        background: #eff6ff !important;
        border: 1.5px solid #bfdbfe !important;
        color: #1d4ed8 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #dbeafe !important;
        border-color: #0f62fe !important;
        box-shadow: 0 4px 12px rgba(15,98,254,0.18) !important;
    }

    /* ── Chat ── */
    [data-testid="stChatMessage"] {
        background: #f8faff !important;
        border: 1px solid #dde3f0 !important;
        border-radius: 14px !important;
        margin-bottom: 0.7rem !important;
    }
    [data-testid="stChatInputTextArea"] {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 12px !important;
    }

    /* ── Divider & scrollbar ── */
    hr { border-color: #dde3f0 !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🌍 Wanderlust Agentic AI</div>
        <div class="hero-tagline">
            Autonomous Travel Planning &nbsp;·&nbsp; Visual Mapping &nbsp;·&nbsp; Live Weather<br>
            Powered by <b>IBM Granite 4</b>
        </div>
        <div class="hero-badges">
            <span class="hero-badge">✈️ Smart Itineraries</span>
            <span class="hero-badge">🗺️ Interactive Maps</span>
            <span class="hero-badge">🌦️ Live Weather</span>
            <span class="hero-badge">💬 AI Concierge</span>
            <span class="hero-badge">🤖 IBM Granite 4</span>
            <span class="hero-badge">🧳 Packing Lists</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=90)
    st.markdown("## ⚙️ Engine Control")
    st.caption("AICTE 2026 · Problem Statement 5")
    st.markdown("---")

    st.markdown("### 🔑 IBM Credentials")

    # ── Secure secrets loading from .streamlit/secrets.toml ──────────────
    default_key     = st.secrets.get("IBM_API_KEY", "")
    default_pid     = st.secrets.get("WATSONX_PROJECT_ID", "")
    weather_api_key = st.secrets.get("OPENWEATHER_API_KEY", "")

    api_key    = st.text_input("IBM IAM API Key", value=default_key, type="password",
                               help="Store as IBM_API_KEY in .streamlit/secrets.toml")
    project_id = st.text_input("watsonx Project ID", value=default_pid, type="password",
                               help="Store as WATSONX_PROJECT_ID in .streamlit/secrets.toml")
    region     = st.selectbox("Cloud Region", ["us-south", "eu-de"], index=0)

    # ── Credential status indicators ─────────────────────────────────────
    st.markdown("**🔒 Secrets Status**")
    _ibm_src     = "secrets.toml ✅" if default_key     else "manual input"
    _pid_src     = "secrets.toml ✅" if default_pid     else "manual input"
    _wx_src      = "secrets.toml ✅" if weather_api_key else "not configured ⚠️"
    _ibm_ok      = "🟢" if api_key.strip()    else "🔴"
    _pid_ok      = "🟢" if project_id.strip() else "🔴"
    _wx_ok       = "🟢" if weather_api_key    else "🟡"
    st.markdown(f"""
        <div style='font-size:0.78rem; line-height:2; color:#334155;
                    background:#f8faff; border:1px solid #dde3f0;
                    border-radius:10px; padding:0.7rem 0.9rem;'>
            {_ibm_ok} <b>IBM API Key</b> — {_ibm_src}<br>
            {_pid_ok} <b>Project ID</b> — {_pid_src}<br>
            {_wx_ok} <b>OpenWeather</b> — {_wx_src}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Agent Parameters")
    temperature = st.slider("Creativity (Temperature)", 0.1, 1.0, 0.7, 0.1)
    max_tokens  = st.slider("Max Output Tokens", 300, 2500, 1500, 100)

    st.markdown("---")
    st.markdown("""
        <div style='font-size:0.73rem; color:#64748b; line-height:1.9;'>
            🔒 Credentials stored in<br>
            &nbsp;&nbsp;<code>.streamlit/secrets.toml</code><br>
            🤖 Model: <b>ibm/granite-4-h-small</b><br>
            🌐 Weather: OpenWeatherMap API<br>
            🖼️ Images: Unsplash / Wikimedia<br>
            💰 Budget: auto-allocated across<br>
            &nbsp;&nbsp;accommodation · food · transport<br>
            &nbsp;&nbsp;local travel · sightseeing
        </div>
    """, unsafe_allow_html=True)

# --- DESTINATION DATA ---
DESTINATION_COORDS = {
    "Goa": {
        "lat": 15.2993, "lon": 74.1240,
        "image": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1200&q=80",
        "spots": [
            {"name": "Baga Beach",     "lat": 15.5553, "lon": 73.7517},
            {"name": "Aguada Fort",    "lat": 15.4920, "lon": 73.7737},
            {"name": "Dudhsagar Falls","lat": 15.3144, "lon": 74.3143},
        ]
    },
    "Manali": {
        "lat": 32.2432, "lon": 77.1892,
        "image": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1200&q=80",
        "spots": [
            {"name": "Solang Valley",  "lat": 32.3166, "lon": 77.1578},
            {"name": "Hadimba Temple", "lat": 32.2483, "lon": 77.1804},
            {"name": "Rohtang Pass",   "lat": 32.3716, "lon": 77.2466},
        ]
    },
    "Jaipur": {
        "lat": 26.9124, "lon": 75.7873,
        "image": "https://images.unsplash.com/photo-1477587458883-47145ed94245?w=1200&q=80",
        "spots": [
            {"name": "Hawa Mahal",  "lat": 26.9239, "lon": 75.8267},
            {"name": "Amer Fort",   "lat": 26.9855, "lon": 75.8513},
            {"name": "City Palace", "lat": 26.9258, "lon": 75.8237},
        ]
    },
    "Kerala": {
        "lat": 9.9312, "lon": 76.2673,
        "image": "https://images.unsplash.com/photo-1593693397690-362cb9666fc2?w=1200&q=80",
        "spots": [
            {"name": "Alleppey Backwaters","lat": 9.4981,  "lon": 76.3388},
            {"name": "Munnar Tea Gardens", "lat": 10.0889, "lon": 77.0595},
            {"name": "Kochi Fort",         "lat": 9.9656,  "lon": 76.2421},
        ]
    },
    "Mumbai": {
        "lat": 19.0760, "lon": 72.8777,
        "image": "https://images.unsplash.com/photo-1529253355930-ddbe423a2ac7?w=1200&q=80",
        "spots": [
            {"name": "Gateway of India","lat": 18.9220, "lon": 72.8347},
            {"name": "Marine Drive",    "lat": 18.9440, "lon": 72.8237},
            {"name": "Elephanta Caves", "lat": 18.9633, "lon": 72.9315},
        ]
    },
    "Agra": {
        "lat": 27.1767, "lon": 78.0081,
        "image": "https://images.unsplash.com/photo-1564507592333-c60657eea523?w=1200&q=80",
        "spots": [
            {"name": "Taj Mahal",       "lat": 27.1751, "lon": 78.0421},
            {"name": "Agra Fort",       "lat": 27.1795, "lon": 78.0211},
            {"name": "Fatehpur Sikri",  "lat": 27.0945, "lon": 77.6695},
        ]
    },
    "Varanasi": {
        "lat": 25.3176, "lon": 82.9739,
        "image": "https://images.unsplash.com/photo-1570459027562-4a916cc6113f?w=1200&q=80",
        "spots": [
            {"name": "Dashashwamedh Ghat",      "lat": 25.3068, "lon": 83.0109},
            {"name": "Kashi Vishwanath Temple", "lat": 25.3109, "lon": 83.0107},
            {"name": "Sarnath",                 "lat": 25.3814, "lon": 83.0234},
        ]
    },
    "Leh Ladakh": {
        "lat": 34.1526, "lon": 77.5771,
        "image": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=1200&q=80",
        "spots": [
            {"name": "Pangong Lake", "lat": 33.7608, "lon": 78.6445},
            {"name": "Leh Palace",   "lat": 34.1642, "lon": 77.5857},
            {"name": "Nubra Valley", "lat": 34.6494, "lon": 77.5619},
        ]
    },
}

# Wikimedia Commons search — returns a real photo for any destination name (no API key needed)
def _destination_image_url(dest_name: str) -> str:
    """Try Wikimedia Commons API for a destination image; fall back to Unsplash source search."""
    try:
        query = dest_name.replace(" ", "%20")
        api_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&titles={query}"
            f"&prop=pageimages&format=json&pithumbsize=1200&origin=*"
        )
        r = requests.get(api_url, timeout=4)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                thumb = page.get("thumbnail", {}).get("source", "")
                if thumb:
                    return thumb
    except Exception:
        pass
    # Fallback: Unsplash source — keyword-based, returns a relevant image for the destination
    safe = dest_name.replace(" ", "-").lower()
    return f"https://source.unsplash.com/1200x600/?{safe},travel,india"


# --- MOCK APIs ---
def _add_minutes(time_str: str, mins: int) -> str:
    """Add minutes to a HH:MM AM/PM string and return arrival time."""
    from datetime import datetime, timedelta
    t = datetime.strptime(time_str, "%I:%M %p")
    t += timedelta(minutes=mins)
    return t.strftime("%I:%M %p").lstrip("0")

# ── City name normalisation ──────────────────────────────────────────────────
# Maps common spellings / partial names → canonical key used in route tables
_CITY_ALIASES = {
    # Delhi variants
    "delhi": "delhi", "new delhi": "delhi", "ndls": "delhi",
    # Mumbai
    "mumbai": "mumbai", "bombay": "mumbai",
    # Kolkata
    "kolkata": "kolkata", "calcutta": "kolkata", "howrah": "kolkata",
    # Chennai
    "chennai": "chennai", "madras": "chennai",
    # Bangalore / Bengaluru
    "bangalore": "bangalore", "bengaluru": "bangalore",
    # Hyderabad
    "hyderabad": "hyderabad", "secunderabad": "hyderabad",
    # Ahmedabad
    "ahmedabad": "ahmedabad",
    # Pune
    "pune": "pune",
    # Jaipur
    "jaipur": "jaipur",
    # Agra
    "agra": "agra",
    # Varanasi / Prayagraj
    "varanasi": "varanasi", "banaras": "varanasi", "kashi": "varanasi",
    "prayagraj": "prayagraj", "allahabad": "prayagraj",
    # Lucknow
    "lucknow": "lucknow",
    # Kanpur
    "kanpur": "kanpur",
    # Patna
    "patna": "patna",
    # Bhopal
    "bhopal": "bhopal",
    # Indore
    "indore": "indore",
    # Nagpur
    "nagpur": "nagpur",
    # Raipur
    "raipur": "raipur",
    # Goa
    "goa": "goa", "panaji": "goa", "madgaon": "goa", "margao": "goa",
    # Kerala
    "kerala": "kerala", "kochi": "kerala", "trivandrum": "kerala",
    "thiruvananthapuram": "kerala", "ernakulam": "kerala",
    # Manali / Shimla / Chandigarh
    "manali": "manali", "shimla": "shimla",
    "chandigarh": "chandigarh",
    # Amritsar / Punjab
    "amritsar": "amritsar", "punjab": "amritsar", "ludhiana": "amritsar",
    # Leh / Jammu / Srinagar
    "leh": "leh ladakh", "leh ladakh": "leh ladakh",
    "jammu": "jammu", "srinagar": "srinagar",
    # Rishikesh / Dehradun / Haridwar
    "rishikesh": "rishikesh", "dehradun": "dehradun", "haridwar": "rishikesh",
    # Udaipur / Jodhpur / Ajmer
    "udaipur": "udaipur", "jodhpur": "jodhpur", "ajmer": "ajmer",
}

def _canon(city: str) -> str:
    """Return canonical city key for route lookup."""
    c = city.strip().lower()
    if c in _CITY_ALIASES:
        return _CITY_ALIASES[c]
    # partial match
    for alias, canon in _CITY_ALIASES.items():
        if alias in c or c in alias:
            return canon
    return c  # unknown city — return as-is

# ── Route-specific FLIGHT data ───────────────────────────────────────────────
# Key: (origin_canon, dest_canon)
# Tuple: (Airline, FlightNo, Departure, duration_minutes, fare_INR)
_FLIGHT_ROUTES = {
    ("delhi","goa"):        [("IndiGo","6E-401","07:30 AM",130,4800),("Air India","AI-203","10:45 AM",135,5600),("Akasa Air","QP-112","06:00 PM",125,4200)],
    ("delhi","mumbai"):     [("IndiGo","6E-221","06:00 AM",120,4200),("Air India","AI-809","01:15 PM",125,5600),("Vistara","UK-951","08:30 PM",120,5200)],
    ("delhi","jaipur"):     [("IndiGo","6E-321","07:00 AM",60,3200),("Air India","AI-105","11:00 AM",65,3800),("Akasa Air","QP-305","05:30 PM",60,3000)],
    ("delhi","kerala"):     [("IndiGo","6E-551","06:30 AM",195,6500),("Air India","AI-677","01:00 PM",200,7800),("Vistara","UK-891","08:00 PM",190,7200)],
    ("delhi","varanasi"):   [("IndiGo","6E-881","06:15 AM",100,3800),("Air India","AI-411","11:30 AM",105,4500),("Vistara","UK-231","05:45 PM",100,4200)],
    ("delhi","agra"):       [("IndiGo","6E-211","08:00 AM",45,2500),("Air India","AI-301","12:00 PM",50,3000),("SpiceJet","SG-101","04:00 PM",45,2200)],
    ("delhi","leh ladakh"): [("IndiGo","6E-991","05:30 AM",80,6500),("Air India","AI-445","09:00 AM",85,7800),("GoFirst","G8-301","01:00 PM",80,6000)],
    ("delhi","amritsar"):   [("IndiGo","6E-613","07:20 AM",80,3500),("Air India","AI-113","12:45 PM",85,4200),("Akasa Air","QP-201","05:00 PM",80,3200)],
    ("delhi","shimla"):     [("IndiGo","6E-711","06:00 AM",75,3200),("Air India","AI-441","12:30 PM",80,3800),("SpiceJet","SG-201","04:15 PM",75,3000)],
    ("delhi","manali"):     [("IndiGo","6E-711","06:00 AM",75,3500),("Air India","AI-441","12:30 PM",80,4200),("SpiceJet","SG-201","04:15 PM",75,3200)],
    ("delhi","rishikesh"):  [("IndiGo","6E-731","06:30 AM",70,3000),("Air India","AI-461","11:00 AM",75,3600),("SpiceJet","SG-211","04:30 PM",70,2800)],
    ("delhi","udaipur"):    [("IndiGo","6E-451","07:00 AM",90,3800),("Air India","AI-333","12:00 PM",95,4500),("Akasa Air","QP-411","05:15 PM",90,3500)],
    ("delhi","kolkata"):    [("IndiGo","6E-331","06:45 AM",130,4800),("Air India","AI-021","12:15 PM",135,5600),("Vistara","UK-771","07:30 PM",130,5200)],
    ("delhi","chennai"):    [("IndiGo","6E-771","06:00 AM",165,5500),("Air India","AI-561","11:30 AM",170,6500),("Vistara","UK-861","08:00 PM",165,6000)],
    ("delhi","hyderabad"):  [("IndiGo","6E-661","06:30 AM",135,4800),("Air India","AI-541","12:00 PM",140,5600),("Akasa Air","QP-501","05:45 PM",135,4500)],
    ("delhi","bangalore"):  [("IndiGo","6E-501","06:00 AM",165,5200),("Air India","AI-501","11:00 AM",170,6200),("Vistara","UK-601","08:30 PM",165,5800)],
    ("delhi","raipur"):     [("IndiGo","6E-441","07:00 AM",120,4500),("Air India","AI-231","01:00 PM",125,5200),("SpiceJet","SG-341","05:30 PM",120,4200)],
    ("delhi","bhopal"):     [("IndiGo","6E-271","07:30 AM",90,3800),("Air India","AI-221","12:30 PM",95,4500),("SpiceJet","SG-271","05:00 PM",90,3500)],
    ("delhi","indore"):     [("IndiGo","6E-281","07:00 AM",95,3800),("Air India","AI-241","12:00 PM",100,4500),("Akasa Air","QP-281","05:30 PM",95,3500)],
    ("delhi","nagpur"):     [("IndiGo","6E-311","07:15 AM",110,4200),("Air India","AI-261","12:15 PM",115,5000),("SpiceJet","SG-311","05:45 PM",110,3900)],
    ("delhi","patna"):      [("IndiGo","6E-351","06:30 AM",95,3800),("Air India","AI-311","11:30 AM",100,4500),("Akasa Air","QP-351","04:45 PM",95,3500)],
    ("delhi","lucknow"):    [("IndiGo","6E-361","07:00 AM",75,3200),("Air India","AI-321","12:00 PM",80,3800),("Vistara","UK-361","05:30 PM",75,3500)],
    ("mumbai","goa"):       [("IndiGo","6E-111","06:30 AM",65,2800),("Air India","AI-619","01:00 PM",70,3500),("Akasa Air","QP-081","07:00 PM",65,2500)],
    ("mumbai","delhi"):     [("IndiGo","6E-222","06:00 AM",120,4200),("Air India","AI-810","01:00 PM",125,5200),("Vistara","UK-952","08:00 PM",120,4800)],
    ("mumbai","kolkata"):   [("IndiGo","6E-432","07:00 AM",165,5200),("Air India","AI-022","12:30 PM",170,6200),("Vistara","UK-432","07:30 PM",165,5800)],
    ("mumbai","bangalore"): [("IndiGo","6E-502","06:30 AM",75,3000),("Air India","AI-502","12:00 PM",80,3800),("Vistara","UK-502","07:00 PM",75,3500)],
    ("mumbai","hyderabad"): [("IndiGo","6E-602","07:00 AM",75,3000),("Air India","AI-542","12:30 PM",80,3800),("Akasa Air","QP-402","05:30 PM",75,2800)],
    ("mumbai","chennai"):   [("IndiGo","6E-772","06:30 AM",105,3800),("Air India","AI-562","12:00 PM",110,4500),("Vistara","UK-772","07:30 PM",105,4200)],
    ("mumbai","raipur"):    [("IndiGo","6E-442","07:30 AM",105,3800),("Air India","AI-232","01:30 PM",110,4500),("SpiceJet","SG-342","06:00 PM",105,3500)],
    ("kolkata","delhi"):    [("IndiGo","6E-332","07:00 AM",130,4800),("Air India","AI-022","12:00 PM",135,5600),("Vistara","UK-332","07:00 PM",130,5200)],
    ("kolkata","mumbai"):   [("IndiGo","6E-433","06:30 AM",165,5200),("Air India","AI-023","12:00 PM",170,6200),("Vistara","UK-433","07:00 PM",165,5800)],
    ("kolkata","chennai"):  [("IndiGo","6E-773","07:00 AM",150,4800),("Air India","AI-563","12:30 PM",155,5600),("Vistara","UK-773","07:30 PM",150,5200)],
    ("kolkata","bangalore"):[("IndiGo","6E-503","06:30 AM",165,5200),("Air India","AI-503","12:00 PM",170,6200),("Vistara","UK-503","07:30 PM",165,5800)],
    ("prayagraj","raipur"): [("IndiGo","6E-521","09:00 AM",90,4200),("Air India","AI-421","02:00 PM",95,5000),("SpiceJet","SG-421","06:30 PM",90,3800)],
    ("prayagraj","delhi"):  [("IndiGo","6E-362","08:00 AM",75,3200),("Air India","AI-322","01:00 PM",80,3800),("Vistara","UK-362","06:00 PM",75,3500)],
    ("prayagraj","mumbai"): [("IndiGo","6E-444","07:30 AM",130,4800),("Air India","AI-234","01:30 PM",135,5600),("SpiceJet","SG-444","06:30 PM",130,4500)],
    ("hyderabad","delhi"):  [("IndiGo","6E-662","07:00 AM",135,4800),("Air India","AI-542","12:30 PM",140,5600),("Vistara","UK-662","07:30 PM",135,5200)],
    ("hyderabad","mumbai"): [("IndiGo","6E-603","07:30 AM",75,3000),("Air India","AI-543","01:00 PM",80,3800),("Akasa Air","QP-403","06:00 PM",75,2800)],
    ("hyderabad","kolkata"):[("IndiGo","6E-603","07:00 AM",130,4500),("Air India","AI-543","12:00 PM",135,5200),("Vistara","UK-603","07:00 PM",130,4800)],
    ("chennai","delhi"):    [("IndiGo","6E-772","06:30 AM",165,5500),("Air India","AI-562","12:00 PM",170,6500),("Vistara","UK-772","07:30 PM",165,6000)],
    ("chennai","mumbai"):   [("IndiGo","6E-773","07:00 AM",105,3800),("Air India","AI-563","12:00 PM",110,4500),("Vistara","UK-773","07:30 PM",105,4200)],
    ("bangalore","delhi"):  [("IndiGo","6E-502","06:00 AM",165,5200),("Air India","AI-502","11:30 AM",170,6200),("Vistara","UK-502","08:00 PM",165,5800)],
    ("bangalore","mumbai"): [("IndiGo","6E-503","06:30 AM",75,3000),("Air India","AI-503","12:30 PM",80,3800),("Vistara","UK-503","07:30 PM",75,3500)],
    ("bangalore","chennai"):[("IndiGo","6E-504","07:00 AM",60,2800),("Air India","AI-504","12:00 PM",65,3500),("Vistara","UK-504","06:30 PM",60,3000)],
    ("raipur","delhi"):     [("IndiGo","6E-442","08:00 AM",120,4500),("Air India","AI-232","01:00 PM",125,5200),("SpiceJet","SG-342","06:30 PM",120,4200)],
    ("raipur","mumbai"):    [("IndiGo","6E-443","07:30 AM",105,3800),("Air India","AI-233","01:30 PM",110,4500),("SpiceJet","SG-343","06:00 PM",105,3500)],
    ("raipur","kolkata"):   [("IndiGo","6E-445","08:30 AM",90,3500),("Air India","AI-235","02:00 PM",95,4200),("SpiceJet","SG-345","07:00 PM",90,3200)],
}

def get_flight_estimates(origin, destination, num_pax=1):
    ok = _canon(origin)
    dk = _canon(destination)
    flights_raw = _FLIGHT_ROUTES.get((ok, dk)) or _FLIGHT_ROUTES.get((dk, ok))
    # fallback: find any route involving destination
    if flights_raw is None:
        for (o, d), v in _FLIGHT_ROUTES.items():
            if d == dk or o == dk:
                flights_raw = v
                break
    # final fallback: generic
    if flights_raw is None:
        flights_raw = [
            ("IndiGo",    "6E-XXX", "07:30 AM", 120, 4800),
            ("Air India", "AI-XXX", "01:15 PM", 125, 5600),
            ("Akasa Air", "QP-XXX", "06:45 PM", 115, 4200),
        ]
    result = []
    for airline, fno, dep, dur_mins, fare in flights_raw:
        h, m = divmod(dur_mins, 60)
        dur_str = f"{h}h {m:02d}m" if m else f"{h}h"
        result.append({
            "Airline":     airline,
            "Flight No":   fno,
            "Departure":   dep,
            "Arrival":     _add_minutes(dep, dur_mins),
            "Duration":    dur_str,
            "Fare/Person": f"₹{fare:,}",
            "Total Fare":  f"₹{fare * num_pax:,}",
        })
    return result

# ── Route-specific TRAIN data ────────────────────────────────────────────────
# Key: (origin_canon, dest_canon)
# Tuple: (Train name, No, Departure, duration_minutes, Class, fare_INR)
_TRAIN_ROUTES = {
    # ── Delhi as origin ──
    ("delhi","agra"):        [("Gatimaan Express","12049","08:10 AM",90,"CC / EC",755),("Taj Express","12280","07:15 AM",120,"CC",370),("Shatabdi Express","12001","06:00 AM",120,"CC / EC",680)],
    ("delhi","jaipur"):      [("Ajmer Shatabdi","12015","06:05 AM",270,"CC / EC",755),("Double Decker","12985","03:55 PM",300,"CC",490),("Pink City Express","12450","05:50 AM",330,"Sleeper/3A",340)],
    ("delhi","varanasi"):    [("Vande Bharat Exp","22436","06:00 AM",480,"CC / EC",1760),("Kashi VN Exp","15113","06:00 AM",720,"Sleeper/3A",420),("Mahanagari Exp","11094","07:30 PM",780,"Sleeper/3A",390)],
    ("delhi","mumbai"):      [("Rajdhani Exp","12951","04:55 PM",960,"2A / 1A",2450),("Aug Kranti Raj","12953","05:25 PM",1020,"2A / 3A",2100),("Mumbai Mail","11057","08:00 PM",1380,"Sleeper/3A",490)],
    ("delhi","goa"):         [("Rajdhani Exp","12431","10:55 AM",1740,"2A / 3A",2100),("Goa Express","12779","03:15 PM",1620,"Sleeper/3A",550),("Konkan Kanya","10111","11:00 PM",1560,"Sleeper/2A",480)],
    ("delhi","kerala"):      [("Kerala Express","12625","11:35 AM",2820,"Sleeper/3A",620),("Rajdhani Exp","12431","10:55 AM",2580,"2A / 1A",3200),("Netravati Exp","16345","07:40 AM",2760,"Sleeper/2A",590)],
    ("delhi","kolkata"):     [("Rajdhani Exp","12301","04:55 PM",1020,"2A / 1A",2300),("Poorva Exp","12303","08:45 AM",1320,"Sleeper/3A",520),("Kalka Mail","13005","07:30 PM",1560,"Sleeper/3A",450)],
    ("delhi","amritsar"):    [("Shatabdi Exp","12013","07:20 AM",360,"CC / EC",1085),("Golden Temple Mail","12904","09:25 PM",600,"Sleeper/3A",390),("Amritsar Exp","11057","06:15 PM",540,"Sleeper/2A",450)],
    ("delhi","manali"):      [("Shatabdi Exp","12011","07:20 AM",210,"CC / EC",850),("Himalayan Queen","52455","06:00 AM",300,"CC / FC",320),("Kalka Mail","12137","07:30 PM",360,"Sleeper/3A",380)],
    ("delhi","shimla"):      [("Kalka Shatabdi","12011","07:40 AM",210,"CC / EC",850),("Himalayan Queen","52455","06:10 AM",300,"FC",320),("Kalka Mail","12137","07:30 PM",360,"Sleeper/3A",380)],
    ("delhi","rishikesh"):   [("Dehradun Exp","19019","09:35 PM",420,"Sleeper/3A",360),("Jan Shatabdi","12055","03:10 PM",330,"CC / 2S",480),("Mussoorie Exp","14041","10:00 PM",480,"Sleeper/3A",330)],
    ("delhi","udaipur"):     [("Chetak Exp","09615","06:10 PM",720,"Sleeper/3A",410),("Mewar Exp","12963","07:00 PM",660,"Sleeper/2A",1350),("Shatabdi Exp","12991","06:15 AM",540,"CC / EC",895)],
    ("delhi","lucknow"):     [("Shatabdi Exp","12003","06:10 AM",270,"CC / EC",950),("Lucknow Mail","12229","10:00 PM",480,"Sleeper/3A",390),("Gomti Exp","12419","06:00 AM",360,"Sleeper/3A",310)],
    ("delhi","patna"):       [("Rajdhani Exp","12309","18:55 PM",600,"2A / 1A",1950),("Sampark Kranti","12565","07:00 PM",720,"Sleeper/3A",460),("Bihar Sampark","12565","06:30 AM",780,"Sleeper/3A",430)],
    ("delhi","bhopal"):      [("Shatabdi Exp","12001","06:00 AM",360,"CC / EC",1245),("Rajdhani Exp","12627","08:00 PM",480,"2A / 1A",1850),("Punjab Mail","12137","10:55 PM",540,"Sleeper/3A",420)],
    ("delhi","indore"):      [("Indore Exp","12415","06:05 AM",660,"Sleeper/3A",480),("Malwa Exp","12919","09:55 PM",720,"Sleeper/3A",450),("Rajdhani Exp","12627","08:00 PM",600,"2A / 1A",1950)],
    ("delhi","nagpur"):      [("Vidarbha Exp","12105","06:10 AM",780,"Sleeper/3A",490),("Rajdhani Exp","12671","08:00 PM",660,"2A / 1A",2100),("Sewagram Exp","12809","07:30 PM",840,"Sleeper/3A",510)],
    ("delhi","raipur"):      [("Gondwana Exp","12411","06:35 AM",900,"Sleeper/3A",530),("Chhattisgarh Exp","18237","08:45 PM",960,"Sleeper/3A",510),("Rajdhani Exp","22811","03:25 PM",780,"2A / 1A",2200)],
    ("delhi","hyderabad"):   [("Rajdhani Exp","12723","06:25 AM",720,"2A / 1A",2350),("Dakshin Exp","12721","10:45 PM",900,"Sleeper/3A",530),("Falaknuma Exp","12701","06:00 AM",960,"Sleeper/3A",510)],
    ("delhi","chennai"):     [("Rajdhani Exp","12433","06:25 AM",1260,"2A / 1A",2850),("Tamil Nadu Exp","12621","10:30 PM",1440,"Sleeper/3A",620),("GT Express","12615","06:15 PM",1320,"Sleeper/3A",590)],
    ("delhi","bangalore"):   [("Rajdhani Exp","22691","08:00 PM",1380,"2A / 1A",2950),("Karnataka Exp","12627","10:30 PM",1560,"Sleeper/3A",650),("Sampark Kranti","12649","06:45 PM",1440,"Sleeper/3A",620)],
    # ── Mumbai as origin ──
    ("mumbai","goa"):        [("Mandovi Exp","10103","07:10 AM",480,"Sleeper/3A",310),("Konkan Kanya","10111","11:00 AM",420,"Sleeper/2A",350),("Rajdhani Exp","12431","08:35 AM",390,"2A / 1A",1800)],
    ("mumbai","delhi"):      [("Rajdhani Exp","12952","04:55 PM",960,"2A / 1A",2450),("Aug Kranti Raj","12954","05:25 PM",1020,"2A / 3A",2100),("Mumbai Mail","11058","08:00 PM",1380,"Sleeper/3A",490)],
    ("mumbai","kolkata"):    [("Gitanjali Exp","12859","06:05 AM",1440,"Sleeper/3A",580),("Mumbai Mail","12321","06:00 PM",1560,"Sleeper/3A",550),("Kurla Exp","01071","07:00 AM",1620,"Sleeper/3A",520)],
    ("mumbai","hyderabad"):  [("Hussainsagar Exp","12701","05:45 PM",480,"Sleeper/3A",360),("Ajanta Exp","22109","11:00 AM",510,"Sleeper/3A",380),("Devagiri Exp","17058","05:50 PM",540,"Sleeper/3A",350)],
    ("mumbai","bangalore"):  [("Udyan Exp","11301","08:05 PM",720,"Sleeper/3A",420),("Rajdhani Exp","22691","08:00 PM",780,"2A / 1A",2100),("Mahalaxmi Exp","11029","06:00 AM",660,"Sleeper/3A",400)],
    ("mumbai","chennai"):    [("Mumbai Exp","11041","08:20 PM",900,"Sleeper/3A",450),("Dadar Exp","11063","11:35 PM",960,"Sleeper/3A",430),("CST Chennai Exp","11007","11:45 PM",960,"Sleeper/3A",430)],
    ("mumbai","nagpur"):     [("Vidarbha Exp","12105","06:00 AM",480,"Sleeper/3A",380),("Sevagram Exp","12810","06:00 PM",540,"Sleeper/3A",360),("Duronto Exp","12289","05:25 PM",420,"2A / 3A",1450)],
    ("mumbai","raipur"):     [("Gondia Exp","18030","06:40 PM",720,"Sleeper/3A",450),("Chhattisgarh Exp","18238","10:00 AM",780,"Sleeper/3A",430),("LTT Raipur Exp","11449","10:30 PM",840,"Sleeper/3A",420)],
    ("mumbai","bhopal"):     [("Punjab Mail","12137","07:05 AM",540,"Sleeper/3A",400),("Paschim Exp","12925","11:35 PM",600,"Sleeper/3A",420),("Rajdhani Exp","12627","05:25 PM",480,"2A / 1A",1850)],
    # ── Kolkata as origin ──
    ("kolkata","delhi"):     [("Rajdhani Exp","12302","04:55 PM",1020,"2A / 1A",2300),("Poorva Exp","12304","08:05 AM",1380,"Sleeper/3A",520),("Howrah Mail","12311","07:00 PM",1560,"Sleeper/3A",450)],
    ("kolkata","mumbai"):    [("Gitanjali Exp","12860","06:05 AM",1440,"Sleeper/3A",580),("Howrah Mumbai Mail","12322","08:00 PM",1560,"Sleeper/3A",550),("Duronto Exp","12262","08:15 AM",1200,"2A / 3A",2200)],
    ("kolkata","chennai"):   [("Coromandel Exp","12841","08:45 AM",780,"Sleeper/3A",470),("Howrah Chennai Mail","12839","11:45 PM",900,"Sleeper/3A",490),("Falaknuma Exp","12703","06:15 PM",960,"Sleeper/3A",500)],
    ("kolkata","bangalore"): [("Howrah YPR Exp","12863","09:55 PM",1200,"Sleeper/3A",560),("Gitanjali Exp","12860","06:05 AM",1440,"Sleeper/3A",580),("Duronto Exp","12274","08:30 AM",1080,"2A / 3A",2100)],
    ("kolkata","varanasi"):  [("Vibhuti Exp","13009","05:50 PM",600,"Sleeper/3A",380),("Poorva Exp","12304","08:05 AM",660,"Sleeper/3A",400),("Kashi Exp","13009","06:00 PM",660,"Sleeper/3A",380)],
    # ── Prayagraj as origin ──
    ("prayagraj","raipur"):  [("Prayagraj Raipur Exp","18203","06:30 AM",660,"Sleeper/3A",390),("Gondwana Exp","12411","09:30 AM",720,"Sleeper/3A",430),("Chhattisgarh Exp","18237","11:00 PM",780,"Sleeper/3A",410)],
    ("prayagraj","delhi"):   [("Prayagraj Exp","12417","04:55 PM",360,"Sleeper/3A",310),("Vande Bharat","22436","06:00 AM",240,"CC / EC",1300),("Shivganga Exp","12560","11:30 PM",420,"Sleeper/3A",290)],
    ("prayagraj","mumbai"):  [("Prayagraj Exp","11071","06:30 AM",1020,"Sleeper/3A",510),("LTT Exp","11093","11:00 PM",1080,"Sleeper/3A",490),("Kamayani Exp","11071","06:30 AM",1020,"Sleeper/3A",510)],
    ("prayagraj","kolkata"): [("Howrah Exp","13009","06:15 PM",420,"Sleeper/3A",330),("Vibhuti Exp","13009","06:15 PM",420,"Sleeper/3A",330),("Poorva Exp","12304","10:15 AM",480,"Sleeper/3A",350)],
    ("prayagraj","varanasi"):[("Prayagraj–Varanasi Pass","54251","07:00 AM",150,"2S",80),("Mahanagari Exp","11094","09:00 AM",120,"Sleeper/3A",120),("Kashi Exp","13009","06:00 PM",120,"Sleeper/3A",110)],
    ("prayagraj","lucknow"): [("Lucknow Exp","14235","06:00 AM",180,"Sleeper/3A",160),("Gomti Exp","12420","08:00 AM",180,"Sleeper/3A",160),("Intercity Exp","12419","06:15 AM",180,"CC",280)],
    # ── Raipur as origin ──
    ("raipur","delhi"):      [("Chhattisgarh Exp","18238","06:00 AM",960,"Sleeper/3A",510),("Gondwana Exp","12412","07:00 PM",900,"Sleeper/3A",530),("Rajdhani Exp","22812","03:25 PM",780,"2A / 1A",2200)],
    ("raipur","mumbai"):     [("LTT Raipur Exp","11450","07:30 AM",840,"Sleeper/3A",420),("Gondia Exp","18029","06:00 PM",720,"Sleeper/3A",450),("Chhattisgarh Exp","18237","08:00 AM",780,"Sleeper/3A",430)],
    ("raipur","kolkata"):    [("Ispat Exp","18005","08:15 AM",540,"Sleeper/3A",360),("Howrah Exp","18029","10:00 PM",600,"Sleeper/3A",380),("Sambalpur Exp","18005","08:00 AM",540,"Sleeper/3A",360)],
    ("raipur","nagpur"):     [("Chhattisgarh Exp","18237","08:30 AM",240,"Sleeper/3A",210),("Gondwana Exp","12411","10:00 AM",270,"Sleeper/3A",230),("Nagpur Raipur Exp","12849","06:00 AM",300,"Sleeper/3A",220)],
    ("raipur","hyderabad"):  [("Durg HYB Exp","12833","06:30 AM",720,"Sleeper/3A",430),("Visakha Exp","18645","09:00 PM",840,"Sleeper/3A",460),("Amaravati Exp","12833","06:30 AM",720,"Sleeper/3A",430)],
    ("raipur","bhopal"):     [("Chhattisgarh Exp","18237","07:00 AM",420,"Sleeper/3A",290),("Jabalpur Exp","11449","11:00 PM",480,"Sleeper/3A",310),("Gondwana Exp","12411","09:00 AM",450,"Sleeper/3A",300)],
    ("raipur","varanasi"):   [("Chhattisgarh Exp","18238","08:30 PM",540,"Sleeper/3A",370),("Gondwana Exp","12412","11:00 AM",600,"Sleeper/3A",390),("Utkal Exp","18477","05:30 PM",660,"Sleeper/3A",380)],
    # ── Hyderabad / Bangalore / Chennai as origin ──
    ("hyderabad","mumbai"):  [("Hussainsagar Exp","12702","05:45 PM",480,"Sleeper/3A",360),("Konark Exp","11020","11:00 PM",540,"Sleeper/3A",380),("Devagiri Exp","17057","06:00 AM",510,"Sleeper/3A",350)],
    ("hyderabad","delhi"):   [("Rajdhani Exp","12724","06:25 AM",720,"2A / 1A",2350),("Dakshin Exp","12722","11:15 AM",900,"Sleeper/3A",530),("Kacheguda Exp","12706","06:00 AM",960,"Sleeper/3A",510)],
    ("hyderabad","bangalore"):[("Rayalaseema Exp","17429","06:00 AM",420,"Sleeper/3A",290),("Kacheguda Exp","17603","06:30 PM",480,"Sleeper/3A",310),("Garib Rath","12733","11:00 PM",390,"3A",380)],
    ("hyderabad","chennai"): [("Charminar Exp","12759","06:15 PM",480,"Sleeper/3A",340),("Falaknuma Exp","12704","06:00 AM",420,"Sleeper/3A",310),("Rayalaseema Exp","17430","07:30 AM",480,"Sleeper/3A",330)],
    ("bangalore","delhi"):   [("Rajdhani Exp","22692","08:00 PM",1380,"2A / 1A",2950),("Karnataka Exp","12628","10:30 PM",1560,"Sleeper/3A",650),("Yeshvantpur Raj","22692","08:00 PM",1380,"2A / 1A",2950)],
    ("bangalore","mumbai"):  [("Udyan Exp","11302","08:05 PM",720,"Sleeper/3A",420),("Gangavati Exp","17301","07:00 AM",780,"Sleeper/3A",400),("CST Exp","11010","08:00 AM",660,"Sleeper/3A",390)],
    ("bangalore","chennai"): [("Shatabdi Exp","12027","06:00 AM",300,"CC / EC",745),("Brindavan Exp","12639","07:30 AM",360,"CC / 2S",310),("Lalbagh Exp","12607","06:30 AM",330,"CC / 2S",280)],
    ("bangalore","hyderabad"):[("Rajdhani Exp","12428","05:30 PM",480,"2A / 1A",1800),("Kacheguda Exp","17604","07:30 AM",480,"Sleeper/3A",310),("Garib Rath","12734","06:00 PM",420,"3A",380)],
    ("chennai","delhi"):     [("Rajdhani Exp","12434","06:25 AM",1260,"2A / 1A",2850),("Tamil Nadu Exp","12622","10:00 PM",1440,"Sleeper/3A",620),("GT Express","12616","07:00 AM",1380,"Sleeper/3A",600)],
    ("chennai","mumbai"):    [("Mumbai Exp","11042","08:20 PM",900,"Sleeper/3A",450),("CST Exp","11008","11:55 PM",960,"Sleeper/3A",430),("Konark Exp","11019","11:00 PM",960,"Sleeper/3A",430)],
    ("chennai","bangalore"): [("Shatabdi Exp","12028","03:00 PM",300,"CC / EC",745),("Brindavan Exp","12640","02:35 PM",360,"CC / 2S",310),("Intercity Exp","12083","03:00 PM",300,"CC",280)],
    ("chennai","kolkata"):   [("Coromandel Exp","12842","08:45 AM",780,"Sleeper/3A",470),("Chennai Mail","12840","09:00 PM",900,"Sleeper/3A",490),("East Coast Exp","12842","08:45 AM",780,"Sleeper/3A",470)],
}

# Fallback: trains for unknown route
_FALLBACK_TRAINS = [
    ("Superfast Exp",  "—", "Check IRCTC", None, "Sleeper/3A", 400),
    ("Rajdhani Exp",   "—", "Check IRCTC", None, "2A / 1A",   1800),
    ("Jan Shatabdi",   "—", "Check IRCTC", None, "CC / 2S",    250),
]

def get_train_estimates(origin, destination, num_pax=1):
    ok = _canon(origin)
    dk = _canon(destination)
    trains_raw = _TRAIN_ROUTES.get((ok, dk)) or _TRAIN_ROUTES.get((dk, ok))
    # partial fallback: find any route with matching destination
    if trains_raw is None:
        for (o, d), v in _TRAIN_ROUTES.items():
            if d == dk:
                trains_raw = v
                break
    if trains_raw is None:
        trains_raw = _FALLBACK_TRAINS

    result = []
    for (name, no, dep, dur_mins, cls, fare) in trains_raw:
        if dur_mins is not None:
            arrival   = _add_minutes(dep, dur_mins)
            h, m      = divmod(dur_mins, 60)
            dur_label = f"{h}h {m:02d}m" if m else f"{h}h"
        else:
            arrival   = "Check IRCTC"
            dur_label = "Varies"
        result.append({
            "Train":       name,
            "No":          no,
            "Departure":   dep,
            "Arrival":     arrival,
            "Duration":    dur_label,
            "Class":       cls,
            "Fare/Person": f"₹{fare:,}" if isinstance(fare, int) else fare,
            "Total Fare":  f"₹{fare * num_pax:,}" if isinstance(fare, int) else "Check IRCTC",
        })
    return result

def get_hotel_estimates(destination, max_budget, num_pax=1):
    rooms = max(1, (num_pax + 1) // 2)  # 2 guests per room
    rates = [min(max_budget, r) for r in [4500, 2800, 1500]]
    return [
        {"Property": "Grand Horizon Resort", "Rating": "4.8 ⭐",
         "Per Night": f"₹{rates[0]:,}", "Rooms Needed": rooms, "Total/Night": f"₹{rates[0]*rooms:,}"},
        {"Property": "Urban Pearl Boutique", "Rating": "4.3 ⭐",
         "Per Night": f"₹{rates[1]:,}", "Rooms Needed": rooms, "Total/Night": f"₹{rates[1]*rooms:,}"},
        {"Property": "Backpackers Hideout",  "Rating": "4.0 ⭐",
         "Per Night": f"₹{rates[2]:,}", "Rooms Needed": rooms, "Total/Night": f"₹{rates[2]*rooms:,}"},
    ]

def get_weather_data(city_name, api_key_str):
    """Returns (data_dict | None, error_str | None)."""
    if not api_key_str:
        return None, None
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?q={city_name}&appid={api_key_str}&units=metric")
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json(), None
        if resp.status_code == 401:
            return None, "Invalid OpenWeather API key (401). Check OPENWEATHER_API_KEY in secrets.toml."
        if resp.status_code == 404:
            return None, f"City '{city_name}' not found on OpenWeatherMap."
        return None, f"OpenWeather error {resp.status_code}."
    except requests.exceptions.Timeout:
        return None, "Weather API timed out."
    except requests.exceptions.ConnectionError:
        return None, "Could not reach OpenWeather API."
    except Exception as exc:
        return None, f"Weather error: {exc}"

def generate_packing_list(temp, condition):
    items = [
        "🪪 ID Proofs, Driver's License & Booking Passes",
        "📱 Phone Charger, Power Bank & Universal Adapter",
        "💊 Basic First-Aid Kit & Personal Medications",
        "🧴 Sunscreen SPF 50+ & Personal Toiletries",
    ]
    if temp < 15:
        items.extend(["🧥 Heavy Jacket, Thermal Wear & Sweaters",
                       "🧣 Muffler, Gloves & Woolen Socks",
                       "💋 Lip Balm & Moisturizer"])
    elif temp <= 28:
        items.extend(["👕 Cotton T-shirts, Jeans & Light Layers",
                       "👟 Comfortable Walking / Hiking Shoes",
                       "🕶️ Sunglasses & Light Jacket"])
    else:
        items.extend(["🩳 Breathable Cotton / Linen Clothes",
                       "🧢 Cap / Hat & UV Protection",
                       "💧 Reusable Water Bottle"])
    if any(w in condition.lower() for w in ["rain", "drizzle", "thunderstorm"]):
        items.extend(["☂️ Umbrella / Raincoat",
                       "🎒 Waterproof Backpack Cover",
                       "🩴 Quick-dry Footwear"])
    return items


# --- IBM GRANITE MODEL ---
@st.cache_resource(show_spinner=False)
def _get_model(api_key_val: str, project_id_val: str, region_val: str) -> ModelInference:
    return ModelInference(
        model_id="ibm/granite-4-h-small",
        credentials={"url": f"https://{region_val}.ml.cloud.ibm.com", "apikey": api_key_val},
        project_id=project_id_val,
    )

def _classify_watsonx_error(exc: Exception) -> str:
    msg = str(exc)
    ml = msg.lower()
    if "401" in msg or "unauthorized" in ml or "invalid api key" in ml:
        return "🔑 Auth failed (401). Your IBM IAM API Key is invalid or expired. Regenerate at cloud.ibm.com."
    if "403" in msg or "forbidden" in ml:
        return "🚫 Access denied (403). Verify your watsonx Project ID and API key permissions."
    if "404" in msg or "not found" in ml:
        return f"🤖 Model not found (404). 'ibm/granite-4-h-small' may be unavailable in region '{region}'. Try 'us-south'."
    if "429" in msg or "rate limit" in ml:
        return "⏳ Rate limit hit (429). Wait a moment and try again."
    if "quota" in ml or "resource units" in ml:
        return "📊 Quota exceeded. watsonx resource units exhausted for this billing period."
    if "timeout" in ml or "timed out" in ml:
        return "⌛ Request timed out. IBM watsonx did not respond. Please retry."
    if "connection" in ml or "network" in ml:
        return "🌐 Network error. Cannot reach IBM watsonx. Check your connection."
    return f"❌ IBM Granite error: {msg}"

def call_granite_model(prompt_text, api_key_val, project_id_val, region_val, temperature_val, max_tokens_val) -> str:
    model = _get_model(api_key_val, project_id_val, region_val)
    result = model.generate_text(
        prompt=prompt_text,
        params={GenParams.MAX_NEW_TOKENS: max_tokens_val, GenParams.TEMPERATURE: temperature_val},
    )
    if not result or not result.strip():
        raise ValueError("Model returned an empty response. Rephrase your prompt and try again.")
    return result


# ═══════════════════════════════════════════════════════════
# TRIP CONFIGURATOR FORM
# ═══════════════════════════════════════════════════════════
st.markdown("""
    <div class="glass-card">
        <div class="section-header">✈️ Trip Configurator</div>
        <div class="section-sub">Choose a destination, set your budget, and let IBM Granite build your perfect itinerary.</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("**🔥 Quick Pick:**")
chip_cols = st.columns(8)
_CHIPS = [
    ("🏖️", "Goa"), ("🏔️", "Manali"), ("🏰", "Jaipur"), ("🌴", "Kerala"),
    ("🏙️", "Mumbai"), ("🕌", "Agra"), ("🛕", "Varanasi"), ("🏔️", "Leh Ladakh"),
]
for i, (icon, name) in enumerate(_CHIPS):
    if chip_cols[i].button(f"{icon} {name}", key=f"chip_{name}"):
        st.session_state["selected_chip"] = name
        st.session_state["custom_dest"] = ""

col1, col2, col3 = st.columns([1.1, 1, 1])
with col1:
    origin = st.text_input("🛫 Origin City", "Delhi")
    dest_keys = list(DESTINATION_COORDS.keys())
    chip_val  = st.session_state.get("selected_chip")
    default_idx = dest_keys.index(chip_val) if chip_val in dest_keys else 0
    destination = st.selectbox("🛬 Destination City", dest_keys, index=default_idx)
    custom_dest = st.text_input(
        "✏️ Or type a custom destination",
        value=st.session_state.get("custom_dest", ""),
        placeholder="e.g. Shimla, Rishikesh, Udaipur…",
        key="custom_dest_input",
    )
    if custom_dest.strip():
        destination = custom_dest.strip()
        st.session_state["custom_dest"] = custom_dest.strip()

with col2:
    duration       = st.number_input("📅 Duration (Days)", min_value=1, max_value=14, value=3)
    num_travellers = st.number_input("👥 Number of Travellers", min_value=1, max_value=20, value=1)
    budget         = st.number_input("💰 Total Budget (INR)", min_value=3000, value=20000, step=1000)

with col3:
    hotel_budget = st.number_input("🏨 Max Hotel / Night (INR)", min_value=1000, value=4500, step=500)
    travel_style = st.selectbox("✨ Travel Style", [
        "Relaxed & Chill", "Action-Packed Adventure", "Cultural Explorer",
        "Luxury & Pampering", "Budget Backpacker",
    ])

st.markdown("<br>", unsafe_allow_html=True)
generate_btn = st.button("🚀 Generate My AI Travel Plan", use_container_width=True)

if generate_btn:
    if not api_key.strip() or not project_id.strip():
        st.error("🔑 Please enter a valid IBM IAM API Key and Project ID in the sidebar.")
    else:
        with st.spinner("🤖 IBM Granite is crafting your personalised travel plan…"):
            try:
                flights = get_flight_estimates(origin, destination, num_travellers)
                trains  = get_train_estimates(origin, destination, num_travellers)
                hotels  = get_hotel_estimates(destination, hotel_budget, num_travellers)

                prompt = f"""<|system|>
You are an intelligent AI Travel Planner Agent built on IBM Granite. Generate an exciting, detailed day-by-day travel plan formatted strictly in Markdown. Do NOT include code blocks.

<|user|>
Create a comprehensive, optimised trip plan:
- Route: {origin} → {destination}
- Duration: {duration} Days
- Travellers: {num_travellers} person(s)
- Total Budget: ₹{budget:,}
- Hotel Budget: ₹{hotel_budget:,}/night
- Travel Style: {travel_style}
Flight Options: {json.dumps(flights)}
Train Options: {json.dumps(trains)}
Hotel Choices: {json.dumps(hotels)}

Format with Markdown headers (###) and emojis:
1. ✨ Overview & Highlights
2. ✈️ / 🚂 Recommended Transport (Flight or Train) & Hotel (with reasoning)
3. 🗓️ Day-by-Day Schedule (best time-of-day for each activity)
4. 🍽️ Local Culinary Spots & Street Food Guide
5. 🗺️ Hidden Gems (lesser-known spots a local would recommend)
6. ⚡ Schedule Optimisation Tips
7. 💰 Financial Summary (accommodation · food · local travel · sightseeing · misc estimates)

<|assistant|>
"""
                response = call_granite_model(
                    prompt, api_key.strip(), project_id.strip(),
                    region, temperature, max_tokens,
                )

                st.session_state.update({
                    "trip_generated": True,
                    "itinerary_result": response,
                    "flights": flights,
                    "trains": trains,
                    "hotels": hotels,
                    "current_dest": destination,
                    "origin": origin,
                    "duration": duration,
                    "budget": budget,
                    "hotel_budget": hotel_budget,
                    "travel_style": travel_style,
                    "num_travellers": num_travellers,
                    "chat_messages": [
                        {"role": "assistant",
                         "content": f"Hi! I'm your AI Concierge. How can I improve your trip to **{destination}**? 🌍"}
                    ],
                })
                st.balloons()
            except Exception as e:
                st.error(_classify_watsonx_error(e))


# ═══════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════
if st.session_state.get("trip_generated", False):
    st.markdown("<br>", unsafe_allow_html=True)

    _dest_key = st.session_state["current_dest"]
    if _dest_key in DESTINATION_COORDS:
        dest_info = DESTINATION_COORDS[_dest_key]
    else:
        dest_info = {
            "lat": 20.5937, "lon": 78.9629,
            "image": _destination_image_url(_dest_key),
            "spots": [{"name": _dest_key, "lat": 20.5937, "lon": 78.9629}],
        }

    # ── Destination hero banner ──────────────────────────────────────────
    st.markdown(f"""
        <div class="dest-hero">
            <img src="{dest_info['image']}"
                 onerror="this.src='https://images.unsplash.com/photo-1524492412937-b28074a5d7da?w=1200&q=80'"
                 alt="{_dest_key}" />
            <div class="dest-hero-overlay"></div>
            <div class="dest-hero-text">
                <div class="dest-hero-title">📍 {_dest_key}</div>
                <div class="dest-hero-sub">
                    <span class="dest-hero-tag">🛫 {st.session_state['origin']}</span>
                    <span style="color:rgba(255,255,255,0.5)">→</span>
                    <span class="dest-hero-tag">🛬 {_dest_key}</span>
                    <span class="dest-hero-tag">📅 {st.session_state['duration']} Days</span>
                    <span class="dest-hero-tag">✨ {st.session_state['travel_style']}</span>
                    <span class="dest-hero-tag">💰 ₹{st.session_state['budget']:,}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Summary metrics ──────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🛫 Route",      f"{st.session_state['origin']} → {_dest_key}")
    m2.metric("📅 Duration",   f"{st.session_state['duration']} Days")
    m3.metric("💰 Budget",     f"₹{st.session_state['budget']:,}")
    m4.metric("👥 Travellers", f"{st.session_state.get('num_travellers', 1)} person(s)")
    m5.metric("✨ Style",      st.session_state["travel_style"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_itinerary, tab_map, tab_budget, tab_weather, tab_chat = st.tabs([
        "✨ AI Itinerary",
        "🗺️ Route Map",
        "📊 Budget Breakdown",
        "🌦️ Weather & Packing",
        "💬 AI Concierge",
    ])

    # ── 1. ITINERARY ─────────────────────────────────────────────────────
    with tab_itinerary:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown(st.session_state["itinerary_result"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        fi_col, tr_col = st.columns(2)
        with fi_col:
            st.markdown("#### ✈️ Available Flights")
            st.dataframe(pd.DataFrame(st.session_state["flights"]),
                         use_container_width=True, hide_index=True)
        with tr_col:
            st.markdown("#### 🚂 Train Options")
            st.dataframe(pd.DataFrame(st.session_state["trains"]),
                         use_container_width=True, hide_index=True)

        st.markdown("#### 🏨 Hotel Options")
        st.dataframe(pd.DataFrame(st.session_state["hotels"]),
                     use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Booking Alerts")
        travel_date = st.date_input(
            "📅 Select Travel Date",
            value=datetime.date.today() + datetime.timedelta(days=14),
            key="travel_date_input",
        )
        days_until = (travel_date - datetime.date.today()).days
        if days_until < 0:
            st.error("⚠️ Travel date is in the past. Please select a future date.")
        elif days_until <= 3:
            st.warning(f"🚨 **Last-minute!** Trip in {days_until} day(s) — book immediately.")
        elif days_until <= 7:
            st.warning(f"⏰ **Reminder:** Trip in {days_until} days — confirm your bookings soon.")
        elif days_until <= 30:
            st.info(f"📌 **Plan Ahead:** {days_until} days to go. Great time for early-bird rates.")
        else:
            st.success(f"✅ **Great timing!** {days_until} days until your trip. Set reminders & watch for deals.")

        if st.checkbox("✅ Mark flights & hotel as booked", key="booking_confirmed"):
            st.success(f"🎉 Booking confirmed for **{_dest_key}** on **{travel_date.strftime('%d %B %Y')}**! Have a wonderful trip!")

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Itinerary (.md)",
            data=st.session_state["itinerary_result"],
            file_name=f"{_dest_key}_Itinerary.md",
            mime="text/markdown",
        )

    # ── 2. MAP ────────────────────────────────────────────────────────────
    with tab_map:
        st.markdown(f"### 🗺️ Destination Map — {_dest_key}")
        st.caption(f"Key attractions and points of interest around {_dest_key}.")

        df_spots = pd.DataFrame(dest_info["spots"])
        # Ensure required columns exist
        for col in ["name", "lat", "lon"]:
            if col not in df_spots.columns:
                df_spots[col] = _dest_key if col == "name" else dest_info.get(col[0:3], 0)

        view_state = pdk.ViewState(
            latitude=dest_info["lat"], longitude=dest_info["lon"],
            zoom=10, pitch=45,
        )
        layer = pdk.Layer(
            "ScatterplotLayer", data=df_spots,
            get_position="[lon, lat]",
            get_color="[15, 98, 254, 200]",
            get_radius=1200, pickable=True,
        )
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                                  tooltip={"text": "{name}"}))
        st.dataframe(df_spots[["name", "lat", "lon"]], use_container_width=True, hide_index=True)

    # ── 3. BUDGET ─────────────────────────────────────────────────────────
    with tab_budget:
        st.markdown("### 📊 Financial & Cost Breakdown")
        st.caption("Estimated allocation across all spend categories based on your total budget, duration and hotel preference.")

        b_total    = st.session_state["budget"]
        b_duration = st.session_state["duration"]
        b_hotel    = st.session_state["hotel_budget"]
        b_pax      = st.session_state.get("num_travellers", 1)

        # ── Allocation logic ──────────────────────────────────────────────
        # Accommodation: hotel × nights (hard cost, capped at 45 % of budget)
        est_accommodation = min(b_hotel * b_duration, b_total * 0.45)
        # Transport (intercity flights/trains): 25 % of budget, capped at ₹9,000
        est_transport     = min(b_total * 0.25, 9000)
        # Food & dining: ₹600/person/day (covers 3 meals + tea/snacks)
        est_food          = 600 * b_pax * b_duration
        # Local travel (autos, cabs, ferries within destination): ₹350/person/day
        est_local         = 350 * b_pax * b_duration
        # Sightseeing & entry fees: ₹250/person/day
        est_sightseeing   = 250 * b_pax * b_duration
        # Miscellaneous & emergency buffer: 5 % of budget
        est_misc          = b_total * 0.05

        est_total = (est_accommodation + est_transport + est_food
                     + est_local + est_sightseeing + est_misc)

        # If total overshoots budget, scale down proportionally (keep accommodation fixed)
        if est_total > b_total:
            flex = est_total - est_accommodation - est_misc
            scale = max(0.0, (b_total - est_accommodation - est_misc) / flex) if flex > 0 else 1.0
            est_transport   *= scale
            est_food        *= scale
            est_local       *= scale
            est_sightseeing *= scale
            est_total = (est_accommodation + est_transport + est_food
                         + est_local + est_sightseeing + est_misc)

        remaining  = max(0.0, b_total - est_total)
        per_person = est_total / max(b_pax, 1)

        # ── Top summary metrics ───────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("💰 Total Budget",    f"₹{b_total:,.0f}")
        mc2.metric("🧾 Est. Total Cost", f"₹{est_total:,.0f}")
        mc3.metric("👤 Per Person",      f"₹{per_person:,.0f}")
        mc4.metric("💚 Buffer",          f"₹{remaining:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Category breakdown with progress bars ─────────────────────────
        st.markdown("#### 📂 Category Breakdown")

        categories = [
            ("🏨", "Accommodation",       est_accommodation, "#0f62fe"),
            ("✈️", "Transport (Intercity)", est_transport,    "#7c3aed"),
            ("🍽️", "Food & Dining",        est_food,         "#059669"),
            ("🚕", "Local Travel",         est_local,        "#d97706"),
            ("🏛️", "Sightseeing & Entry",  est_sightseeing,  "#db2777"),
            ("🎒", "Misc & Emergency",     est_misc,         "#64748b"),
        ]

        for icon, label, amt, color in categories:
            pct = (amt / b_total * 100) if b_total > 0 else 0
            pp  = amt / max(b_pax, 1)
            st.markdown(f"""
                <div style="margin-bottom:0.9rem;">
                  <div style="display:flex; justify-content:space-between;
                              align-items:center; margin-bottom:0.25rem;">
                    <span style="font-weight:600; color:#1e293b; font-size:0.9rem;">
                      {icon} {label}
                    </span>
                    <span style="font-size:0.88rem; color:#475569;">
                      <b style="color:#0f172a;">₹{amt:,.0f}</b>
                      &nbsp;·&nbsp; ₹{pp:,.0f}/person
                      &nbsp;·&nbsp; <span style="color:{color};">{pct:.1f}%</span>
                    </span>
                  </div>
                  <div style="background:#e2e8f0; border-radius:99px; height:10px; overflow:hidden;">
                    <div style="width:{min(pct,100):.1f}%; background:{color};
                                height:100%; border-radius:99px;
                                transition:width 0.6s ease;"></div>
                  </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Per-day breakdown ─────────────────────────────────────────────
        st.markdown("#### 📅 Daily Spend Estimate")
        d1, d2, d3 = st.columns(3)
        d1.metric("🍽️ Food / day",       f"₹{600 * b_pax:,.0f}", delta=f"₹600 × {b_pax} person(s)")
        d2.metric("🚕 Local travel / day", f"₹{350 * b_pax:,.0f}", delta=f"₹350 × {b_pax} person(s)")
        d3.metric("🏛️ Sightseeing / day",  f"₹{250 * b_pax:,.0f}", delta=f"₹250 × {b_pax} person(s)")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bar chart ─────────────────────────────────────────────────────
        chart_data = pd.DataFrame({
            "Category": ["Accommodation", "Transport", "Food", "Local Travel", "Sightseeing", "Misc"],
            "Estimated Cost (₹)": [
                est_accommodation, est_transport, est_food,
                est_local, est_sightseeing, est_misc,
            ],
        })
        st.bar_chart(chart_data, x="Category", y="Estimated Cost (₹)", color="#0f62fe")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Budget health indicator ───────────────────────────────────────
        if remaining >= b_total * 0.10:
            st.success(
                f"✅ **Healthy budget!** ₹{remaining:,.0f} buffer remaining "
                f"({remaining/b_total*100:.0f}% of total) — good headroom for spontaneous experiences!"
            )
        elif remaining > 0:
            st.warning(
                f"⚠️ **Tight budget.** Only ₹{remaining:,.0f} buffer left "
                f"({remaining/b_total*100:.0f}%). Consider reducing hotel nights or travel style."
            )
        else:
            st.error(
                f"🚨 **Over budget by ₹{abs(remaining):,.0f}.** "
                "Reduce hotel budget per night, shorten the trip, or increase total budget."
            )

        # ── Assumptions note ─────────────────────────────────────────────
        st.markdown("""
            <div class="info-box" style="margin-top:1rem;">
              📌 <b>Estimation basis:</b> Accommodation = hotel rate × nights &nbsp;·&nbsp;
              Food = ₹600/person/day &nbsp;·&nbsp; Local travel = ₹350/person/day &nbsp;·&nbsp;
              Sightseeing = ₹250/person/day &nbsp;·&nbsp; Misc = 5% of budget.
              Actual costs may vary by season, destination and preferences.
            </div>
        """, unsafe_allow_html=True)

    # ── 4. WEATHER & PACKING ──────────────────────────────────────────────
    with tab_weather:
        st.markdown(f"### 🌤️ Live Weather — {_dest_key}")

        w_data, w_error = get_weather_data(_dest_key, weather_api_key)
        if w_data:
            w_temp     = w_data["main"]["temp"]
            w_desc     = w_data["weather"][0]["description"].title()
            w_humidity = w_data["main"]["humidity"]
            w_wind     = w_data["wind"]["speed"]
            st.markdown('<div class="info-box">🟢 <b>Live weather data loaded successfully.</b></div>',
                        unsafe_allow_html=True)
        else:
            if w_error:
                st.warning(f"⚠️ {w_error} Showing sample data.")
            else:
                st.markdown('<div class="info-box">ℹ️ Add <code>OPENWEATHER_API_KEY</code> to <code>secrets.toml</code> for live weather.</div>',
                            unsafe_allow_html=True)
            w_temp, w_desc, w_humidity, w_wind = 26.0, "Partly Cloudy", 60, 12.0

        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.markdown(f'<div class="weather-card"><div class="wc-label">🌡️ Temperature</div><div class="wc-value">{w_temp:.1f}°<span class="wc-unit">C</span></div></div>', unsafe_allow_html=True)
        wc2.markdown(f'<div class="weather-card"><div class="wc-label">🌤️ Condition</div><div class="wc-value" style="font-size:1.1rem;line-height:1.3">{w_desc}</div></div>', unsafe_allow_html=True)
        wc3.markdown(f'<div class="weather-card"><div class="wc-label">💧 Humidity</div><div class="wc-value">{w_humidity}<span class="wc-unit">%</span></div></div>', unsafe_allow_html=True)
        wc4.markdown(f'<div class="weather-card"><div class="wc-label">💨 Wind Speed</div><div class="wc-value">{w_wind}<span class="wc-unit"> m/s</span></div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧳 Smart Packing Checklist")
        st.caption("Auto-generated based on live weather conditions at your destination.")

        packing_items = generate_packing_list(w_temp, w_desc)
        cols = st.columns(2)
        for idx, item in enumerate(packing_items):
            with cols[idx % 2]:
                st.checkbox(item, value=False, key=f"pack_{idx}")

    # ── 5. AI CONCIERGE ───────────────────────────────────────────────────
    with tab_chat:
        st.markdown("### 💬 AI Travel Concierge")
        st.caption("Ask questions, request changes, or explore alternatives — powered by IBM Granite 4.")

        for msg in st.session_state.get("chat_messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_prompt := st.chat_input("e.g. 'Add vegetarian street food for Day 1' or 'What to pack for Manali in December?'"):
            st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
            with st.chat_message("assistant"):
                with st.spinner("IBM Granite is thinking…"):
                    try:
                        bot_prompt = f"""<|system|>
You are a helpful AI travel concierge. Answer conversationally using the trip context below.

Trip: {st.session_state['origin']} → {_dest_key}, {st.session_state['duration']} days, {st.session_state.get('num_travellers', 1)} traveller(s), ₹{st.session_state['budget']:,} budget, {st.session_state['travel_style']} style.

<|user|>
{user_prompt}
<|assistant|>
"""
                        bot_response = call_granite_model(
                            bot_prompt, api_key.strip(), project_id.strip(),
                            region, temperature, max_tokens,
                        )
                        st.markdown(bot_response)
                        st.session_state["chat_messages"].append(
                            {"role": "assistant", "content": bot_response}
                        )
                    except Exception as err:
                        st.error(_classify_watsonx_error(err))
