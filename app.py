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

# --- LIGHT MODE CSS ---
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
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5ff 100%);
        border-right: 1px solid #dde3f0;
    }
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #dde3f0 !important;
    }

    /* ── Hero banner — keep blue, it looks great on light too ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f62fe 0%, #1192e8 45%, #0043ce 100%);
        padding: 3rem 2.5rem 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 16px 48px rgba(15,98,254,0.30), 0 0 0 1px rgba(15,98,254,0.12);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -80px; left: -40px;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .hero-tagline {
        font-size: 1.15rem;
        opacity: 0.93;
        font-weight: 400;
        letter-spacing: 0.2px;
    }
    .hero-badges {
        margin-top: 1.2rem;
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ── Card / panel ── */
    .glass-card {
        background: #ffffff;
        border: 1px solid #dde3f0;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 2px 12px rgba(15,98,254,0.06);
        margin-bottom: 1rem;
    }

    /* ── Section headers ── */
    .section-header {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Quick-pick chip buttons ── */
    div[data-testid="column"] > div > div > div > button {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #1e40af !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
        box-shadow: 0 1px 4px rgba(15,98,254,0.08) !important;
    }
    div[data-testid="column"] > div > div > div > button:hover {
        background: #eff6ff !important;
        border-color: #0f62fe !important;
        color: #0f62fe !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(15,98,254,0.18) !important;
    }

    /* ── Form inputs ── */
    .stTextInput input,
    .stNumberInput input,
    .stNumberInput input[type="number"],
    .stTextArea textarea,
    input, textarea {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 10px !important;
        caret-color: #0f62fe !important;
    }
    input::placeholder, textarea::placeholder {
        color: #94a3b8 !important;
    }
    input:focus, textarea:focus {
        border-color: #0f62fe !important;
        box-shadow: 0 0 0 3px rgba(15,98,254,0.12) !important;
        outline: none !important;
        background: #f8faff !important;
        color: #1e293b !important;
    }
    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div[class*="ValueContainer"] span,
    .stSelectbox [data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"] * {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    /* Number input stepper wrapper */
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
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* ── Generate button ── */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0f62fe 0%, #1192e8 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 6px 20px rgba(15,98,254,0.30) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 10px 28px rgba(15,98,254,0.42) !important;
        color: white !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1.5px solid #dde3f0 !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 2px 8px rgba(15,98,254,0.07) !important;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.3rem !important; font-weight: 700 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9 !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid #dde3f0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.1rem !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f62fe, #1192e8) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(15,98,254,0.25) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem !important;
    }

    /* ── Itinerary result card ── */
    .result-container {
        background: #ffffff;
        padding: 2rem 2.2rem;
        border-radius: 18px;
        border-left: 5px solid #0f62fe;
        border: 1.5px solid #dde3f0;
        border-left: 5px solid #0f62fe;
        box-shadow: 0 4px 20px rgba(15,98,254,0.08);
        margin-top: 1rem;
        color: #1e293b;
    }
    .result-container h1, .result-container h2,
    .result-container h3, .result-container h4 {
        color: #0f62fe !important;
    }
    .result-container p, .result-container li { color: #334155 !important; }
    .result-container strong { color: #1e293b !important; }

    /* ── Weather metric cards ── */
    .weather-card {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1.5px solid #bfdbfe;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
    }
    .weather-card .wc-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .weather-card .wc-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1d4ed8;
    }
    .weather-card .wc-unit {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* ── Packing checklist ── */
    .stCheckbox label { color: #334155 !important; font-size: 0.92rem !important; }
    .stCheckbox [data-testid="stMarkdownContainer"] p { color: #334155 !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* ── Subheaders / text ── */
    h1, h2, h3, h4, .stSubheader { color: #0f172a !important; }
    p, .stMarkdown p { color: #334155 !important; }
    .stCaption { color: #94a3b8 !important; }

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
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #f8faff !important;
        border: 1px solid #dde3f0 !important;
        border-radius: 14px !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stChatInputTextArea"] {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 12px !important;
    }

    /* ── Divider ── */
    hr { border-color: #dde3f0 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🌍 Wanderlust Agentic AI</div>
        <div class="hero-tagline">Autonomous Travel Planning · Visual Mapping · Live Weather · Powered by <b>IBM Granite 4</b></div>
        <div class="hero-badges">
            <span class="hero-badge">✈️ Smart Itineraries</span>
            <span class="hero-badge">🗺️ Interactive Maps</span>
            <span class="hero-badge">🌦️ Live Weather</span>
            <span class="hero-badge">💬 AI Concierge</span>
            <span class="hero-badge">🤖 IBM Granite 4</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ENGINE CONTROL ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=100)
    st.markdown("## ⚙️ Engine Control")
    st.caption("AICTE 2026 · Problem Statement 5")
    st.markdown("---")

    st.markdown("### 🔑 IBM Credentials")

    # FIX: safe secrets loading with no hardcoded credential defaults
    default_key = st.secrets.get("IBM_API_KEY", "")
    default_pid = st.secrets.get("WATSONX_PROJECT_ID", "")
    # FIX: weather_api_key defined at sidebar scope so it is always bound
    weather_api_key = st.secrets.get("OPENWEATHER_API_KEY", "")

    api_key = st.text_input("IBM IAM API Key", value=default_key, type="password")
    project_id = st.text_input("watsonx Project ID", value=default_pid)
    region = st.selectbox("Cloud Region", ["us-south", "eu-de"], index=0)

    st.markdown("---")
    st.markdown("### 🎛️ Agent Parameters")
    temperature = st.slider("Creativity (Temperature)", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Output Tokens", min_value=300, max_value=2500, value=1500, step=100)

    st.markdown("---")
    st.markdown("""
        <div style='font-size:0.75rem; color:#475569; line-height:1.6;'>
            🔒 Credentials are stored in <code>.streamlit/secrets.toml</code><br>
            🤖 Model: <b>ibm/granite-4-h-small</b><br>
            🌐 Weather: OpenWeatherMap API
        </div>
    """, unsafe_allow_html=True)

# --- DESTINATION DATA & MOCK APIs ---
DESTINATION_COORDS = {
    "Goa": {
        "lat": 15.2993, "lon": 74.1240,
        "image": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?q=80&w=1200",
        "spots": [
            {"name": "Baga Beach", "lat": 15.5553, "lon": 73.7517},
            {"name": "Aguada Fort", "lat": 15.4920, "lon": 73.7737},
            {"name": "Dudhsagar Falls", "lat": 15.3144, "lon": 74.3143}
        ]
    },
    "Manali": {
        "lat": 32.2432, "lon": 77.1892,
        "image": "https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?q=80&w=1200",
        "spots": [
            {"name": "Solang Valley", "lat": 32.3166, "lon": 77.1578},
            {"name": "Hadimba Temple", "lat": 32.2483, "lon": 77.1804},
            {"name": "Rohtang Pass", "lat": 32.3716, "lon": 77.2466}
        ]
    },
    "Jaipur": {
        "lat": 26.9124, "lon": 75.7873,
        "image": "https://images.unsplash.com/photo-1599661046289-e31897793e14?q=80&w=1200",
        "spots": [
            {"name": "Hawa Mahal", "lat": 26.9239, "lon": 75.8267},
            {"name": "Amer Fort", "lat": 26.9855, "lon": 75.8513},
            {"name": "City Palace", "lat": 26.9258, "lon": 75.8237}
        ]
    },
    "Kerala": {
        "lat": 9.9312, "lon": 76.2673,
        "image": "https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?q=80&w=1200",
        "spots": [
            {"name": "Alleppey Backwaters", "lat": 9.4981, "lon": 76.3388},
            {"name": "Munnar Tea Gardens", "lat": 10.0889, "lon": 77.0595},
            {"name": "Kochi Fort", "lat": 9.9656, "lon": 76.2421}
        ]
    },
    "Mumbai": {
        "lat": 19.0760, "lon": 72.8777,
        "image": "https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=1200",
        "spots": [
            {"name": "Gateway of India", "lat": 18.9220, "lon": 72.8347},
            {"name": "Marine Drive", "lat": 18.9440, "lon": 72.8237},
            {"name": "Elephanta Caves", "lat": 18.9633, "lon": 72.9315}
        ]
    },
    "Agra": {
        "lat": 27.1767, "lon": 78.0081,
        "image": "https://images.unsplash.com/photo-1564507592333-c60657eea523?q=80&w=1200",
        "spots": [
            {"name": "Taj Mahal", "lat": 27.1751, "lon": 78.0421},
            {"name": "Agra Fort", "lat": 27.1795, "lon": 78.0211},
            {"name": "Fatehpur Sikri", "lat": 27.0945, "lon": 77.6695}
        ]
    },
    "Varanasi": {
        "lat": 25.3176, "lon": 82.9739,
        "image": "https://images.unsplash.com/photo-1561361058-c24e017c5e9b?q=80&w=1200",
        "spots": [
            {"name": "Dashashwamedh Ghat", "lat": 25.3068, "lon": 83.0109},
            {"name": "Kashi Vishwanath Temple", "lat": 25.3109, "lon": 83.0107},
            {"name": "Sarnath", "lat": 25.3814, "lon": 83.0234}
        ]
    },
    "Leh Ladakh": {
        "lat": 34.1526, "lon": 77.5771,
        "image": "https://images.unsplash.com/photo-1605649461784-edc96e3b3a5f?q=80&w=1200",
        "spots": [
            {"name": "Pangong Lake", "lat": 33.7608, "lon": 78.6445},
            {"name": "Leh Palace", "lat": 34.1642, "lon": 77.5857},
            {"name": "Nubra Valley", "lat": 34.6494, "lon": 77.5619}
        ]
    },
}


def get_flight_estimates(origin, destination):
    return [
        {"Airline": "IndiGo Express", "Flight No": "6E-401", "Departure": "07:30 AM", "Est. Fare": "₹4,800"},
        {"Airline": "Air India Premier", "Flight No": "AI-809", "Departure": "01:15 PM", "Est. Fare": "₹5,600"},
        {"Airline": "Akasa Air", "Flight No": "QP-112", "Departure": "06:45 PM", "Est. Fare": "₹4,200"}
    ]


def get_hotel_estimates(destination, max_budget):
    return [
        {"Property": "Grand Horizon Resort", "Rating": "4.8 ⭐", "Avg. Nightly Rate": f"₹{min(max_budget, 4500)}"},
        {"Property": "Urban Pearl Boutique", "Rating": "4.3 ⭐", "Avg. Nightly Rate": f"₹{min(max_budget, 2800)}"},
        {"Property": "Backpackers Hideout", "Rating": "4.0 ⭐", "Avg. Nightly Rate": f"₹{min(max_budget, 1500)}"}
    ]


def get_weather_data(city_name, api_key_str):
    """Fetch live weather. Returns (data_dict, error_str) — one will always be None."""
    if not api_key_str:
        return None, None  # key not configured, caller shows mock silently
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key_str}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json(), None
        if response.status_code == 401:
            return None, "Invalid OpenWeather API key (401). Check OPENWEATHER_API_KEY in secrets.toml."
        if response.status_code == 404:
            return None, f"City '{city_name}' not found on OpenWeatherMap (404)."
        return None, f"OpenWeather API error {response.status_code}: {response.text[:120]}"
    except requests.exceptions.Timeout:
        return None, "Weather API request timed out. Showing sample data."
    except requests.exceptions.ConnectionError:
        return None, "Could not reach OpenWeather API. Check network connectivity."
    except Exception as exc:
        return None, f"Unexpected weather error: {exc}"


def generate_packing_list(temp, condition):
    items = [
        "🪪 ID Proofs, Driver's License & Booking Passes",
        "📱 Phone Charger, Power Bank & Universal Adapter",
        "💊 Basic First-Aid Kit & Personal Medications",
        "🧴 Sunscreen & Personal Toiletries"
    ]
    if temp < 15:
        items.extend(["🧥 Heavy Jackets, Thermal Wear & Sweaters", "🧣 Muffler, Gloves & Woolen Socks", "☕ Lip Balm & Moisturizer"])
    elif 15 <= temp <= 28:
        items.extend(["👕 Comfortable Cotton T-shirts & Jeans", "👟 Walking / Hiking Shoes", "🕶️ Sunglasses & Light Jacket"])
    else:
        items.extend(["🩳 Breathable Cotton/Linen Clothes", "🧢 Cap/Hat & Sun Protection", "💧 Reusable Water Bottle"])

    if any(w in condition.lower() for w in ["rain", "drizzle", "thunderstorm"]):
        items.extend(["☂️ Umbrella / Raincoat", "🎒 Waterproof Backpack Cover", "🩴 Quick-dry Footwear"])

    return items


# FIX: cache the ModelInference client so IAM auth happens only once per unique
# set of credentials, not on every generate call.
@st.cache_resource(show_spinner=False)
def _get_model(api_key_val: str, project_id_val: str, region_val: str) -> ModelInference:
    credentials = {
        "url": f"https://{region_val}.ml.cloud.ibm.com",
        "apikey": api_key_val,
    }
    return ModelInference(
        model_id="ibm/granite-4-h-small",
        credentials=credentials,
        project_id=project_id_val,
    )


def _classify_watsonx_error(exc: Exception) -> str:
    """Return a user-friendly message for known IBM watsonx / IAM error patterns."""
    msg = str(exc)
    msg_lower = msg.lower()
    if "401" in msg or "unauthorized" in msg_lower or "invalid api key" in msg_lower:
        return (
            "🔑 Authentication failed (401). Your IBM IAM API Key is invalid or expired. "
            "Generate a new key at cloud.ibm.com → Manage → Access → API Keys."
        )
    if "403" in msg or "forbidden" in msg_lower or "project" in msg_lower:
        return (
            "🚫 Access denied (403). Verify your watsonx Project ID is correct and that "
            "the API key has 'Editor' or 'Admin' access to that project."
        )
    if "404" in msg or "not found" in msg_lower or "model" in msg_lower:
        return (
            "🤖 Model not found (404). 'ibm/granite-4-h-small' may not be available in "
            f"region '{region}'. Try switching to 'us-south'."
        )
    if "429" in msg or "too many requests" in msg_lower or "rate limit" in msg_lower:
        return "⏳ Rate limit hit (429). Too many requests — wait a moment and try again."
    if "quota" in msg_lower or "resource units" in msg_lower:
        return "📊 Quota exceeded. Your watsonx resource units are exhausted for this billing period."
    if "timeout" in msg_lower or "timed out" in msg_lower:
        return "⌛ Request timed out. IBM watsonx did not respond in time. Please retry."
    if "connection" in msg_lower or "network" in msg_lower or "unreachable" in msg_lower:
        return "🌐 Network error. Could not reach IBM watsonx. Check your internet connection."
    # Fallback — include the raw message for diagnostics
    return f"❌ IBM Granite error: {msg}"


# FIX: explicit parameters instead of closing over widget variables
def call_granite_model(
    prompt_text: str,
    api_key_val: str,
    project_id_val: str,
    region_val: str,
    temperature_val: float,
    max_tokens_val: int,
) -> str:
    model = _get_model(api_key_val, project_id_val, region_val)
    result = model.generate_text(
        prompt=prompt_text,
        params={
            GenParams.MAX_NEW_TOKENS: max_tokens_val,
            GenParams.TEMPERATURE: temperature_val,
        },
    )
    # FIX: validate the response is a non-empty string
    if not result or not result.strip():
        raise ValueError(
            "The model returned an empty response. This may be due to a content "
            "filter or an overly restrictive prompt. Please rephrase and try again."
        )
    return result


# --- INTERACTIVE FORM ---
st.markdown("""
    <div class="glass-card">
        <div class="section-header">✈️ Trip Configurator</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("**🔥 Quick Pick a Destination:**")
chip_cols = st.columns(8)

# Quick-pick chips for all destinations
_CHIPS = [
    ("🏖️", "Goa"), ("🏔️", "Manali"), ("🏰", "Jaipur"), ("🌴", "Kerala"),
    ("🏙️", "Mumbai"), ("🕌", "Agra"), ("🛕", "Varanasi"), ("🏔️", "Leh Ladakh"),
]
for i, (icon, name) in enumerate(_CHIPS):
    if chip_cols[i].button(f"{icon} {name}"):
        st.session_state["selected_chip"] = name
        st.session_state["custom_dest"] = ""   # clear custom when chip chosen

col1, col2, col3 = st.columns(3)
with col1:
    origin = st.text_input("🛫 Origin City", "Delhi")
    dest_keys = list(DESTINATION_COORDS.keys())
    chip_val = st.session_state.get("selected_chip")
    default_idx = dest_keys.index(chip_val) if chip_val in dest_keys else 0
    destination = st.selectbox("🛬 Destination City", dest_keys, index=default_idx)
    # Custom destination override — allows any city not in the preset list
    custom_dest = st.text_input(
        "✏️ Or type a custom destination",
        value=st.session_state.get("custom_dest", ""),
        placeholder="e.g. Shimla, Rishikesh, Udaipur…",
        key="custom_dest_input"
    )
    if custom_dest.strip():
        destination = custom_dest.strip()
        st.session_state["custom_dest"] = custom_dest.strip()

with col2:
    duration = st.number_input("📅 Duration (Days)", min_value=1, max_value=14, value=3)
    budget = st.number_input("💰 Total Budget (INR)", min_value=3000, value=20000, step=1000)

with col3:
    hotel_budget = st.number_input("🏨 Max Hotel / Night (INR)", min_value=1000, value=4500, step=500)
    travel_style = st.selectbox("✨ Travel Style", ["Relaxed & Chill", "Action-Packed Adventure", "Cultural Explorer", "Luxury & Pampering", "Budget Backpacker"])

st.markdown("<br>", unsafe_allow_html=True)
generate_btn = st.button("🚀 Generate My AI Travel Plan", use_container_width=True)

if generate_btn:
    if not api_key.strip() or not project_id.strip():
        st.error("🔑 Please enter a valid IBM IAM API Key and Project ID in the sidebar.")
    else:
        with st.spinner("🤖 Agent orchestrating APIs & Reasoning with IBM Granite..."):
            try:
                flights = get_flight_estimates(origin, destination)
                hotels = get_hotel_estimates(destination, hotel_budget)

                prompt = f"""<|system|>
You are an intelligent AI Travel Planner Agent built on IBM Granite. Generate an exciting, detailed day-by-day travel plan formatted strictly in Markdown text. Do NOT generate Java or Python code blocks.

<|user|>
Create a comprehensive, optimised trip plan:
- Route: {origin} to {destination}
- Duration: {duration} Days
- Total Budget: ₹{budget:,}
- Hotel Budget: ₹{hotel_budget:,}/night
- Travel Style: {travel_style}
Transport Options: {json.dumps(flights)}
Hotel Choices: {json.dumps(hotels)}

Format clearly with Markdown headers (###) and emojis:
1. ✨ Overview & Highlights
2. ✈️ Recommended Flight & Hotel Selection (with reasoning)
3. 🗓️ Day-by-Day Optimised Schedule (include best time-of-day for each activity to avoid crowds and save travel time)
4. 🍽️ Local Culinary Spots & Street Food Guide
5. 🗺️ Local Guides & Hidden Gems (lesser-known spots a local would recommend)
6. ⚡ Schedule Optimisation Tips (how to rearrange days to save time, money or avoid peak hours)

<|assistant|>
"""
                response = call_granite_model(
                    prompt,
                    api_key.strip(),
                    project_id.strip(),
                    region,
                    temperature,
                    max_tokens,
                )

                st.session_state["trip_generated"] = True
                st.session_state["itinerary_result"] = response
                st.session_state["flights"] = flights
                st.session_state["hotels"] = hotels
                st.session_state["current_dest"] = destination
                st.session_state["origin"] = origin
                st.session_state["duration"] = duration
                st.session_state["budget"] = budget
                st.session_state["hotel_budget"] = hotel_budget
                st.session_state["travel_style"] = travel_style
                st.session_state["chat_messages"] = [
                    {"role": "assistant", "content": f"Hi! I'm your AI Concierge. How can I adjust your trip to {destination}?"}
                ]
                st.balloons()
            except Exception as e:
                st.error(_classify_watsonx_error(e))

# --- DISPLAY DASHBOARD (PERSISTS ON CHAT RERUN) ---
if st.session_state.get("trip_generated", False):
    st.markdown("<br>", unsafe_allow_html=True)

    _dest_key = st.session_state["current_dest"]
    if _dest_key in DESTINATION_COORDS:
        dest_info = DESTINATION_COORDS[_dest_key]
    else:
        # Dynamically fetch an image matching the destination name via Unsplash Source
        _img_query = _dest_key.replace(" ", "%20")
        dest_info = {
            "lat": 20.5937, "lon": 78.9629,
            "image": f"https://source.unsplash.com/1200x400/?{_img_query},travel",
            "spots": [{"name": _dest_key, "lat": 20.5937, "lon": 78.9629}]
        }

    # Hero destination image with overlay
    st.markdown(f"""
        <div style="position:relative; border-radius:20px; overflow:hidden; margin-bottom:1.5rem; box-shadow:0 16px 48px rgba(0,0,0,0.4);">
            <img src="{dest_info['image']}" style="width:100%; height:320px; object-fit:cover; display:block;" />
            <div style="position:absolute; inset:0; background:linear-gradient(to top, rgba(5,10,40,0.85) 0%, transparent 55%);"></div>
            <div style="position:absolute; bottom:1.5rem; left:1.8rem;">
                <div style="font-size:2rem; font-weight:800; color:white; text-shadow:0 2px 8px rgba(0,0,0,0.5);">
                    📍 {st.session_state['current_dest']}
                </div>
                <div style="font-size:1rem; color:rgba(255,255,255,0.75); margin-top:0.2rem;">
                    {st.session_state['origin']} &nbsp;→&nbsp; {st.session_state['current_dest']} &nbsp;·&nbsp; {st.session_state['duration']} Days &nbsp;·&nbsp; {st.session_state['travel_style']}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🛫 Route", f"{st.session_state['origin']} → {st.session_state['current_dest']}")
    m2.metric("📅 Duration", f"{st.session_state['duration']} Days")
    m3.metric("💰 Budget", f"₹{st.session_state['budget']:,}")
    m4.metric("✨ Style", st.session_state["travel_style"])

    st.markdown("<br>", unsafe_allow_html=True)

    _TAB_LABELS = [
        "✨ AI Itinerary",
        "🗺️ Route Map",
        "📊 Budget Allocation",
        "🌦️ Weather & Packing",
        "💬 AI Re-Planner",
    ]
    _default_tab = st.session_state.get("active_tab", "✨ AI Itinerary")
    tab_itinerary, tab_map, tab_budget, tab_weather, tab_chat = st.tabs(
        _TAB_LABELS,
        default=_default_tab,
    )

    # 1. ITINERARY TAB
    with tab_itinerary:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown(st.session_state["itinerary_result"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Flight & Hotel tables ──────────────────────────────────────
        fi_col, ho_col = st.columns(2)
        with fi_col:
            st.markdown("#### ✈️ Available Flights")
            st.dataframe(pd.DataFrame(st.session_state["flights"]), use_container_width=True, hide_index=True)
        with ho_col:
            st.markdown("#### 🏨 Hotel Options")
            st.dataframe(pd.DataFrame(st.session_state["hotels"]), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Booking confirmation & alert simulation ────────────────────
        st.markdown("#### 📋 Booking Summary & Alerts")
        travel_date = st.date_input(
            "📅 Select Travel Date",
            value=datetime.date.today() + datetime.timedelta(days=14),
            key="travel_date_input"
        )
        days_until = (travel_date - datetime.date.today()).days
        if days_until < 0:
            st.error("⚠️ Travel date is in the past. Please select a future date.")
        elif days_until <= 3:
            st.warning(f"🚨 **Last-minute Alert!** Your trip is in {days_until} day(s). Book immediately — prices may be higher and availability limited.")
        elif days_until <= 7:
            st.warning(f"⏰ **Booking Reminder:** Your trip is in {days_until} days. Confirm your hotel and flight bookings soon.")
        elif days_until <= 30:
            st.info(f"📌 **Plan Ahead:** {days_until} days until departure. Great time to book flights and accommodation for the best rates.")
        else:
            st.success(f"✅ **Plenty of time!** {days_until} days until your trip. Set calendar reminders and watch for early-bird deals.")

        booking_confirmed = st.checkbox("✅ Mark flights & hotel as booked", key="booking_confirmed")
        if booking_confirmed:
            st.success(f"🎉 Booking confirmed for **{st.session_state['current_dest']}** on **{travel_date.strftime('%d %B %Y')}**! Have a wonderful trip!")

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Itinerary (Markdown/Text)",
            data=st.session_state["itinerary_result"],
            file_name=f"{st.session_state['current_dest']}_Travel_Itinerary.md",
            mime="text/markdown"
        )

    # 2. MAP TAB
    with tab_map:
        st.markdown(f"### 🗺️ Destination Map — {st.session_state['current_dest']}")
        st.caption(f"Explore key attractions and points of interest around {st.session_state['current_dest']}.")
        df_spots = pd.DataFrame(dest_info["spots"])

        view_state = pdk.ViewState(
            latitude=dest_info["lat"],
            longitude=dest_info["lon"],
            zoom=10,
            pitch=45
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_spots,
            get_position="[lon, lat]",
            get_color="[15, 98, 254, 200]",
            get_radius=1200,
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}"}
        ))
        st.dataframe(df_spots[["name", "lat", "lon"]], use_container_width=True)

    # 3. BUDGET TAB
    with tab_budget:
        st.markdown("### 📊 Estimated Budget Breakdown")

        b_total = st.session_state["budget"]
        b_duration = st.session_state["duration"]
        b_hotel = st.session_state["hotel_budget"]

        est_transport = min(b_total * 0.3, 8000)
        est_hotel = b_hotel * b_duration
        est_food = b_total * 0.25
        est_activities = max(0, b_total - (est_transport + est_hotel + est_food))

        # Summary metric row
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("✈️ Transport", f"₹{est_transport:,.0f}")
        bc2.metric("🏨 Hotel Total", f"₹{est_hotel:,.0f}")
        bc3.metric("🍽️ Food", f"₹{est_food:,.0f}")
        bc4.metric("🎭 Activities", f"₹{est_activities:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        chart_data = pd.DataFrame({
            "Category": ["Transport", "Accommodation", "Food & Dining", "Activities & Sightseeing"],
            "Estimated Cost (INR)": [est_transport, est_hotel, est_food, est_activities]
        })

        st.bar_chart(chart_data, x="Category", y="Estimated Cost (INR)", color="#0f62fe")

        remaining = b_total - (est_transport + est_hotel + est_food)
        if remaining > 0:
            st.success(f"✅ Estimated remaining buffer: **₹{remaining:,.0f}** — great headroom for spontaneous experiences!")
        else:
            st.warning("⚠️ Your hotel budget may exceed total budget for this trip duration. Consider adjusting.")

    # 4. WEATHER & PACKING TAB
    with tab_weather:
        st.markdown(f"### 🌤️ Live Weather — {st.session_state['current_dest']}")

        # FIX: get_weather_data now returns (data, error) tuple
        w_data, w_error = get_weather_data(st.session_state["current_dest"], weather_api_key)

        if w_data:
            w_temp = w_data["main"]["temp"]
            w_desc = w_data["weather"][0]["description"].title()
            w_humidity = w_data["main"]["humidity"]
            w_wind = w_data["wind"]["speed"]
        else:
            if w_error:
                st.warning(f"⚠️ {w_error} Showing sample data.")
            else:
                st.info("ℹ️ Live weather disabled. Add OPENWEATHER_API_KEY to secrets.toml for real data.")
            w_temp, w_desc, w_humidity, w_wind = 26.0, "Partly Cloudy", 60, 12.0

        # Custom weather cards
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.markdown(f'<div class="weather-card"><div class="wc-label">🌡️ Temperature</div><div class="wc-value">{w_temp}°<span class="wc-unit">C</span></div></div>', unsafe_allow_html=True)
        wc2.markdown(f'<div class="weather-card"><div class="wc-label">🌤️ Condition</div><div class="wc-value" style="font-size:1.15rem">{w_desc}</div></div>', unsafe_allow_html=True)
        wc3.markdown(f'<div class="weather-card"><div class="wc-label">💧 Humidity</div><div class="wc-value">{w_humidity}<span class="wc-unit">%</span></div></div>', unsafe_allow_html=True)
        wc4.markdown(f'<div class="weather-card"><div class="wc-label">💨 Wind Speed</div><div class="wc-value">{w_wind}<span class="wc-unit"> m/s</span></div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🧳 Smart Packing Checklist")
        st.caption("Dynamically generated based on current weather conditions.")

        packing_items = generate_packing_list(w_temp, w_desc)
        cols = st.columns(2)
        for idx, item in enumerate(packing_items):
            with cols[idx % 2]:
                st.checkbox(item, value=False, key=f"pack_item_{idx}")

    # 5. CHATBOT TAB
    with tab_chat:
        st.session_state["active_tab"] = "💬 AI Re-Planner"
        st.markdown("### 💬 AI Travel Concierge")
        st.caption("Ask questions, request changes, or explore alternatives — powered by IBM Granite 4.")

        for msg in st.session_state.get("chat_messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_prompt := st.chat_input("e.g. 'Add vegetarian street food recommendations for Day 1'"):
            st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
            st.session_state["active_tab"] = "💬 AI Re-Planner"

            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Re-evaluating plan with IBM Granite..."):
                    try:
                        bot_prompt = f"""<|system|>
You are an AI travel concierge. Use the trip context below to answer user queries conversationally in raw text.

Trip Context:
Destination: {st.session_state['current_dest']}, Duration: {st.session_state['duration']} Days, Budget: ₹{st.session_state['budget']}.

User Question: {user_prompt}
<|assistant|>
"""
                        bot_response = call_granite_model(
                            bot_prompt,
                            api_key.strip(),
                            project_id.strip(),
                            region,
                            temperature,
                            max_tokens,
                        )
                        st.markdown(bot_response)
                        st.session_state["chat_messages"].append({"role": "assistant", "content": bot_response})
                    except Exception as err:
                        st.error(_classify_watsonx_error(err))
