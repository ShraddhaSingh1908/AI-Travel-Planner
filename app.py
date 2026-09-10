import streamlit as st
import json
import time
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

# --- ADVANCED CUSTOM CSS FOR SAAS UI LOOK ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0f62fe 0%, #1192e8 50%, #001d6c 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15, 98, 254, 0.25);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero-tagline {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    .result-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border-left: 6px solid #0f62fe;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0f62fe 0%, #1192e8 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.7rem 1.8rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🌟 Wanderlust Agentic AI</div>
        <div class="hero-tagline">Autonomous Travel Planning, Visual Mapping & Live Weather with <b>IBM Granite 4</b></div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ENGINE CONTROL ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=110)
    st.title("⚙️ Engine Control")
    st.caption("AICTE 2026 Problem Statement 5")
    st.markdown("---")
    
    st.subheader("🔑 Credentials")
    
    try:
        default_key = st.secrets.get("IBM_API_KEY", "")
        default_pid = st.secrets.get("WATSONX_PROJECT_ID", "c4ad81f3-bec2-4d0f-8369-fdb3ac4c56fd")
        weather_api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
    except Exception:
        default_key = ""
        default_pid = "c4ad81f3-bec2-4d0f-8369-fdb3ac4c56fd"
        weather_api_key = ""

    api_key = st.text_input("IBM IAM API Key", value=default_key, type="password")
    project_id = st.text_input("watsonx Project ID", value=default_pid)
    region = st.selectbox("Cloud Region", ["us-south", "eu-de"], index=0)

    st.markdown("---")
    st.markdown("### 🎛️ Agent Parameters")
    temperature = st.slider("Creativity (Temperature)", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Output Tokens", min_value=300, max_value=2500, value=1500, step=100)

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
    }
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
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key_str}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

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

def call_granite_model(prompt_text):
    credentials = {
        "url": f"https://{region}.ml.cloud.ibm.com",
        "apikey": api_key.strip()
    }
    model = ModelInference(
        model_id="ibm/granite-4-h-small",
        params={GenParams.MAX_NEW_TOKENS: max_tokens, GenParams.TEMPERATURE: temperature},
        credentials=credentials,
        project_id=project_id.strip()
    )
    return model.generate_text(prompt=prompt_text)

# --- INTERACTIVE FORM ---
st.subheader("✈️ Interactive Trip Configurator")

st.write("🔥 **Quick Pick Destination:**")
chip_cols = st.columns(4)
selected_chip = None

if chip_cols[0].button("🏖️ Goa"):
    selected_chip = "Goa"
if chip_cols[1].button("🏔️ Manali"):
    selected_chip = "Manali"
if chip_cols[2].button("🏰 Jaipur"):
    selected_chip = "Jaipur"
if chip_cols[3].button("🌴 Kerala"):
    selected_chip = "Kerala"

col1, col2, col3 = st.columns(3)
with col1:
    origin = st.text_input("🛫 Origin City", "Delhi")
    dest_keys = list(DESTINATION_COORDS.keys())
    default_idx = dest_keys.index(selected_chip) if selected_chip in dest_keys else 0
    destination = st.selectbox("🛬 Destination City", dest_keys, index=default_idx)
    
with col2:
    duration = st.number_input("📅 Duration (Days)", min_value=1, max_value=14, value=3)
    budget = st.number_input("💰 Total Budget (INR)", min_value=3000, value=20000, step=1000)
    
with col3:
    hotel_budget = st.number_input("🏨 Max Hotel / Night (INR)", min_value=1000, value=4500, step=500)
    travel_style = st.selectbox("✨ Travel Style", ["Relaxed & Chill", "Action-Packed Adventure", "Cultural Explorer", "Luxury & Pampering", "Budget Backpacker"])

st.markdown("<br>", unsafe_allow_html=True)
generate_btn = st.button("🚀 Execute Agent & Generate Plan", use_container_width=True)

# Save values into session state on submit
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
Create a trip plan:
- Route: {origin} to {destination}
- Duration: {duration} Days
- Total Budget: ₹{budget:,}
- Hotel Budget: ₹{hotel_budget:,}/night
- Travel Style: {travel_style}
Transport Options: {json.dumps(flights)}
Hotel Choices: {json.dumps(hotels)}

Format clearly with Markdown headers (###) and emojis:
1. ✨ Overview & Highlights
2. ✈️ Flight & Hotel Selection
3. 🗓️ Day-by-Day Detailed Schedule
4. 🍽️ Local Culinary Spots & Essential Tips

<|assistant|>
"""
                response = call_granite_model(prompt)
                
                # PERSIST STATE SO RERUNS DO NOT RESET IT
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
                st.error(f"❌ Error connecting to IBM Granite Engine: {str(e)}")

# --- DISPLAY DASHBOARD (PERSISTS ON CHAT RERUN) ---
if st.session_state.get("trip_generated", False):
    st.markdown("---")
    
    # Destination Banner Image
    dest_info = DESTINATION_COORDS.get(st.session_state['current_dest'], DESTINATION_COORDS["Goa"])
    st.image(
        dest_info["image"], 
        caption=f"📸 Destination Spotlight: {st.session_state['current_dest']}", 
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route", f"{st.session_state['origin']} ➔ {st.session_state['current_dest']}")
    m2.metric("Duration", f"{st.session_state['duration']} Days")
    m3.metric("Budget", f"₹{st.session_state['budget']:,}")
    m4.metric("Style", st.session_state['travel_style'])

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs definition
    tab_itinerary, tab_map, tab_budget, tab_weather, tab_chat = st.tabs([
        "✨ AI Itinerary", 
        "🗺️ Route Map", 
        "📊 Budget Allocation", 
        "🌦️ Weather & Packing",
        "💬 AI Re-Planner"
    ])

    # 1. ITINERARY TAB
    with tab_itinerary:
        st.markdown(f'<div class="result-container">{st.session_state["itinerary_result"]}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Itinerary (Markdown/Text)",
            data=st.session_state["itinerary_result"],
            file_name=f"{st.session_state['current_dest']}_Travel_Itinerary.md",
            mime="text/markdown"
        )

    # 2. MAP TAB
    with tab_map:
        st.subheader(f"🗺️ Destination Map & Spots: {st.session_state['current_dest']}")
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
        st.subheader("📊 Estimated Expense Allocation")
        
        b_total = st.session_state['budget']
        b_duration = st.session_state['duration']
        b_hotel = st.session_state['hotel_budget']
        
        est_transport = min(b_total * 0.3, 8000)
        est_hotel = b_hotel * b_duration
        est_food = b_total * 0.25
        est_activities = max(0, b_total - (est_transport + est_hotel + est_food))

        chart_data = pd.DataFrame({
            "Category": ["Transport", "Accommodation", "Food & Dining", "Activities & Sightseeing"],
            "Estimated Cost (INR)": [est_transport, est_hotel, est_food, est_activities]
        })
        
        st.bar_chart(chart_data, x="Category", y="Estimated Cost (INR)", color="#0f62fe")

    # 4. WEATHER & PACKING TAB
    with tab_weather:
        st.subheader(f"🌤️ Live Weather Forecast for {st.session_state['current_dest']}")
        
        w_data = get_weather_data(st.session_state['current_dest'], weather_api_key) if weather_api_key else None
        
        if w_data:
            w_temp = w_data["main"]["temp"]
            w_desc = w_data["weather"][0]["description"].title()
            w_humidity = w_data["main"]["humidity"]
            w_wind = w_data["wind"]["speed"]
        else:
            st.info("ℹ️ Displaying sample weather preview. Verify OPENWEATHER_API_KEY in secrets.toml for live sync.")
            w_temp, w_desc, w_humidity, w_wind = 26.0, "Partly Cloudy", 60, 12.0

        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        w_col1.metric("Temperature", f"{w_temp}°C")
        w_col2.metric("Condition", w_desc)
        w_col3.metric("Humidity", f"{w_humidity}%")
        w_col4.metric("Wind Speed", f"{w_wind} m/s")

        st.markdown("---")
        st.subheader("🧳 Dynamic AI Packing Checklist")
        
        packing_items = generate_packing_list(w_temp, w_desc)
        cols = st.columns(2)
        for idx, item in enumerate(packing_items):
            with cols[idx % 2]:
                st.checkbox(item, value=False, key=f"pack_item_{idx}")

    # 5. CHATBOT TAB (NOW PERSISTENT)
    with tab_chat:
        st.subheader("💬 Re-Plan & Chat with IBM Granite Agent")
        st.caption("Ask questions or request real-time itinerary modifications!")

        # Display history
        for msg in st.session_state.get("chat_messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input handles re-runs without losing dashboard state
        if user_prompt := st.chat_input("e.g. 'Add vegetarian street food recommendations for Day 1'"):
            st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
            
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
                        bot_response = call_granite_model(bot_prompt)
                        st.markdown(bot_response)
                        st.session_state["chat_messages"].append({"role": "assistant", "content": bot_response})
                        st.rerun()  # Clean refresh within the tab
                    except Exception as err:
                        st.error(f"Re-planning failed: {str(err)}")