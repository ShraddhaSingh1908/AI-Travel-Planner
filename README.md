# 🌍 Wanderlust Agentic AI — IBM Granite Travel Planner

> **AICTE 2026 · Problem Statement 5**  
> An autonomous AI travel planning agent powered by **IBM Granite 4** and built with **Streamlit**.

---

## ✨ Features

| Module | Description |
|---|---|
| ✈️ AI Itinerary Generator | Day-by-day personalised travel plans via IBM Granite |
| 💰 Financial & Cost Breakdown | Allocation across accommodation, food, local travel, sightseeing & misc |
| 🗺️ Interactive Map | PyDeck-powered destination map with tourist spots |
| 🌤️ Live Weather | Real-time weather via OpenWeatherMap API |
| 🧳 Smart Packing List | Weather-aware auto-generated checklist |
| 💬 AI Concierge Chatbot | In-app chat to modify and refine your plan |
| 🖼️ Destination Images | Auto-fetched via Wikimedia Commons / Unsplash |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
Copy the example secrets file and fill in your keys:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:
```toml
IBM_API_KEY          = "your-ibm-iam-api-key"
WATSONX_PROJECT_ID   = "your-watsonx-project-id"
OPENWEATHER_API_KEY  = "your-openweathermap-api-key"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🔑 Getting API Keys

| Key | Where to get it |
|---|---|
| `IBM_API_KEY` | [cloud.ibm.com](https://cloud.ibm.com) → Manage → Access → API Keys |
| `WATSONX_PROJECT_ID` | [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → Your Project → Manage → General |
| `OPENWEATHER_API_KEY` | [openweathermap.org/api](https://openweathermap.org/api) → Free tier |

---

## 🔒 Security

- **Never commit `.streamlit/secrets.toml`** — it is already in `.gitignore`
- All credentials are loaded via `st.secrets` (Streamlit's secure secrets manager)
- For Streamlit Cloud deployment, add secrets in the app's **Settings → Secrets** panel

---

## 📦 Requirements

```
streamlit>=1.30.0
ibm-watsonx-ai>=1.1.0,<2.0.0
requests>=2.31.0
pandas>=2.0.0
pydeck>=0.8.0
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub (without `secrets.toml`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set `app.py` as the main file
4. Under **Settings → Secrets**, paste:
```toml
IBM_API_KEY          = "..."
WATSONX_PROJECT_ID   = "..."
OPENWEATHER_API_KEY  = "..."
```
5. Click **Deploy** ✅

---

## 📸 Screenshots

> Add your screenshots here after deployment.

---

## 🤖 Powered By

- [IBM Granite 4](https://www.ibm.com/granite) — `ibm/granite-4-h-small`
- [IBM watsonx.ai](https://www.ibm.com/watsonx)
- [Streamlit](https://streamlit.io)
- [OpenWeatherMap](https://openweathermap.org)
- [PyDeck](https://deckgl.readthedocs.io)

---

*Made with ❤️ for AICTE 2026*
