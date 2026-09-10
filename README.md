# 🌍 Wanderlust Agentic AI — Intelligent Travel Planner

[![IBM Granite](https://img.shields.io/badge/Model-IBM%20Granite%204-052125?style=for-the-badge&logo=ibm)](https://www.ibm.com/granite)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![watsonx.ai](https://img.shields.io/badge/Platform-IBM%20watsonx.ai-1261FE?style=for-the-badge&logo=ibm)](https://www.ibm.com/watsonx)

An autonomous, multi-modal travel planning platform built for **AICTE Problem Statement No. 5**. Powered by **IBM Granite 4** foundation models on **watsonx.ai**, this application orchestrates destination mapping, live weather forecasting, budget allocation, and dynamic itinerary generation with a persistent real-time conversational agent.

---

## ✨ Key Features

* **🤖 Autonomous Itinerary Engine:** Uses `ibm/granite-4-h-small` to generate structured, personalized day-by-day itineraries based on travel style, route, and constraints.
* **💬 Persistent AI Re-Planner:** Interactive chat concierge capable of revising itineraries on the fly while retaining session state.
* **🗺️ 3D Visual Mapping:** Interactive landmark maps powered by **PyDeck** and geospatial coordinate mapping.
* **🌤️ Weather & Dynamic Packing:** Integrates OpenWeatherMap APIs to deliver live condition updates paired with temperature- and rain-aware packing checklists.
* **📊 Expense Allocation Analytics:** Automated budget breakdown visualizations across transport, lodging, dining, and activities.
* **🎨 Modern SaaS Dashboard:** Custom-styled Streamlit UI featuring destination spotlight banners, downloadable itineraries (`.md`), and quick-pick destination chips.

---

## 🛠️ Tech Stack

* **LLM Core:** IBM Granite (`ibm/granite-4-h-small`) via `ibm-watsonx-ai`
* **Frontend / UI:** Streamlit with custom CSS
* **Geospatial & Charts:** PyDeck, Pandas
* **API Integrations:** OpenWeatherMap REST API, IBM IAM Authentication

---

## 🚀 Quickstart Guide

### 1. Prerequisites

Ensure you have Python 3.10+ installed along with active credentials for:
1. **IBM Cloud / watsonx.ai** (IAM API Key & Project ID)
2. **OpenWeatherMap API** (Optional, for live weather sync)

### 2. Running Locally

```bash
# Clone repository
git clone [https://github.com/ShraddhaSingh1908/agentic-ai-travel-planner.git](https://github.com/ShraddhaSingh1908/agentic-ai-travel-planner.git)
cd agentic-ai-travel-planner

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
