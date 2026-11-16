import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import requests

# ------------------------------------------------------------
# AQUAGROW - AI SMART IRRIGATION PLATFORM
# Demo application: simulated sensors + Open-Meteo weather API
# ------------------------------------------------------------

st.set_page_config(
    page_title="AquaGrow | Smart Irrigation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {background: #f5f8f4;}
    [data-testid="stSidebar"] {background: #123d32;}
    [data-testid="stSidebar"] * {color: #ffffff !important;}
    .hero {
        background: linear-gradient(110deg, #123d32, #26734d);
        padding: 25px 28px; border-radius: 18px; color: white;
        margin-bottom: 18px;
    }
    .hero h1 {color: white; margin: 0; font-size: 2.15rem;}
    .hero p {color: #e1f3e8; margin: 7px 0 0 0;}
    .metric-card {
        background: white; border: 1px solid #e2ebe2; border-radius: 14px;
        padding: 15px 17px; box-shadow: 0 2px 8px rgba(20,60,40,.04);
    }
    .metric-label {color: #61736a; font-size: .88rem;}
    .metric-value {color: #153e31; font-size: 1.75rem; font-weight: 750;}
    .metric-note {color: #718078; font-size: .78rem;}
    .section-title {color: #174b38; font-weight: 750; font-size: 1.2rem;}
    .good {background:#e6f5eb;color:#17633b;padding:10px 13px;border-radius:10px;}
    .warn {background:#fff3dc;color:#875600;padding:10px 13px;border-radius:10px;}
    .bad {background:#fde9e7;color:#a32e24;padding:10px 13px;border-radius:10px;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2ebe2; padding: 12px;
        border-radius: 12px;
    }
    .small-muted {color:#718078;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Plant-specific growing profiles ----------
# These are illustrative demo settings, not universal agronomic limits.
# Soil-moisture percentages depend on the sensor, soil/substrate and calibration.
PLANT_PROFILES = {
    "Tomato":   {"moisture_min": 40, "moisture_max": 60, "air_temp": 29.0, "humidity": 55, "soil_temp": 25.0, "soil_moisture": 42, "plant_factor": 1.10, "description": "Consistent moisture; avoid waterlogging."},
    "Mint":     {"moisture_min": 50, "moisture_max": 70, "air_temp": 26.0, "humidity": 65, "soil_temp": 22.0, "soil_moisture": 58, "plant_factor": 1.00, "description": "Prefers evenly moist soil."},
    "Lettuce":  {"moisture_min": 45, "moisture_max": 65, "air_temp": 22.0, "humidity": 60, "soil_temp": 19.0, "soil_moisture": 52, "plant_factor": 0.95, "description": "Shallow roots; keep the root zone evenly moist."},
    "Cucumber": {"moisture_min": 45, "moisture_max": 65, "air_temp": 28.0, "humidity": 65, "soil_temp": 24.0, "soil_moisture": 48, "plant_factor": 1.15, "description": "Needs regular moisture, especially while fruiting."},
    "Chilli":   {"moisture_min": 35, "moisture_max": 55, "air_temp": 30.0, "humidity": 50, "soil_temp": 26.0, "soil_moisture": 39, "plant_factor": 0.90, "description": "Allow the surface to dry slightly between watering."},
    "Other":    {"moisture_min": 35, "moisture_max": 60, "air_temp": 28.0, "humidity": 50, "soil_temp": 24.0, "soil_moisture": 38, "plant_factor": 1.00, "description": "Generic demonstration profile; configure for the actual plant."},
}

def default_sensor_values(plant_name):
    profile = PLANT_PROFILES[plant_name]
    return {
        "soil_moisture": profile["soil_moisture"],
        "air_temp": profile["air_temp"],
        "humidity": profile["humidity"],
        "soil_temp": profile["soil_temp"],
        "rain_sensor": 0,
    }

# ---------- Demo data and session state ----------
DEFAULTS = {
    "soil_moisture": 42,
    "air_temp": 29.0,
    "humidity": 55,
    "soil_temp": 25.0,
    "rain_sensor": 0,
    "water_today": 2.9,  # Illustrative daily plant-water use (target demo range: 2.7–3.2 L).
    "irrigation_cycles": 2,
    "pump_on": False,
    "mode": "Automatic",
    "emergency_stop": False,
    "history": None,
    "last_irrigation": datetime.now() - timedelta(hours=5),
    "location_name": "Al Ain, UAE",
    "latitude": 24.1917,
    "longitude": 55.7606,
    "plant": "Tomato",
    "soil_type": "Loamy",
    "growth_stage": "Flowering",
    "area_m2": 2.0,
    "plant_sensor_profiles": {name: default_sensor_values(name) for name in PLANT_PROFILES},
    "recycled_water_log": None,
    "recycled_tank_capacity": 100.0,
    "recycled_tank_level": 0.0,
    # Water-meter totals are cumulative litres reported by the ESP32.
    "wastewater_in_total_l": 0.0,
    "treated_out_total_l": 0.0,
    "wastewater_meter_previous_l": None,
    "treated_meter_previous_l": None,
    "wastewater_daily_log": None,
    "water_meter_mode": "Demo simulator",
    "esp32_water_endpoint": "http://192.168.1.50/water-data",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# One-time migration: replace the old illustrative 18.5 L starter reading
# with a more suitable 2.9 L demo baseline. Do not overwrite later readings.
if (
    st.session_state.get("water_today") == 18.5
    and not st.session_state.get("water_today_demo_baseline_migrated", False)
):
    st.session_state.water_today = 2.9
    st.session_state.water_today_demo_baseline_migrated = True

if st.session_state.recycled_water_log is None:
    st.session_state.recycled_water_log = pd.DataFrame(columns=["Date", "Time", "Treated wastewater recycled (L)"])

if st.session_state.wastewater_daily_log is None:
    st.session_state.wastewater_daily_log = pd.DataFrame(
        columns=["Date", "Time", "Wastewater collected (L)",
                 "Treated water produced (L)", "Treatment loss (L)"]
    )

if st.session_state.history is None:
    rng = np.random.default_rng(21)
    times = pd.date_range(end=datetime.now(), periods=36, freq="20min")
    base = np.linspace(49, st.session_state.soil_moisture, 36)
    moisture = np.clip(base + rng.normal(0, 2.0, 36), 10, 85)
    st.session_state.history = pd.DataFrame({
        "Time": times,
        "Soil moisture (%)": moisture,
        "Air temperature (°C)": 31 + 3*np.sin(np.linspace(0, 3, 36)) + rng.normal(0, .5, 36),
        "Humidity (%)": 48 + 7*np.cos(np.linspace(0, 3, 36)) + rng.normal(0, 1, 36),
        "Water flow (L/min)": np.maximum(0, rng.normal(.15, .08, 36))
    })

# ---------- Helpers ----------
def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def make_ai_models():
    """Illustrative starter models trained on synthetic data.
    Replace with real, labelled crop/soil/sensor observations for deployment.
    """
    rng = np.random.default_rng(42)
    n = 1800
    moisture = rng.uniform(10, 85, n)
    temp = rng.uniform(15, 45, n)
    humidity = rng.uniform(15, 95, n)
    rain = rng.uniform(0, 100, n)
    plant_factor = rng.uniform(.75, 1.25, n)
    soil_factor = rng.uniform(.8, 1.2, n)
    # Plant water-demand factor is an input so different plant profiles
    # can influence the illustrative prediction.
    # Illustrative target, not a universal agronomic formula.
    deficit = np.maximum(0, 48 - moisture)
    water_ml = np.maximum(
        0,
        (deficit * 7 + np.maximum(0, temp - 25) * 3
         + np.maximum(0, 55 - humidity) * .6 - rain * .9)
        * plant_factor * soil_factor
    )
    water_ml = np.clip(water_ml, 0, 650)
    need = (water_ml > 70).astype(int)
    X = pd.DataFrame({
        "soil_moisture": moisture,
        "air_temperature": temp,
        "humidity": humidity,
        "rain_probability": rain,
        "plant_factor": plant_factor,
    })
    X_train, X_test, y_train, y_test = train_test_split(
        X, water_ml, test_size=.2, random_state=42
    )
    reg = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3)
    reg.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, reg.predict(X_test))
    clf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=3)
    clf.fit(X, need)
    return reg, clf, mae

@st.cache_resource
def get_models():
    return make_ai_models()

def fetch_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m",
        "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min",
        "forecast_days": 3,
        "timezone": "auto"
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    return response.json()

def weather_frame(weather):
    hourly = weather.get("hourly", {})
    return pd.DataFrame({
        "Time": pd.to_datetime(hourly.get("time", [])),
        "Temperature (°C)": hourly.get("temperature_2m", []),
        "Humidity (%)": hourly.get("relative_humidity_2m", []),
        "Rain probability (%)": hourly.get("precipitation_probability", []),
        "Rainfall (mm)": hourly.get("precipitation", []),
        "Wind (km/h)": hourly.get("wind_speed_10m", [])
    })

def get_prediction(moisture, temp, humidity, rain_probability, plant_factor):
    reg, clf, mae = get_models()
    features = pd.DataFrame([{
        "soil_moisture": moisture,
        "air_temperature": temp,
        "humidity": humidity,
        "rain_probability": rain_probability,
        "plant_factor": plant_factor,
    }])
    water = max(0, float(reg.predict(features)[0]))
    probability = float(clf.predict_proba(features)[0][1]) * 100
    return water, probability, mae, reg

def status_for_plant(moisture, profile):
    low = profile["moisture_min"]
    high = profile["moisture_max"]
    if moisture < low - 10:
        return "Water-stressed", "bad"
    if moisture < low:
        return "Below preferred range", "warn"
    if moisture <= high:
        return "Within preferred range", "good"
    if moisture <= high + 10:
        return "Above preferred range", "warn"
    return "Very wet – check drainage", "warn"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌱 AquaGrow")
    st.caption("AI Smart Irrigation Platform")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        [
            "1. Dashboard",
            "2. Live Sensor Monitoring",
            "3. Weather Intelligence",
            "4. AI Prediction Centre",
            "5. Smart Irrigation Control",
            "6. Water Conservation Report",
            "7. Wastewater Recycling"
        ],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("### 🌿 Garden profile")
    previous_plant = st.session_state.plant
    selected_plant = st.selectbox(
        "Plant type", list(PLANT_PROFILES.keys()),
        index=list(PLANT_PROFILES.keys()).index(previous_plant)
    )
    if selected_plant != previous_plant:
        # Save the readings for the plant being left, then load the selected
        # plant's own readings. Each plant keeps its values independently.
        st.session_state.plant_sensor_profiles[previous_plant] = {
            "soil_moisture": st.session_state.soil_moisture,
            "air_temp": st.session_state.air_temp,
            "humidity": st.session_state.humidity,
            "soil_temp": st.session_state.soil_temp,
            "rain_sensor": st.session_state.rain_sensor,
        }
        st.session_state.plant = selected_plant
        for key, value in st.session_state.plant_sensor_profiles[selected_plant].items():
            st.session_state[key] = value
    profile = PLANT_PROFILES[st.session_state.plant]
    st.caption(f"Preferred demo soil-moisture range: {profile['moisture_min']}–{profile['moisture_max']}%")
    st.caption(profile["description"])
    st.session_state.soil_type = st.selectbox(
        "Soil type", ["Loamy", "Sandy", "Clay", "Silt"],
        index=["Loamy", "Sandy", "Clay", "Silt"].index(st.session_state.soil_type)
    )
    st.session_state.growth_stage = st.selectbox(
        "Growth stage", ["Seedling", "Vegetative", "Flowering", "Fruiting", "Mature"],
        index=["Seedling", "Vegetative", "Flowering", "Fruiting", "Mature"].index(st.session_state.growth_stage)
    )
    st.session_state.area_m2 = st.number_input("Garden area (m²)", min_value=.1, max_value=500.0,
                                                value=float(st.session_state.area_m2), step=.5)
    st.markdown("---")
    st.caption("Demo mode: sensor values are simulated. Weather is fetched live when internet access is available.")

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🌱 AquaGrow</h1>
    <p>AI-powered predictive irrigation • Grow smarter. Save every drop.</p>
</div>
""", unsafe_allow_html=True)

now = datetime.now()
moisture = float(st.session_state.soil_moisture)
air_temp = float(st.session_state.air_temp)
humidity = float(st.session_state.humidity)
soil_temp = float(st.session_state.soil_temp)
rain_sensor = int(st.session_state.rain_sensor)
profile = PLANT_PROFILES[st.session_state.plant]
st.session_state.plant_sensor_profiles[st.session_state.plant] = {
    "soil_moisture": moisture,
    "air_temp": air_temp,
    "humidity": humidity,
    "soil_temp": soil_temp,
    "rain_sensor": rain_sensor,
}

# Fetch forecast once per session/page rerun, with graceful fallback
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
    st.session_state.weather_error = None
try:
    if st.session_state.weather_data is None:
        st.session_state.weather_data = fetch_weather(
            st.session_state.latitude, st.session_state.longitude
        )
        st.session_state.weather_error = None
except Exception as e:
    st.session_state.weather_error = str(e)

weather = st.session_state.weather_data
forecast_df = weather_frame(weather) if weather else pd.DataFrame()
current_rain_prob = 0
current_weather_text = "Weather unavailable"
if weather:
    current = weather.get("current", {})
    current_weather_text = (
        f"{current.get('temperature_2m', air_temp)}°C • "
        f"{current.get('relative_humidity_2m', humidity)}% humidity • "
        f"{current.get('wind_speed_10m', 0)} km/h wind"
    )
    if not forecast_df.empty:
        future = forecast_df[forecast_df["Time"] >= pd.Timestamp.now().floor("h")]
        if not future.empty:
            current_rain_prob = float(future.iloc[:6]["Rain probability (%)"].max())

predicted_water, irrigation_probability, model_mae, model = get_prediction(
    moisture, air_temp, humidity, current_rain_prob, profile["plant_factor"]
)
plant_status, plant_status_class = status_for_plant(moisture, profile)
time_to_irrigation = "Now" if irrigation_probability >= 65 and current_rain_prob < 60 and moisture < profile["moisture_min"] else (
    "Reassess in 2–3 hours" if moisture < profile["moisture_min"] else "Reassess in 6 hours"
)
if current_rain_prob >= 60:
    ai_reason = "Rain is likely soon, so postpone irrigation and recheck soil moisture."
elif moisture < profile["moisture_min"]:
    ai_reason = f"Soil moisture is below the preferred range for {st.session_state.plant}. Temperature and humidity also influence estimated water demand."
elif moisture <= profile["moisture_max"]:
    ai_reason = f"Soil moisture is within the preferred demonstration range for {st.session_state.plant}. Continue monitoring before watering."
else:
    ai_reason = "Current soil moisture appears adequate. Continue monitoring before watering."

# ---------- Screen 1: Dashboard ----------
if page == "1. Dashboard":
    st.subheader("Garden overview")
    st.caption(f"{st.session_state.location_name}  •  {now.strftime('%A, %d %B %Y | %I:%M %p')}")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Soil moisture", f"{moisture:.0f}%", "Latest sensor reading")
    with c2: metric_card("Air temperature", f"{air_temp:.1f}°C", f"Humidity: {humidity:.0f}%")
    with c3: metric_card("Water consumed today", f"{st.session_state.water_today:.1f} L", "Illustrative plant-use target: 2.7–3.2 L/day")
    with c4: metric_card("Plant health status", plant_status, st.session_state.plant)

    st.markdown("### Garden status")
    st.info(f"Selected plant: **{st.session_state.plant}** • Preferred demo soil-moisture range: **{profile['moisture_min']}–{profile['moisture_max']}%** • {profile['description']}")
    left, right = st.columns([1.35, 1])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=moisture,
            title={"text": "Soil moisture (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#26834a"},
                   "steps": [{"range": [0, profile["moisture_min"]], "color": "#f8d7da"},
                             {"range": [profile["moisture_min"], profile["moisture_max"]], "color": "#d9f0df"},
                             {"range": [profile["moisture_max"], 100], "color": "#dce8f5"}]}
        ))
        fig.update_layout(height=280, margin=dict(l=20,r=20,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Current weather")
        st.markdown(f"**{current_weather_text}**")
        if weather:
            daily = weather.get("daily", {})
            if daily.get("time"):
                st.write(f"Rain chance today: **{daily.get('precipitation_probability_max',[0])[0]}%**")
                st.write(f"Expected rainfall: **{daily.get('precipitation_sum',[0])[0]} mm**")
        else:
            st.info("Live weather could not be loaded. Check your internet connection.")
        st.markdown("#### Next irrigation")
        st.markdown(f"**{time_to_irrigation}**")
        st.caption(ai_reason)

    st.markdown("### Recent soil moisture trend")
    hist = st.session_state.history
    fig = px.line(hist, x="Time", y="Soil moisture (%)", markers=True,
                  title="Soil moisture readings")
    fig.update_traces(line_color="#23834a")
    fig.update_layout(height=300, yaxis_range=[0,100])
    st.plotly_chart(fig, use_container_width=True)

# ---------- Screen 2: Live Sensor Monitoring ----------
elif page == "2. Live Sensor Monitoring":
    st.subheader("Live sensor monitoring")
    st.info(f"This starter version uses simulated readings for **{st.session_state.plant}**. Its illustrative preferred soil-moisture range is {profile['moisture_min']}–{profile['moisture_max']}%. Use the controls below to test the interface. Connect ESP32 telemetry for real sensor values.")
    with st.expander("🧪 Demo sensor simulator", expanded=True):
        a, b, c = st.columns(3)
        with a:
            st.session_state.soil_moisture = st.slider("Soil moisture (%)", 0, 100, int(moisture))
            st.session_state.air_temp = st.slider("Air temperature (°C)", 10.0, 50.0, float(air_temp), .5)
        with b:
            st.session_state.humidity = st.slider("Relative humidity (%)", 0, 100, int(humidity))
            st.session_state.soil_temp = st.slider("Soil temperature (°C)", 5.0, 45.0, float(soil_temp), .5)
        with c:
            st.session_state.rain_sensor = st.selectbox("Rain sensor", [0, 1], index=rain_sensor,
                                                        format_func=lambda x: "Dry (0)" if x == 0 else "Rain detected (1)")
            if st.button("Add reading to history", use_container_width=True):
                new_row = pd.DataFrame([{
                    "Time": datetime.now(),
                    "Soil moisture (%)": st.session_state.soil_moisture,
                    "Air temperature (°C)": st.session_state.air_temp,
                    "Humidity (%)": st.session_state.humidity,
                    "Water flow (L/min)": 0.0 if not st.session_state.pump_on else .8
                }])
                st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(150)
                st.success("New simulated reading added.")

    st.markdown("### Current sensor readings")
    cols = st.columns(4)
    readings = [
        ("Soil moisture", f"{st.session_state.soil_moisture}%", "💧"),
        ("Air temperature", f"{st.session_state.air_temp:.1f}°C", "🌡️"),
        ("Relative humidity", f"{st.session_state.humidity}%", "☁️"),
        ("Soil temperature", f"{st.session_state.soil_temp:.1f}°C", "🌱")
    ]
    for col, (label, value, emoji) in zip(cols, readings):
        with col: metric_card(f"{emoji} {label}", value, "Simulated sensor")
    st.markdown("### Sensor gauges")
    g1, g2, g3 = st.columns(3)
    for col, label, val, maxval, color in [
        (g1, "Soil moisture", st.session_state.soil_moisture, 100, "#24834a"),
        (g2, "Air humidity", st.session_state.humidity, 100, "#2583bd"),
        (g3, "Temperature", st.session_state.air_temp, 50, "#d58a28")
    ]:
        with col:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=val,
                title={"text": label}, gauge={"axis":{"range":[0,maxval]},
                "bar":{"color":color}}))
            fig.update_layout(height=220, margin=dict(l=15,r=15,t=45,b=5))
            st.plotly_chart(fig, use_container_width=True)
    hist = st.session_state.history
    st.markdown("### Historical sensor graphs")
    tab1, tab2, tab3 = st.tabs(["Soil moisture", "Temperature & humidity", "Water flow"])
    with tab1:
        st.plotly_chart(px.line(hist, x="Time", y="Soil moisture (%)", markers=True), use_container_width=True)
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Time"], y=hist["Air temperature (°C)"], name="Temperature (°C)"))
        fig.add_trace(go.Scatter(x=hist["Time"], y=hist["Humidity (%)"], name="Humidity (%)", yaxis="y2"))
        fig.update_layout(yaxis=dict(title="Temperature (°C)"),
                          yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.plotly_chart(px.area(hist, x="Time", y="Water flow (L/min)"), use_container_width=True)
    st.metric("Rain sensor status", "Rain detected" if st.session_state.rain_sensor else "No rain detected")

# ---------- Screen 3: Weather Intelligence ----------
elif page == "3. Weather Intelligence":
    st.subheader("Weather intelligence")
    st.write("AquaGrow uses the Open-Meteo forecast service to consider upcoming rain before recommending irrigation.")
    with st.expander("📍 Location settings", expanded=False):
        loc_col1, loc_col2 = st.columns(2)
        with loc_col1:
            new_lat = st.number_input("Latitude", value=float(st.session_state.latitude), format="%.4f")
        with loc_col2:
            new_lon = st.number_input("Longitude", value=float(st.session_state.longitude), format="%.4f")
        new_name = st.text_input("Location name", value=st.session_state.location_name)
        if st.button("Update location and reload weather"):
            st.session_state.latitude = new_lat
            st.session_state.longitude = new_lon
            st.session_state.location_name = new_name
            st.session_state.weather_data = None
            st.rerun()
    if weather:
        current = weather.get("current", {})
        d1, d2, d3, d4 = st.columns(4)
        with d1: metric_card("Temperature", f"{current.get('temperature_2m','—')}°C", "Current")
        with d2: metric_card("Humidity", f"{current.get('relative_humidity_2m','—')}%", "Current")
        with d3: metric_card("Wind speed", f"{current.get('wind_speed_10m','—')} km/h", "Current")
        with d4: metric_card("Rainfall now", f"{current.get('precipitation','—')} mm", "Current")
        if not forecast_df.empty:
            st.markdown("### Next 24 hours")
            next24 = forecast_df.head(24)
            fig = px.bar(next24, x="Time", y="Rain probability (%)",
                         title="Hourly probability of precipitation")
            fig.update_traces(marker_color="#3b9b76")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=next24["Time"], y=next24["Temperature (°C)"],
                                      name="Temperature (°C)", mode="lines+markers"))
            fig2.add_trace(go.Scatter(x=next24["Time"], y=next24["Humidity (%)"],
                                      name="Humidity (%)", mode="lines", yaxis="y2"))
            fig2.update_layout(yaxis=dict(title="Temperature (°C)"),
                               yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
                               title="Temperature and humidity forecast")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("### Three-day forecast")
            daily = weather.get("daily", {})
            daily_df = pd.DataFrame({
                "Date": daily.get("time", []),
                "Min temp (°C)": daily.get("temperature_2m_min", []),
                "Max temp (°C)": daily.get("temperature_2m_max", []),
                "Rain probability (%)": daily.get("precipitation_probability_max", []),
                "Rainfall (mm)": daily.get("precipitation_sum", [])
            })
            st.dataframe(daily_df, use_container_width=True, hide_index=True)
        if current_rain_prob >= 60:
            st.markdown('<div class="warn">🌧️ Rain is likely in the near-term forecast. AquaGrow recommends postponing irrigation and checking soil moisture again.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="good">☀️ No high-probability rainfall window detected in the next few hours. Use soil readings and the AI recommendation to decide whether to irrigate.</div>', unsafe_allow_html=True)
    else:
        st.error("Weather service is currently unavailable. Check the internet connection and try again.")
        st.caption(f"Technical detail: {st.session_state.weather_error}")
        st.info("The rest of AquaGrow can still be explored using simulated weather-independent readings.")

# ---------- Screen 4: AI Prediction Centre ----------
elif page == "4. AI Prediction Centre":
    st.subheader("AI prediction centre")
    st.write("A Random Forest starter model estimates irrigation demand from soil moisture, temperature, humidity and forecast rainfall.")
    if not weather:
        st.warning("Live weather is unavailable. The prediction will use a rainfall probability of 0% for this demonstration.")
    st.markdown(f"### Current model inputs — {st.session_state.plant}")
    st.caption(f"Plant water-demand factor: {profile['plant_factor']:.2f} • Preferred demo soil-moisture range: {profile['moisture_min']}–{profile['moisture_max']}%")
    a,b,c,d = st.columns(4)
    with a: metric_card("Soil moisture", f"{moisture:.0f}%", "Sensor input")
    with b: metric_card("Temperature", f"{air_temp:.1f}°C", "Sensor input")
    with c: metric_card("Humidity", f"{humidity:.0f}%", "Sensor input")
    with d: metric_card("Rain probability", f"{current_rain_prob:.0f}%", "Next 6 hours")
    st.markdown("### AI prediction")
    p1,p2,p3 = st.columns(3)
    with p1: metric_card("Predicted water requirement", f"{predicted_water:.0f} ml", "Illustrative model estimate")
    with p2: metric_card("Probability irrigation is needed", f"{irrigation_probability:.0f}%", "Model classification")
    with p3: metric_card("Expected irrigation time", time_to_irrigation, "Decision layer")
    if current_rain_prob >= 60:
        st.markdown('<div class="warn">🌧️ Rainfall-aware adjustment: postpone irrigation because forecast rainfall is likely. Recheck soil moisture after the forecast rain window.</div>', unsafe_allow_html=True)
    elif irrigation_probability >= 65:
        st.markdown('<div class="good">💧 Irrigation is recommended. Check the plant profile and confirm the proposed quantity before watering.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="good">🌱 Irrigation is not currently required according to this demonstration model. Continue monitoring.</div>', unsafe_allow_html=True)
    st.markdown("### Why did the model make this recommendation?")
    st.write(ai_reason)
    st.caption(f"Training/test MAE on synthetic demo data: {model_mae:.1f} ml. This is not a real-world accuracy claim.")
    importance = pd.DataFrame({
        "Input": ["Soil moisture", "Temperature", "Humidity", "Rain probability", "Plant water-demand factor"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig = px.bar(importance, x="Importance", y="Input", orientation="h",
                 title="Model feature importance (synthetic training data)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("How to make this a genuine AI model"):
        st.markdown("""
        1. Collect sensor readings at regular intervals over several weeks.
        2. Record plant type, growth stage, soil type and garden area.
        3. Record the water quantity actually applied and the soil moisture response.
        4. Define a target such as measured irrigation volume needed to reach a validated soil-moisture range.
        5. Train and test on separate dates or plots, not just randomly mixed readings.
        6. Compare predictions with a horticultural reference or controlled experiments.
        7. Retrain as more reliable data becomes available.
        """)

# ---------- Screen 5: Smart Irrigation Control ----------
elif page == "5. Smart Irrigation Control":
    st.subheader("Smart irrigation control")
    st.warning("Prototype safety: pump commands on this page are simulated. No physical pump or relay is connected.")
    left, right = st.columns([1,1])
    with left:
        st.markdown("### Irrigation settings")
        st.session_state.mode = st.radio("Operating mode", ["Automatic", "Manual"],
                                         index=0 if st.session_state.mode == "Automatic" else 1,
                                         horizontal=True)
        duration = st.slider("Irrigation duration (seconds)", 5, 180, 30, 5)
        flow_rate = st.number_input("Estimated pump flow rate (L/min)", min_value=.1, max_value=20.0,
                                    value=0.5, step=.1)
        estimated_volume = flow_rate * duration / 60
        st.metric("Estimated water per cycle", f"{estimated_volume:.2f} L")
        st.session_state.emergency_stop = st.checkbox("🚨 Emergency stop enabled",
                                                       value=st.session_state.emergency_stop)
    with right:
        st.markdown("### Pump status")
        st.metric("Pump", "ON" if st.session_state.pump_on else "OFF")
        st.metric("Today's irrigation cycles", st.session_state.irrigation_cycles)
        st.metric("Water consumed today", f"{st.session_state.water_today:.1f} L", "Demo target: 2.7–3.2 L/day")
        if st.session_state.emergency_stop:
            st.markdown('<div class="bad">Emergency stop is active. Irrigation commands are disabled.</div>', unsafe_allow_html=True)
        elif st.session_state.mode == "Automatic":
            if irrigation_probability >= 65 and current_rain_prob < 60 and moisture < profile["moisture_min"]:
                st.markdown('<div class="good">Automatic recommendation: irrigation conditions are met. Review before enabling real hardware.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn">Automatic mode is monitoring. No irrigation cycle is currently triggered.</div>', unsafe_allow_html=True)
        else:
            st.info("Manual mode: use the controls below to simulate a watering cycle.")
    b1,b2,b3 = st.columns(3)
    with b1:
        if st.button("▶ Start pump (demo)", type="primary", use_container_width=True,
                     disabled=st.session_state.emergency_stop or st.session_state.pump_on):
            if st.session_state.mode == "Manual" or (
                irrigation_probability >= 65 and current_rain_prob < 60 and moisture < profile["moisture_min"]
            ):
                st.session_state.pump_on = True
                st.rerun()
            else:
                st.warning("Automatic safety conditions are not met.")
    with b2:
        if st.button("■ Stop pump", use_container_width=True, disabled=not st.session_state.pump_on):
            st.session_state.pump_on = False
            st.session_state.irrigation_cycles += 1
            st.session_state.water_today += estimated_volume
            st.session_state.last_irrigation = datetime.now()
            st.rerun()
    with b3:
        if st.button("Reset emergency stop", use_container_width=True):
            st.session_state.emergency_stop = False
            st.rerun()
    st.caption("For a real installation, add hardware interlocks, maximum run-time protection, leak detection, manual isolation and an appropriate driver/power supply. Never switch a pump directly from an ESP32 GPIO pin.")

# ---------- Screen 6: Water Conservation Report ----------
elif page == "6. Water Conservation Report":
    st.subheader("Water conservation report")
    st.write(f"Track water consumption for **{st.session_state.plant}** and compare it with a defined baseline irrigation schedule.")
    st.caption(f"Plant-specific preferred demo soil-moisture range: {profile['moisture_min']}–{profile['moisture_max']}%.")
    baseline = st.number_input("Baseline water use per day (litres)", min_value=0.0, value=30.0, step=1.0)
    actual = float(st.session_state.water_today)
    saved = max(0.0, baseline - actual)
    percent_saved = (saved / baseline * 100) if baseline > 0 else 0
    a,b,c,d = st.columns(4)
    with a: metric_card("Today's water use", f"{actual:.1f} L", "Demo counter")
    with b: metric_card("Baseline use", f"{baseline:.1f} L", "User-defined comparison")
    with c: metric_card("Estimated water saved", f"{saved:.1f} L", "Baseline minus actual")
    with d: metric_card("Estimated saving", f"{percent_saved:.1f}%", "Compared with baseline")
    st.markdown("### Daily water use comparison")
    report_df = pd.DataFrame({
        "Category": ["Baseline schedule", "AquaGrow demo"],
        "Water use (L)": [baseline, actual]
    })
    fig = px.bar(report_df, x="Category", y="Water use (L)", color="Category",
                 title="Water use: baseline vs AquaGrow")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Historical soil moisture")
    hist = st.session_state.history
    st.plotly_chart(px.line(hist, x="Time", y="Soil moisture (%)", title="Soil moisture over time"),
                    use_container_width=True)
    st.markdown("### Irrigation summary")
    summary = pd.DataFrame([{
        "Date": now.strftime("%Y-%m-%d"),
        "Plant": st.session_state.plant,
        "Garden area (m²)": st.session_state.area_m2,
        "Irrigation cycles": st.session_state.irrigation_cycles,
        "Water used (L)": actual,
        "Baseline (L)": baseline,
        "Estimated saved (L)": saved
    }])
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download report as CSV", summary.to_csv(index=False),
                       file_name="aquagrow_water_report.csv", mime="text/csv")
    st.caption("Water savings are only meaningful when the baseline is measured under comparable plant, weather, area and growing conditions.")

# ---------- Screen 7: Wastewater Recycling ----------
elif page == "7. Wastewater Recycling":
    st.subheader("Automatic wastewater measurement & recycling")
    st.write(
        "Measure wastewater entering the system and treated water leaving the "
        "treatment unit. AquaGrow calculates the recycled volume and estimated "
        "treatment loss."
    )

    st.info(
        "For automatic hardware monitoring, connect two flow meters to the ESP32: "
        "one before treatment and one after treatment. The ESP32 must provide "
        "cumulative litre readings through its local HTTP endpoint."
    )

    st.markdown("### 1. Select monitoring mode")
    mode = st.radio(
        "Water measurement source",
        ["Demo simulator", "ESP32 live meter"],
        horizontal=True,
        index=0 if st.session_state.water_meter_mode == "Demo simulator" else 1,
        key="water_meter_mode_choice"
    )
    st.session_state.water_meter_mode = mode

    if mode == "Demo simulator":
        st.caption(
            "Demo mode: enter new measured volumes to simulate flow-meter readings. "
            "These are illustrative values, not live measurements."
        )
        sim_a, sim_b = st.columns(2)
        with sim_a:
            raw_increment = st.number_input(
                "New wastewater collected (L)",
                min_value=0.0, max_value=10000.0, value=12.0, step=0.5,
                key="demo_raw_increment"
            )
        with sim_b:
            treated_increment = st.number_input(
                "New treated water produced (L)",
                min_value=0.0, max_value=10000.0, value=9.0, step=0.5,
                key="demo_treated_increment"
            )

        if st.button("＋ Add demo meter reading", type="primary"):
            if treated_increment > raw_increment:
                st.warning(
                    "Treated water cannot exceed the wastewater collected in this "
                    "demo reading. Check the two values."
                )
            elif raw_increment > 0 or treated_increment > 0:
                loss = max(0.0, raw_increment - treated_increment)
                now_reading = datetime.now()
                st.session_state.wastewater_in_total_l += float(raw_increment)
                st.session_state.treated_out_total_l += float(treated_increment)
                st.session_state.recycled_tank_level = min(
                    st.session_state.recycled_tank_capacity,
                    st.session_state.recycled_tank_level + float(treated_increment)
                )
                daily_row = pd.DataFrame([{
                    "Date": now_reading.strftime("%Y-%m-%d"),
                    "Time": now_reading.strftime("%H:%M:%S"),
                    "Wastewater collected (L)": float(raw_increment),
                    "Treated water produced (L)": float(treated_increment),
                    "Treatment loss (L)": loss
                }])
                st.session_state.wastewater_daily_log = pd.concat(
                    [st.session_state.wastewater_daily_log, daily_row],
                    ignore_index=True
                )
                recycled_row = pd.DataFrame([{
                    "Date": now_reading.strftime("%Y-%m-%d"),
                    "Time": now_reading.strftime("%H:%M:%S"),
                    "Treated wastewater recycled (L)": float(treated_increment)
                }])
                st.session_state.recycled_water_log = pd.concat(
                    [st.session_state.recycled_water_log, recycled_row],
                    ignore_index=True
                )
                st.success("Demo meter reading added to the water records.")
                st.rerun()

    else:
        st.caption(
            "Live mode: enter the ESP32 endpoint URL. The endpoint should return "
            "cumulative litres from both flow meters."
        )
        endpoint = st.text_input(
            "ESP32 water-meter endpoint",
            value=st.session_state.esp32_water_endpoint,
            help="Example: http://192.168.1.50/water-data"
        )
        st.session_state.esp32_water_endpoint = endpoint

        st.markdown("**Expected ESP32 JSON response**")
        st.code(
            '{"wastewater_in_total_l": 125.5, '
            '"treated_out_total_l": 98.2, "tank_level_l": 42.0}',
            language="json"
        )
        st.caption(
            "The two total fields must be cumulative meter readings in litres "
            "since the meter was reset, not the flow rate in L/min."
        )

        if st.button("↻ Read live water meters", type="primary"):
            try:
                response = requests.get(endpoint, timeout=5)
                response.raise_for_status()
                meter_data = response.json()

                required = ["wastewater_in_total_l", "treated_out_total_l"]
                missing = [key for key in required if key not in meter_data]
                if missing:
                    st.error("ESP32 response is missing: " + ", ".join(missing))
                else:
                    new_raw = float(meter_data["wastewater_in_total_l"])
                    new_treated = float(meter_data["treated_out_total_l"])

                    if new_raw < 0 or new_treated < 0:
                        st.error("Meter totals must not be negative.")
                    else:
                        previous_raw = st.session_state.wastewater_meter_previous_l
                        previous_treated = st.session_state.treated_meter_previous_l

                        # First live reading establishes the baseline. Subsequent
                        # readings add only the new volume, preventing double-counting.
                        raw_delta = 0.0 if previous_raw is None else max(0.0, new_raw - previous_raw)
                        treated_delta = 0.0 if previous_treated is None else max(0.0, new_treated - previous_treated)

                        # If a meter has reset, establish a new baseline and notify
                        # the user rather than adding a negative or misleading delta.
                        reset_detected = (
                            previous_raw is not None and new_raw < previous_raw
                        ) or (
                            previous_treated is not None and new_treated < previous_treated
                        )

                        st.session_state.wastewater_meter_previous_l = new_raw
                        st.session_state.treated_meter_previous_l = new_treated

                        if reset_detected:
                            st.warning(
                                "A meter total appears to have reset. A new baseline "
                                "has been saved; the reset interval was not added."
                            )
                        elif raw_delta > 0 or treated_delta > 0:
                            if treated_delta > raw_delta:
                                st.warning(
                                    "Treated-water increment is greater than the "
                                    "wastewater increment. Check meter direction, "
                                    "meter reset history and plumbing."
                                )
                            else:
                                loss_delta = max(0.0, raw_delta - treated_delta)
                                reading_time = datetime.now()
                                st.session_state.wastewater_in_total_l += raw_delta
                                st.session_state.treated_out_total_l += treated_delta

                                if "tank_level_l" in meter_data:
                                    st.session_state.recycled_tank_level = max(
                                        0.0, float(meter_data["tank_level_l"])
                                    )
                                else:
                                    st.session_state.recycled_tank_level = min(
                                        st.session_state.recycled_tank_capacity,
                                        st.session_state.recycled_tank_level + treated_delta
                                    )

                                new_log_row = pd.DataFrame([{
                                    "Date": reading_time.strftime("%Y-%m-%d"),
                                    "Time": reading_time.strftime("%H:%M:%S"),
                                    "Wastewater collected (L)": raw_delta,
                                    "Treated water produced (L)": treated_delta,
                                    "Treatment loss (L)": loss_delta
                                }])
                                st.session_state.wastewater_daily_log = pd.concat(
                                    [st.session_state.wastewater_daily_log, new_log_row],
                                    ignore_index=True
                                )
                                recycled_row = pd.DataFrame([{
                                    "Date": reading_time.strftime("%Y-%m-%d"),
                                    "Time": reading_time.strftime("%H:%M:%S"),
                                    "Treated wastewater recycled (L)": treated_delta
                                }])
                                st.session_state.recycled_water_log = pd.concat(
                                    [st.session_state.recycled_water_log, recycled_row],
                                    ignore_index=True
                                )

                        if "tank_level_l" in meter_data:
                            st.session_state.recycled_tank_level = max(
                                0.0, float(meter_data["tank_level_l"])
                            )
                        st.success("ESP32 meter data received.")
                        st.rerun()

            except requests.RequestException as exc:
                st.error(f"Could not connect to the ESP32: {exc}")
            except (ValueError, TypeError) as exc:
                st.error(f"Invalid meter data received: {exc}")

    # Totals are accumulated from measured increments (or demo entries).
    raw_total = float(st.session_state.wastewater_in_total_l)
    treated_total = float(st.session_state.treated_out_total_l)
    treatment_loss_total = max(0.0, raw_total - treated_total)
    recovery_rate = (treated_total / raw_total * 100) if raw_total > 0 else 0.0

    st.markdown("### 2. Automatic water measurement")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Wastewater collected", f"{raw_total:.1f} L", "Cumulative inlet meter")
    with m2:
        metric_card("Treated water produced", f"{treated_total:.1f} L", "Cumulative outlet meter")
    with m3:
        metric_card("Estimated treatment loss", f"{treatment_loss_total:.1f} L", "Collected minus treated")
    with m4:
        metric_card("Recovery efficiency", f"{recovery_rate:.1f}%", "Treated ÷ collected × 100")

    st.markdown("### 3. Water recycling process")
    if raw_total <= 0:
        st.info("No water has been measured yet. Add a demo reading or connect the ESP32 meters.")
    else:
        st.progress(min(1.0, recovery_rate / 100))
        st.caption(
            f"Out of every 100 L collected, approximately {recovery_rate:.1f} L "
            f"has been measured as treated water. The remaining "
            f"{100 - recovery_rate:.1f} L represents estimated treatment loss."
        )

    if treated_total > raw_total:
        st.warning(
            "The cumulative treated-water total is greater than the collected-water "
            "total. Check that both meters cover the same measurement period and "
            "that their counters have not been reset independently."
        )

    st.markdown("### 4. Recycled-water storage tank")
    tank_a, tank_b = st.columns(2)
    with tank_a:
        tank_capacity = st.number_input(
            "Tank capacity (litres)", min_value=1.0, max_value=100000.0,
            value=float(st.session_state.recycled_tank_capacity), step=5.0,
            key="recycled_tank_capacity_input"
        )
        st.session_state.recycled_tank_capacity = tank_capacity
    with tank_b:
        tank_level = st.number_input(
            "Current stored treated water (litres)", min_value=0.0,
            max_value=float(tank_capacity),
            value=min(float(st.session_state.recycled_tank_level), float(tank_capacity)),
            step=1.0, key="recycled_tank_level_input"
        )
        st.session_state.recycled_tank_level = tank_level

    fill_percent = (tank_level / tank_capacity * 100) if tank_capacity else 0
    st.progress(min(1.0, fill_percent / 100))
    st.caption(
        f"Stored treated water: {tank_level:.1f} L of {tank_capacity:.1f} L "
        f"({fill_percent:.1f}% full). In live mode, an optional tank-level "
        f"sensor can update this value automatically."
    )

    st.markdown("### 5. Water measurement history")
    water_log = st.session_state.wastewater_daily_log.copy()
    if not water_log.empty:
        daily = water_log.copy()
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily_totals = daily.groupby("Date", as_index=False)[
            ["Wastewater collected (L)", "Treated water produced (L)", "Treatment loss (L)"]
        ].sum()
        fig = px.bar(
            daily_totals, x="Date",
            y=["Wastewater collected (L)", "Treated water produced (L)", "Treatment loss (L)"],
            barmode="group", title="Wastewater collected vs treated water produced"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            water_log.sort_values(["Date", "Time"], ascending=False),
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "⬇ Download water measurement report (CSV)",
            water_log.to_csv(index=False),
            file_name="aquagrow_automatic_water_measurements.csv",
            mime="text/csv"
        )
    else:
        st.info("Water measurement history will appear here after the first reading.")

    st.markdown("### 6. Sensor installation guide")
    st.markdown("""
    - **Flow meter 1 (wastewater inlet):** Install in the pipe carrying collected wastewater
      into the collection/treatment system.
    - **Flow meter 2 (treated-water outlet):** Install after filtration/treatment, before
      the treated-water storage tank.
    - **Optional tank sensor:** Use a suitable ultrasonic or level sensor to measure the
      actual storage-tank level.
    - **ESP32:** Count flow-meter pulses, convert pulses to litres using the calibration
      factor for the exact flow-meter model, and serve cumulative totals as JSON.
    - **Electrical safety:** Use suitable signal conditioning, waterproof enclosures and
      isolated pump drivers. Do not connect a pump directly to an ESP32 GPIO pin.
    """)

    st.markdown("**Important measurement note:**")
    st.caption(
        "The dashboard reports approximate volumes based on flow-meter calibration. "
        "For accurate results, calibrate each meter with a known volume (for example, "
        "a measured 10-litre container). Wastewater quality and treatment suitability "
        "must be checked separately; a flow meter cannot confirm that water is safe "
        "for irrigation. Records remain in the current app session unless persistent "
        "storage is added."
    )

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    '<div class="small-muted">🌱 AquaGrow • Student prototype • Demo sensor data unless connected to real hardware. '
    'AI model is trained on synthetic example data and must be retrained and validated before practical use.</div>',
    unsafe_allow_html=True
)
