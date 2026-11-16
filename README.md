# AquaGrow – AI Smart Irrigation Platform

## Included screens
1. Dashboard
2. Live Sensor Monitoring
3. Weather Intelligence (Open-Meteo)
4. AI Prediction Centre (Random Forest demonstration model)
5. Smart Irrigation Control (simulated controls only)
6. Water Conservation Report (CSV export)

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the platform:

   `streamlit run app.py`

5. Open the local URL shown in the terminal (usually http://localhost:8501).

## Important prototype notes
- Sensor readings and pump controls are simulated. This app does not control physical hardware.
- Weather forecast data is retrieved from the Open-Meteo API and requires internet access.
- The Random Forest model is trained on synthetic illustrative data. It is not an agronomic recommendation model and must be trained and validated using real, labelled observations before use with real plants.
- Do not connect a pump or valve until appropriate electrical isolation, driver circuitry, hardware interlocks and supervised testing are in place.
- The default location is Al Ain, UAE. Change coordinates in Weather Intelligence > Location settings.
