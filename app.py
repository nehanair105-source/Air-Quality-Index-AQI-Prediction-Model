import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI India Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf  = joblib.load("rf_model.pkl")
    sc  = joblib.load("scaler.pkl")
    enc = joblib.load("encoder.pkl")
    with open("aqi_encoding.json") as f:
        mapping = json.load(f)          # {"0": "Good", "1": "Moderate", ...}
    return rf, sc, enc, mapping

rf_model, scaler, encoder, aqi_mapping = load_artifacts()

# ─── AQI Visual Config ──────────────────────────────────────────────────────────
AQI_CONFIG = {
    "Good":                          {"emoji": "😊", "color": "#C6F135", "bg": "#1a2e0a"},
    "Moderate":                      {"emoji": "🙂", "color": "#F5E642", "bg": "#2e2a0a"},
    "Unhealthy for Sensitive Groups":{"emoji": "😐", "color": "#F5A623", "bg": "#2e1d0a"},
    "Unhealthy":                     {"emoji": "😷", "color": "#E05C2A", "bg": "#2e120a"},
    "Very Unhealthy":                {"emoji": "☠️",  "color": "#B03060", "bg": "#2e0a1a"},
}

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}

.stApp { background-color: #0f1117; }

/* Header */
.header-block {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
    border: 1px solid #1e2535;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 18px;
}
.header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #C6F135;
    margin: 0;
    line-height: 1.1;
}
.header-sub {
    font-size: 0.95rem;
    color: #8892a4;
    margin-top: 6px;
}

/* Cards */
.card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 20px;
}

/* Result card */
.result-card {
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    margin-top: 8px;
    border: 1px solid rgba(255,255,255,0.06);
}
.result-emoji { font-size: 3.5rem; line-height: 1; margin-bottom: 10px; }
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.result-caption { font-size: 0.85rem; color: #8892a4; }

/* Metric pills */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
.metric-pill {
    background: #1e2535;
    border-radius: 10px;
    padding: 10px 16px;
    flex: 1;
    min-width: 80px;
    text-align: center;
}
.metric-pill-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #C6F135;
}
.metric-pill-lbl { font-size: 0.7rem; color: #4a5568; margin-top: 2px; }

/* Confidence bar */
.conf-bar-bg {
    background: #1e2535;
    border-radius: 99px;
    height: 8px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #C6F135, #7BC24A);
    transition: width 0.6s ease;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: #8892a4 !important; font-size: 0.85rem !important; }
.stButton > button {
    width: 100%;
    background: #C6F135;
    color: #0f1117;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 14px;
    cursor: pointer;
    margin-top: 8px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }
div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <div>
        <div class="header-title">🌿 AQI India Predictor</div>
        <div class="header-sub">Random Forest · 85.3% Accuracy · India Air Quality Classification</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    # ── Pollutants ──────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Pollutant Levels</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        pm25   = st.number_input("PM2.5 (µg/m³)",  min_value=0.0, max_value=200.0, value=35.0, step=0.5)
        co     = st.number_input("CO (µg/m³)",      min_value=0.0, max_value=3000.0,value=400.0,step=10.0)
        so2    = st.number_input("SO₂ (µg/m³)",     min_value=0.0, max_value=150.0, value=12.0, step=0.5)
    with c2:
        no2    = st.number_input("NO₂ (µg/m³)",     min_value=0.0, max_value=130.0, value=16.0, step=0.5)
        o3     = st.number_input("O₃ (µg/m³)",      min_value=0.0, max_value=230.0, value=80.0, step=1.0)
        aod    = st.number_input("AOD",              min_value=0.0, max_value=2.0,   value=0.46, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Weather ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Weather Conditions</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        dew_point     = st.number_input("Dew Point (°C)",   min_value=-15.0, max_value=30.0, value=17.0, step=0.5)
        cloud_cover   = st.slider("Cloud Cover (%)",        0, 100, 50)
    with c4:
        heavy_rain    = st.selectbox("Heavy Rain",          ["No", "Yes"])
        is_weekend    = st.selectbox("Weekend?",            ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Time & Location ─────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Time & Location</div>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        season = st.selectbox("Season", ["winter", "summer", "monsoon", "post_monsoon"])
    with c6:
        time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
    with c7:
        state = st.selectbox("State", sorted([
           "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh",
           "delhi","goa","gujarat","haryana","himachal pradesh","jharkhand",
           "karnataka","kerala","madhya pradesh","maharashtra","manipur",
           "meghalaya","mizoram","nagaland","odisha","punjab","rajasthan",
           "sikkim","tamil nadu","telangana","tripura","uttar pradesh",
           "uttarakhand","west bengal","agartala"
        ]))
        
    c8, c9 = st.columns(2)
    with c8:
        festival_period    = st.selectbox("Festival Period",     ["No", "Yes"])
    with c9:
        crop_burning       = st.selectbox("Crop Burning Season", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict AQI Category")

# ─── Prediction ────────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="card-title" style="margin-bottom:16px;">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        # Encode booleans
        hr   = 1 if heavy_rain   == "Yes" else 0
        iw   = 1 if is_weekend   == "Yes" else 0
        fp   = 1 if festival_period == "Yes" else 0
        cb   = 1 if crop_burning == "Yes" else 0

        # OHE for season, time_of_day, state
        cat_df = pd.DataFrame([[season, time_of_day, state]],
                              columns=["season", "time_of_day", "state"])
        ohe_arr = encoder.transform(cat_df)
        ohe_cols = encoder.get_feature_names_out(["season", "time_of_day", "state"])
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols)

        # Numeric features (same order as training)
        num_df = pd.DataFrame([[iw, dew_point, hr, cloud_cover,
                                 pm25, co, no2, so2, o3, aod, fp, cb]],
                              columns=["is_weekend","dew_point_c","heavy_rain",
                                       "cloud_cover_percent","pm2_5_ugm3","co_ugm3",
                                       "no2_ugm3","so2_ugm3","o3_ugm3","aod",
                                       "festival_period","crop_burning_season"])

        # Combine
        final_df = pd.concat([num_df, ohe_df], axis=1)
        final_df.columns = final_df.columns.str.replace(" ", "_")

        # Scale numeric cols
        num_cols = ["is_weekend","dew_point_c","heavy_rain","cloud_cover_percent",
                    "pm2_5_ugm3","co_ugm3","no2_ugm3","so2_ugm3","o3_ugm3","aod",
                    "festival_period","crop_burning_season"]
        # Scale only the columns scaler was trained on
        scaler_cols = scaler.feature_names_in_
        final_df[scaler_cols] = scaler.transform(final_df[scaler_cols])
        # Align columns to training
        train_cols = rf_model.feature_names_in_
        for col in train_cols:
            if col not in final_df.columns:
                final_df[col] = 0
        final_df = final_df[train_cols]

        # Predict
        pred_idx   = rf_model.predict(final_df)[0]
        pred_label = aqi_mapping[str(pred_idx)]
        proba      = rf_model.predict_proba(final_df)[0]
        confidence = proba[pred_idx] * 100

        cfg = AQI_CONFIG[pred_label]

        # Result card
        st.markdown(f"""
        <div class="result-card" style="background:{cfg['bg']}; border-color:{cfg['color']}33;">
            <div class="result-emoji">{cfg['emoji']}</div>
            <div class="result-label" style="color:{cfg['color']};">{pred_label}</div>
            <div class="result-caption">Predicted AQI Category</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        st.markdown(f"""
        <div style="margin-top:20px;">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#8892a4;">
                <span>Model Confidence</span><span style="color:{cfg['color']};font-weight:600;">{confidence:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence}%;background:linear-gradient(90deg,{cfg['color']},#7BC24A);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # All class probabilities
        st.markdown('<div style="margin-top:24px;"><div class="card-title">All Category Probabilities</div>', unsafe_allow_html=True)
        labels_order = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy"]
        prob_df = pd.DataFrame({
            "Category": labels_order,
            "Probability (%)": [round(proba[i]*100, 1) for i in range(5)]
        })
        for _, row in prob_df.iterrows():
            c = AQI_CONFIG[row["Category"]]
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px;">
                    <span style="color:#c8d0de;">{c['emoji']} {row['Category']}</span>
                    <span style="color:{c['color']};font-weight:600;">{row['Probability (%)']:.1f}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div style="height:100%;border-radius:99px;width:{row['Probability (%)']:.1f}%;background:{c['color']};opacity:0.85;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Input summary pills
        st.markdown(f"""
        <div style="margin-top:24px;" class="card-title">Input Summary</div>
        <div class="metric-row">
            <div class="metric-pill"><div class="metric-pill-val">{pm25}</div><div class="metric-pill-lbl">PM2.5</div></div>
            <div class="metric-pill"><div class="metric-pill-val">{no2}</div><div class="metric-pill-lbl">NO₂</div></div>
            <div class="metric-pill"><div class="metric-pill-val">{so2}</div><div class="metric-pill-lbl">SO₂</div></div>
            <div class="metric-pill"><div class="metric-pill-val">{o3}</div><div class="metric-pill-lbl">O₃</div></div>
            <div class="metric-pill"><div class="metric-pill-val">{co}</div><div class="metric-pill-lbl">CO</div></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#2a3245;">
            <div style="font-size:4rem;">🌿</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;margin-top:12px;">
                Fill in the form and click Predict
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 16px;color:#2a3245;font-size:0.78rem;">
    AQI India Predictor · Random Forest · 85.3% Accuracy
</div>
""", unsafe_allow_html=True)
