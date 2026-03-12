import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paddy Yield Predictor 🌾",
    page_icon="🌾",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("paddy_yield_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 Agricultural Yield Intelligence System")
st.markdown("**Predict Paddy Yield using Machine Learning**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 About This App")
    st.info(
        "This app predicts the **total paddy yield in Kg** based on "
        "agronomic, soil, nursery, fertilizer, and weather inputs. "
        "The model was trained using 9 ML algorithms with GridSearchCV "
        "and the best model was selected based on Test R² Score."
    )
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    - **Algorithms compared:** 9  
    - **Tuning:** GridSearchCV (5-fold CV)  
    - **Pipeline:** Preprocessor → SelectKBest → Model  
    - **Target:** Paddy Yield (Kg)
    """)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Predict Yield", "📖 How to Use"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:

    if not model_loaded:
        st.error("⚠️ Model file `paddy_yield_model.pkl` not found. Place it in the same folder as `app.py`.")
        st.stop()

    st.subheader("📝 Enter Farm & Weather Details")

    # ── Section 1: Field & Agronomic ──────────────────────────
    st.markdown("#### 🌱 Field & Agronomic Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        Hectares       = st.selectbox("Hectares (Land Area)", options=[1, 2, 3, 4, 5, 6])
        Agriblock      = st.selectbox("Agriblock (Region)",
                                      options=["Cuddalore", "Kurinjipadi", "Panruti",
                                               "Kallakurichi", "Sankarapuram", "Chinnasalem"])
        Variety        = st.selectbox("Paddy Variety", options=["CO_43", "ponmani", "delux ponni"])

    with col2:
        Soil_Types     = st.selectbox("Soil Type", options=["alluvial", "clay"])
        Nursery        = st.selectbox("Nursery Method", options=["dry", "wet"])
        Seedrate_in_Kg = st.number_input("Seed Rate (Kg)", min_value=50, max_value=300, value=150, step=10)

    with col3:
        Urea_40Days      = st.number_input("Urea at 40 Days (Kg)",      min_value=0, max_value=200,  value=80,  step=5)
        Potassh_50Days   = st.number_input("Potash at 50 Days (Kg)",    min_value=0, max_value=200,  value=60,  step=5)
        Pest_60Day_in_ml = st.number_input("Pesticide at 60 Days (ml)", min_value=0, max_value=1000, value=200, step=10)

    Trash_in_bundles = st.number_input("Trash (Bundles)", min_value=0, max_value=1000, value=500, step=10)

    st.markdown("---")

    # ── Section 2: Rainfall ────────────────────────────────────
    st.markdown("#### 🌧️ Rainfall (mm)")
    ra, rb, rc, rd = st.columns(4)
    re, rf, rg, rh = st.columns(4)
    with ra:
        v_30DRain__in_mm    = st.number_input("30D Rain",     min_value=0.0, max_value=500.0, value=18.5, step=0.1, key="ra")
    with rb:
        v_30DAI_in_mm       = st.number_input("30DAI",        min_value=0.0, max_value=500.0, value=18.5, step=0.1, key="rb")
    with rc:
        v_30_50DRain__in_mm = st.number_input("30-50D Rain",  min_value=0.0, max_value=500.0, value=18.5, step=0.1, key="rc")
    with rd:
        v_30_50DAI_in_mm    = st.number_input("30-50DAI",     min_value=0.0, max_value=500.0, value=18.5, step=0.1, key="rd")
    with re:
        v_51_70DRain_in_mm  = st.number_input("51-70D Rain",  min_value=0.0, max_value=500.0, value=18.5, step=0.1, key="re")
    with rf:
        v_51_70AI_in_mm     = st.number_input("51-70AI",      min_value=0.0, max_value=500.0, value=18.5, step=0.1, key="rf")
    with rg:
        v_71_105DRain_in_mm = st.number_input("71-105D Rain", min_value=0.0, max_value=500.0, value=18.5, step=0.1, key="rg")
    with rh:
        v_71_105DAI_in_mm   = st.number_input("71-105DAI",    min_value=0.0, max_value=500.0, value=18.5, step=0.1, key="rh")

    st.markdown("---")

    # ── Section 3: Temperature ─────────────────────────────────
    st.markdown("#### 🌡️ Temperature (°C) — 4 Intervals")
    intervals      = ["D1_D30", "D31_D60", "D61_D90", "D91_D120"]
    interval_names = ["Days 1–30", "Days 31–60", "Days 61–90", "Days 91–120"]
    temp_data = {}
    t_cols = st.columns(4)
    for i, (intv, name) in enumerate(zip(intervals, interval_names)):
        with t_cols[i]:
            st.markdown(f"**{name}**")
            temp_data[f"Min_temp_{intv}"] = st.number_input(
                "Min Temp", min_value=10.0, max_value=35.0, value=19.0, step=0.1, key=f"mint_{intv}")
            temp_data[f"Max_temp_{intv}"] = st.number_input(
                "Max Temp", min_value=20.0, max_value=45.0, value=32.0, step=0.1, key=f"maxt_{intv}")

    st.markdown("---")

    # ── Section 4: Wind — EXACT unique values per interval ─────
    st.markdown("#### 💨 Wind — 4 Intervals")

    # Each interval has its own unique wind direction values from training data
    wind_options = {
        "D1_D30":   ["SW", "NW", "ENE", "W", "SSE", "E"],
        "D31_D60":  ["W", "S", "NE", "WNW", "ENE"],
        "D61_D90":  ["NNW", "SE", "NNE", "SW", "NE"],
        "D91_D120": ["WSW", "SSE", "W", "S", "NW", "NNW"],
    }

    wind_data = {}
    w_cols = st.columns(4)
    for i, (intv, name) in enumerate(zip(intervals, interval_names)):
        with w_cols[i]:
            st.markdown(f"**{name}**")
            wind_data[f"Inst_Wind_Speed_{intv}_in_Knots"] = st.number_input(
                "Wind Speed (Knots)", min_value=0.0, max_value=50.0,
                value=5.0, step=0.1, key=f"ws_{intv}")
            wind_data[f"Wind_Direction_{intv}"] = st.selectbox(
                "Wind Direction", options=wind_options[intv], key=f"wd_{intv}")

    st.markdown("---")

    # ── Section 5: Humidity ────────────────────────────────────
    st.markdown("#### 💧 Relative Humidity (%) — 4 Intervals")
    humid_data = {}
    h_cols = st.columns(4)
    for i, (intv, name) in enumerate(zip(intervals, interval_names)):
        with h_cols[i]:
            humid_data[f"Relative_Humidity_{intv}"] = st.slider(
                f"{name}", min_value=40, max_value=100, value=80, key=f"hum_{intv}")

    st.markdown("---")

    # ── Predict Button ─────────────────────────────────────────
    pred_col, _ = st.columns([1, 3])
    with pred_col:
        predict_btn = st.button("🔮 Predict Yield", type="primary", use_container_width=True)

    if predict_btn:
        input_dict = {
            "Hectares":            Hectares,
            "Agriblock":           Agriblock,
            "Variety":             Variety,
            "Soil_Types":          Soil_Types,
            "Seedrate_in_Kg":      Seedrate_in_Kg,
            "Nursery":             Nursery,
            "Urea_40Days":         Urea_40Days,
            "Potassh_50Days":      Potassh_50Days,
            "Pest_60Day_in_ml":    Pest_60Day_in_ml,
            "30DRain__in_mm":      v_30DRain__in_mm,
            "30DAI_in_mm":         v_30DAI_in_mm,
            "30_50DRain__in_mm":   v_30_50DRain__in_mm,
            "30_50DAI_in_mm":      v_30_50DAI_in_mm,
            "51_70DRain_in_mm":    v_51_70DRain_in_mm,
            "51_70AI_in_mm":       v_51_70AI_in_mm,
            "71_105DRain_in_mm":   v_71_105DRain_in_mm,
            "71_105DAI_in_mm":     v_71_105DAI_in_mm,
            "Trash_in_bundles":    Trash_in_bundles,
        }
        input_dict.update(temp_data)
        input_dict.update(wind_data)
        input_dict.update(humid_data)

        input_df = pd.DataFrame([input_dict])

        try:
            prediction = model.predict(input_df)[0]

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("🌾 Predicted Paddy Yield", f"{prediction:,.0f} Kg")
            with r2:
                st.metric("📦 Approx. Bags (50 Kg)", f"{int(prediction // 50)} bags")
            with r3:
                st.metric("⚖️ In Tonnes", f"{prediction / 1000:.2f} T")

            if prediction >= 30000:
                st.success("✅ **High Yield** — Excellent farming conditions predicted!")
                st.balloons() 
            elif prediction >= 15000:
                st.warning("⚠️ **Medium Yield** — Average output expected.")
            else:
                st.error("❌ **Low Yield** — Consider improving soil/fertilizer inputs.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 2 — HOW TO USE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📖 How to Use This App")
    st.markdown("""
    ### Step-by-Step Guide

    **1. Field & Agronomic Details** — select land area, region, variety, soil, nursery, seed rate, fertilizers

    **2. Rainfall** — enter mm values for all 8 rainfall columns across growth stages

    **3. Temperature, Wind & Humidity** — enter values for all 4 intervals (D1–D30, D31–D60, D61–D90, D91–D120)

    **4. Click 🔮 Predict Yield** — result shows Kg, Bags, Tonnes + High/Medium/Low category

    ---
    ### 📊 Output Categories

    | Yield | Category | Meaning |
    |---|---|---|
    | ≥ 30,000 Kg | ✅ High | Excellent conditions |
    | 15,000–30,000 Kg | ⚠️ Medium | Average output |
    | < 15,000 Kg | ❌ Low | Poor conditions |

    ---
    ### 🛠️ Tech Stack
    """)
    c1, c2, c3, c4 = st.columns(4)
    c1.info("🐍 Python")
    c2.info("🤖 Scikit-learn")
    c3.info("📊 Streamlit")
    c4.info("🚀 XGBoost")