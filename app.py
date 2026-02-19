import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── LOAD ARTIFACTS ────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ── RISK CATEGORISATION (from predict_proba) ──────────────
def categorise_risk(probability):
    if probability < 0.35:
        return "🟢 Low Risk", "green", probability
    elif probability < 0.65:
        return "🟡 Moderate Risk", "orange", probability
    else:
        return "🔴 High Risk", "red", probability

# ── FEATURE ENGINEERING (mirrors your notebook exactly) ───
def engineer_features(input_dict):
    df = pd.DataFrame([input_dict])

    # BMI category
    def bmi_category(bmi):
        if bmi < 18.5:       return "Underweight"
        elif bmi <= 24.9:    return "Normal"
        elif bmi <= 29.9:    return "Overweight"
        elif bmi <= 34.9:    return "Obesity 1"
        elif bmi <= 39.9:    return "Obesity 2"
        else:                return "Obesity 3"

    df["NewBMI"] = df["BMI"].apply(bmi_category)

    # Glucose category
    df["New_Glucose"] = pd.cut(
        df["Glucose"],
        bins=[0, 74, 99, 139, 200],
        labels=["Low", "Normal", "Overweight", "High"]
    )
    df["New_Glucose"] = df["New_Glucose"].astype(str)

    # Insulin score
    df["NewInsulinScore"] = df["Insulin"].apply(
        lambda x: "Normal" if 16 <= x <= 166 else "Abnormal"
    )

    # One-hot encode (drop_first=True, same as training)
    categorical_columns = ["NewBMI", "New_Glucose", "NewInsulinScore"]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df.columns = df.columns.str.replace(" ", "_")

    # Align to training columns — fill missing OHE columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]  # same column order as training

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter the patient's medical details below to assess their diabetes risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies     = st.number_input("Pregnancies",         min_value=0,    max_value=20,   value=1)
    glucose         = st.number_input("Glucose (mg/dL)",     min_value=0,    max_value=300,  value=110)
    blood_pressure  = st.number_input("Blood Pressure (mmHg)", min_value=0,  max_value=200,  value=72)
    skin_thickness  = st.number_input("Skin Thickness (mm)", min_value=0,    max_value=100,  value=20)

with col2:
    insulin         = st.number_input("Insulin (mu U/ml)",   min_value=0,    max_value=900,  value=80)
    bmi             = st.number_input("BMI",                 min_value=0.0,  max_value=70.0, value=25.0, format="%.1f")
    dpf             = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age             = st.number_input("Age",                 min_value=21,   max_value=120,  value=30)

st.divider()

if st.button("🔍 Predict Risk", use_container_width=True):

    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    # Pipeline: engineer → scale → predict
    X_input = engineer_features(input_dict)
    X_scaled = scaler.transform(X_input)
    probability = model.predict_proba(X_scaled)[0][1]  # prob of diabetes

    label, colour, prob = categorise_risk(probability)

    # ── RESULTS ───────────────────────────────────────────
    st.subheader("Result")
    st.markdown(f"### {label}", unsafe_allow_html=False)
    st.progress(float(probability))
    st.metric("Diabetes Probability", f"{probability:.1%}")

    if colour == "green":
        st.success("Low probability of diabetes. Maintain a healthy lifestyle.")
    elif colour == "orange":
        st.warning("Moderate risk detected. Consider consulting a healthcare professional.")
    else:
        st.error("High risk detected. Please seek medical advice promptly.")

    with st.expander("📊 Probability Breakdown"):
        st.write(f"- **Low Risk threshold** : < 35% → Your score: `{probability:.1%}`")
        st.write(f"- **Moderate Risk threshold** : 35% – 65% → Your score: `{probability:.1%}`")
        st.write(f"- **High Risk threshold** : > 65% → Your score: `{probability:.1%}`")

    st.caption("⚠️ This tool is for informational purposes only and is not a clinical diagnosis.")
