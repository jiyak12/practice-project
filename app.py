import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# ── LOAD ARTIFACTS ────────────────────────────────────────
model           = joblib.load("diabetes_model.pkl")
scaler          = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ── FEATURE ENGINEERING (mirrors notebook exactly) ────────
def engineer_features(preg, glucose, bp, skin, insulin, bmi, dpf, age):
    data = {
        "Pregnancies": preg, "Glucose": glucose,
        "BloodPressure": bp, "SkinThickness": skin,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age
    }
    df = pd.DataFrame([data])

    # BMI category
    def bmi_cat(b):
        if b < 18.5:         return "Underweight"
        elif b <= 24.9:      return "Normal"
        elif b <= 29.9:      return "Overweight"
        elif b <= 34.9:      return "Obesity_1"
        elif b <= 39.9:      return "Obesity_2"
        else:                return "Obesity_3"

    df["NewBMI"] = df["BMI"].apply(bmi_cat)

    # Interaction features
    df["NEW_g_p"] = df["Glucose"] * df["Pregnancies"]
    df["NEW_i_g"] = df["Glucose"] * df["Insulin"]

    # Glucose bands
    df["New_Glucose"] = pd.cut(
        df["Glucose"], bins=[0, 74, 99, 139, 200],
        labels=["Low", "Normal", "Overweight", "High"]
    ).astype(str)

    # Insulin score
    df["NewInsulinScore"] = "Normal" if 16 <= insulin <= 166 else "Abnormal"

    # One-hot encode
    cat_cols = ["NewBMI", "New_Glucose", "NewInsulinScore"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df.columns = df.columns.str.replace(" ", "_")

    # Align to training columns — missing OHE cols become 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

# ── RISK LABEL ────────────────────────────────────────────
def categorise_risk(prob):
    if prob < 0.30:
        return "🟢 Low Risk", "success"
    elif prob < 0.70:
        return "🟡 Moderate Risk", "warning"
    else:
        return "🔴 High Risk", "error"

# ── UI ────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("Predict the probability of diabetes using medical parameters.")
st.markdown("---")

st.sidebar.header("Enter Patient Details")
pregnancies    = st.sidebar.slider("Pregnancies",                  0,   20,  1)
glucose        = st.sidebar.slider("Glucose Level",                0,  200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure",               0,  150,  70)
skin_thickness = st.sidebar.slider("Skin Thickness",               0,  100,  20)
insulin        = st.sidebar.slider("Insulin",                      0,  900,  80)
bmi            = st.sidebar.slider("BMI",                        0.0, 70.0, 25.0)
dpf            = st.sidebar.slider("Diabetes Pedigree Function", 0.0,  3.0,  0.5)
age            = st.sidebar.slider("Age",                         21,  120,  30)

if st.button("🔍 Predict Risk"):
    X_input  = engineer_features(pregnancies, glucose, blood_pressure,
                                  skin_thickness, insulin, bmi, dpf, age)
    X_scaled = scaler.transform(X_input)

    probability = model.predict_proba(X_scaled)[0][1]
    label, level = categorise_risk(probability)

    st.markdown("## 📊 Prediction Result")
    st.progress(int(probability * 100))

    if level == "success":
        st.success(f"{label}  —  {probability*100:.2f}%")
    elif level == "warning":
        st.warning(f"{label}  —  {probability*100:.2f}%")
    else:
        st.error(f"{label}  —  {probability*100:.2f}%")

st.markdown("---")
st.caption("⚠️ Disclaimer: For educational purposes only. Not a substitute for medical advice.")
