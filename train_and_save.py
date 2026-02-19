import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# ── 1. LOAD ───────────────────────────────────────────────
df = pd.read_csv("diabetes.csv")

# ── 2. MISSING VALUE IMPUTATION (outcome-stratified median)
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)

def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

for col in df.columns.drop("Outcome"):
    df.loc[(df['Outcome'] == 0) & (df[col].isnull()), col] = median_target(col)[col][0]
    df.loc[(df['Outcome'] == 1) & (df[col].isnull()), col] = median_target(col)[col][1]

# ── 3. OUTLIER CAPPING (IQR) ─────────────────────────────
def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.25)
    q3 = dataframe[variable].quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def replace_with_thresholds(dataframe, columns):
    for var in columns:
        low, up = outlier_thresholds(dataframe, var)
        dataframe.loc[dataframe[var] < low, var] = low
        dataframe.loc[dataframe[var] > up, var] = up

df[df.columns] = df[df.columns].astype(float)
replace_with_thresholds(df, df.columns)

# ── 4. FEATURE ENGINEERING ───────────────────────────────
# BMI categories
df["NewBMI"] = "Obesity 3"
df.loc[df["BMI"] < 18.5, "NewBMI"] = "Underweight"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = "Normal"
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = "Overweight"
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = "Obesity 1"
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = "Obesity 2"

# Interaction features (no Outcome leakage here — safe to replicate)
df["NEW_g_p"] = df["Glucose"] * df["Pregnancies"]
df["NEW_i_g"] = df["Glucose"] * df["Insulin"]

# Glucose bands
df['New_Glucose'] = pd.cut(
    df['Glucose'], bins=[0, 74, 99, 139, 200],
    labels=["Low", "Normal", "Overweight", "High"]
)

# Insulin score
df["NewInsulinScore"] = df.apply(
    lambda row: "Normal" if 16 <= row["Insulin"] <= 166 else "Abnormal", axis=1
)

# ── 5. ONE-HOT ENCODE ─────────────────────────────────────
categorical_columns = [col for col in df.columns
                       if len(df[col].unique()) <= 10 and col != "Outcome"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df.columns = df.columns.str.replace(" ", "_")

# ── 6. SPLIT & SCALE ──────────────────────────────────────
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Save column list — app.py needs this exact order
joblib.dump(list(X.columns), "feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 7. TRAIN & SAVE ───────────────────────────────────────
model = LGBMClassifier(
    random_state=42, class_weight='balanced',
    n_estimators=300, learning_rate=0.05
)
model.fit(X_train_scaled, y_train)

joblib.dump(model,  "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Saved: diabetes_model.pkl | scaler.pkl | feature_columns.pkl")
