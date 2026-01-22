import streamlit as st
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "processed", "sales_features.csv")

RF_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
XGB_PATH = os.path.join(BASE_DIR, "models", "xgboost.pkl")
LSTM_PATH = os.path.join(BASE_DIR, "models", "lstm_model.keras")
RF_FEATURES_PATH = os.path.join(BASE_DIR, "models", "rf_features.pkl")

# ---------------- TITLE ----------------
st.title("📈 Sales Forecasting Dashboard")
st.caption("End-to-End Time Series Forecasting using ML & Deep Learning")

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv(DATA_PATH)
except Exception:
    st.error("❌ Processed data not found. Please run the notebooks first.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

model_name = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["Random Forest", "XGBoost", "LSTM"]
)

forecast_days = st.sidebar.selectbox(
    "Forecast Horizon (days)",
    [7, 30]
)

# ---------------- MODEL DESCRIPTION ----------------
if model_name == "Random Forest":
    st.info("🌲 **Random Forest** uses ensemble decision trees to capture non-linear patterns.")
elif model_name == "XGBoost":
    st.info("🚀 **XGBoost** is a gradient boosting model optimized for accuracy and speed.")
else:
    st.info("🧠 **LSTM** is a deep learning model designed for sequential time-series data.")

# ---------------- LOAD MODEL SAFELY ----------------
try:
    if model_name == "Random Forest":
        model = joblib.load(RF_PATH)
        feature_names = joblib.load(RF_FEATURES_PATH)

    elif model_name == "XGBoost":
        model = joblib.load(XGB_PATH)
        feature_names = joblib.load(RF_FEATURES_PATH)

    else:
        model = load_model(LSTM_PATH)

except Exception:
    st.error("❌ Selected model could not be loaded. Please retrain the model.")
    st.stop()

# ---------------- HISTORICAL SALES ----------------
st.subheader("📊 Historical Sales")
st.line_chart(df.set_index("date")["sales"])

# ---------------- FUTURE FORECAST FUNCTION ----------------
def forecast_future(model, last_row, n_days, feature_names):
    future_preds = []
    current_input = last_row[feature_names].copy()

    for _ in range(n_days):
        pred = model.predict(current_input.values.reshape(1, -1))[0]
        future_preds.append(pred)

        # Shift lag features
        for i in range(len(feature_names) - 1, 0, -1):
            current_input.iloc[i] = current_input.iloc[i - 1]

        current_input.iloc[0] = pred

    return future_preds

# ---------------- PREDICTIONS ----------------
if model_name in ["Random Forest", "XGBoost"]:
    X = df.drop(columns=["sales", "date"])

    preds = model.predict(X)

    st.subheader("📈 Model Predictions (Historical)")
    pred_df = pd.DataFrame({
        "Actual Sales": df["sales"].values,
        "Predicted Sales": preds
    })
    st.line_chart(pred_df)

    # ---------------- FUTURE FORECAST ----------------
    last_row = X.iloc[-1]
    future_preds = forecast_future(
        model,
        last_row,
        forecast_days,
        feature_names
    )

    st.subheader(f"🔮 Future {forecast_days}-Day Forecast")
    future_df = pd.DataFrame({
        "Day": range(1, forecast_days + 1),
        "Predicted Sales": future_preds
    })
    st.line_chart(future_df.set_index("Day"))

else:
    st.warning("⚠️ LSTM future forecasting is demonstrated in notebooks only.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by Nidhi Gupta | Sales Forecasting using Time Series & Machine Learning")

