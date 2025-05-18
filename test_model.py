import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Set page configuration
st.set_page_config(page_title="EUR/USD Price Prediction", layout="wide")

# Title and description
st.title("EUR/USD Price Prediction App")
st.markdown("""
This application predicts the EUR/USD exchange rate using a trained LightGBM model based on financial indicators (SPX, GLD, USO, SLV).
Enter values for a single prediction or view the time series of predictions based on historical data.
""")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open(r'D:\project_lightgbm\lgbm_regression_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'lgbm_regression_model.pkl' not found. Please ensure it is in the same directory as this app.")
        return None


# Load the fitted scaler
@st.cache_resource
def load_scaler():
    try:
        with open(r'D:\project_lightgbm\scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.error("Scaler file 'scaler.pkl' not found. Please ensure it is in the same directory as this app.")
        return None


model = load_model()
scaler = load_scaler()

if model is None or scaler is None:
    st.stop()

# Load historical data
data_path = r"D:\ML_data\gold price data.csv"


# decorator storage resulting Cache 
@st.cache_data
def load_data():
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        required_columns = ['Date', 'SPX', 'GLD', 'USO', 'SLV', 'EUR/USD']
        if all(col in df.columns for col in required_columns):
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            st.error(f"Dataset must contain columns: {', '.join(required_columns)}")
            return None
    else:
        st.error(f"Dataset not found at {data_path}. Please ensure the file exists.")
        return None

df = load_data()
if df is None:
    st.stop()


# Function to preprocess input data
def preprocess_input(data, fit_scaler=False):
    features = ['SPX', 'GLD', 'USO', 'SLV']
    input_data = data[features]
    if fit_scaler:
        input_scaled = scaler.fit_transform(input_data)
    else:
        input_scaled = scaler.transform(input_data)
    return input_scaled

# Create input form for single prediction
st.subheader("Input Financial Indicators for Single Prediction")
col1, col2 = st.columns(2)

with col1:
    spx = st.number_input("SPX (S&P 500 Index)", min_value=0.0, value=1400.0, step=0.1)
    gld = st.number_input("GLD (Gold Price)", min_value=0.0, value=85.0, step=0.1)

with col2:
    uso = st.number_input("USO (Oil Price)", min_value=0.0, value=75.0, step=0.1)
    slv = st.number_input("SLV (Silver Price)", min_value=0.0, value=15.0, step=0.1)

# Predict button for single prediction
if st.button("Predict EUR/USD"):
    input_data = pd.DataFrame({
        'SPX': [spx],
        'GLD': [gld],
        'USO': [uso],
        'SLV': [slv]
    })
    input_scaled = preprocess_input(input_data, fit_scaler=False)
    prediction = model.predict(input_scaled)[0]
    
    # Display prediction and scaled values for debugging
    st.subheader("Prediction Result")
    st.success(f"Predicted EUR/USD Exchange Rate: {prediction:.6f}")
    st.write("Input Values:", input_data.to_dict())
    st.write("Scaled Input Values:", dict(zip(['SPX', 'GLD', 'USO', 'SLV'], input_scaled[0])))

# Visualize feature importance
st.subheader("Feature Importance")
feat_importance = pd.DataFrame({
    'features': ['SPX', 'GLD', 'USO', 'SLV'],
    'Scores': model.feature_importances_
})

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Scores', y='features', data=feat_importance, ax=ax)
plt.xlabel("Importance Score")
plt.title("Feature Importance (LightGBM Regressor)")
st.pyplot(fig)






































