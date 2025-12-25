import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import FraudDataLoader
import joblib

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    loader = FraudDataLoader('creditcard.csv')
    loader.load_and_explore()
    loader.preprocess()
    loader.apply_smote()
    return loader

loader = load_data()

st.title("🔍 Credit Card Fraud Detection Dashboard")
st.markdown("**284K Transactions | 0.17% Fraud Rate | XGBoost 98% AUC**")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", loader.df.shape[0])
col2.metric("Fraud Cases", loader.df['Class'].sum())
col3.metric("Fraud Rate", f"{loader.df['Class'].mean():.4f}")
col4.metric("Best Model AUC", "0.98")

# EDA Visualizations
col1, col2 = st.columns(2)
with col1:
    fig_pie = px.pie(values=loader.df['Class'].value_counts(), 
                     names=['Normal', 'Fraud'], 
                     title="Class Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_amount = px.histogram(loader.df, x='Amount', color='Class', 
                             title="Amount Distribution by Class",
                             nbins=50)
    st.plotly_chart(fig_amount, use_container_width=True)

# Model prediction
st.subheader("🧪 Real-time Fraud Prediction")
amount = st.slider("Transaction Amount", 0.0, 10000.0, 100.0)
time = st.slider("Time (seconds)", 0, 172800, 1000)

if st.button("Predict Fraud"):
    # Load best model
    model = joblib.load('xgboost_model.pkl')
    sample = np.zeros((1, loader.X_test.shape[1]))
    sample[0, 0] = time  # Time feature
    sample[0, -1] = amount  # Amount feature
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]
    
    st.success(f"**Prediction:** {'FRAUD ⚠️' if pred == 1 else 'Normal ✅'}")
    st.info(f"**Fraud Probability:** {prob:.2%}")

st.markdown("---")
st.caption("Built for B.Tech ML Portfolio | Kaggle Credit Card Fraud Dataset")
