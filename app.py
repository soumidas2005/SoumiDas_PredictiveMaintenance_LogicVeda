import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ks_2samp

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🚀 Predictive Maintenance Dashboard")

# ======================================
# SIDEBAR (Day 15)
# ======================================
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.6)

# ======================================
# SAMPLE DATA
# ======================================
np.random.seed(42)

time = np.arange(0, 50)
sensor_values = np.random.normal(50, 10, 50)
risk = np.random.uniform(0, 1, 50)

# ======================================
# DAY 16: TREND + ANOMALY
# ======================================
st.subheader("📈 Sensor Trend & Anomaly Detection")

df = pd.DataFrame({
    "Time": time,
    "Sensor": sensor_values,
    "Risk": risk
})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Time"],
    y=df["Sensor"],
    mode='lines',
    name='Sensor Value'
))

# Anomaly
anomaly = df[df["Risk"] > threshold]

fig.add_trace(go.Scatter(
    x=anomaly["Time"],
    y=anomaly["Sensor"],
    mode='markers',
    marker=dict(color='red', size=8),
    name='Anomaly'
))

st.plotly_chart(fig, use_container_width=True)

# ======================================
# DAY 17: RUL + GAUGE
# ======================================
st.subheader("⏳ Remaining Useful Life (RUL)")

predicted_rul = np.linspace(100, 10, 50)

fig_rul = go.Figure()
fig_rul.add_trace(go.Scatter(y=predicted_rul, mode='lines', name='RUL'))

st.plotly_chart(fig_rul, use_container_width=True)

latest_rul = int(predicted_rul[0])
max_rul = int(np.max(predicted_rul))

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=latest_rul,
    title={'text': "Days Remaining"},
    gauge={
        'axis': {'range': [0, max_rul]},
        'steps': [
            {'range': [0, max_rul*0.3], 'color': "red"},
            {'range': [max_rul*0.3, max_rul*0.6], 'color': "yellow"},
            {'range': [max_rul*0.6, max_rul], 'color': "green"}
        ]
    }
))

st.plotly_chart(gauge, use_container_width=True)

# ======================================
# DAY 18: ALERT SYSTEM
# ======================================
st.subheader("🚨 Alert System")

current_value = float(np.random.rand())

st.write(f"Current Risk Value: {current_value:.2f}")

if current_value > threshold:
    st.error("⚠️ ALERT TRIGGERED")
else:
    st.success("✅ System Normal")

# ======================================
# DAY 19: DRIFT DETECTION
# ======================================
st.subheader("📊 Data Drift Detection")

historical = np.random.normal(0.3, 0.1, 100)
live = np.array(risk)

ks_stat, p_value = ks_2samp(historical, live)

st.write(f"KS Statistic: {ks_stat:.4f}")
st.write(f"P-Value: {p_value:.4f}")

if p_value < 0.05:
    st.error("⚠️ Drift Detected")
else:
    st.success("✅ No Drift")

# ======================================
# DAY 20 + DAY 21: PIPELINE
# ======================================
st.subheader("⚙️ Monitoring Pipeline")

if st.button("▶ Run Full Pipeline"):

    st.write("Running pipeline...")

    # Drift Check
    if p_value < 0.05:
        st.error("⚠ Drift Detected → Retraining")

        # Model comparison
        old_acc = 0.82
        new_acc = round(np.random.uniform(0.85, 0.95), 3)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Old Model Accuracy", old_acc)

        with col2:
            st.metric("New Model Accuracy", new_acc)

        # Promotion
        if new_acc > old_acc:
            st.success("✅ New Model Promoted")
        else:
            st.warning("❌ Old Model Retained")

    else:
        st.success("✅ No Drift → No Action Needed")

    st.info("🎯 Pipeline Completed")