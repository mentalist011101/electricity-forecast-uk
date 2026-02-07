
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("UK Electricity Load Forecasting")

# ======================
# DATA LOADING
# ======================
@st.cache_data
def load_data():
    daily_energy = pd.read_pickle("daily_energy.pkl")
    trend = pd.read_pickle("trend.pkl")
    seasonal = pd.read_pickle("seasonal.pkl")
    residual = pd.read_pickle("residual.pkl")
    anomalies = pd.read_pickle("anomalies.pkl")
    final_preds = pd.read_pickle("final_preds.pkl")
    return daily_energy, trend, seasonal, residual, anomalies, final_preds

daily_energy, trend, seasonal, residual, anomalies, final_preds = load_data()

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Controls")
horizon = st.sidebar.selectbox(
    "Forecast horizon (days)",
    options=sorted(final_preds.keys())
)

# ======================
# SECTION 1 – OBSERVED VS FORECAST
# ======================
st.subheader("Observed vs Forecast")

df_plot = pd.DataFrame({
    "Observed": daily_energy,
    "Forecast": final_preds[horizon]
}).dropna()

st.line_chart(df_plot.tail(365))

# ======================
# SECTION 2 – STL COMPONENTS
# ======================
st.subheader("STL Decomposition")

fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
ax[0].plot(trend, label="Trend", color="orange")
ax[1].plot(seasonal, label="Seasonal", color="green")
ax[2].plot(residual, label="Residual", color="red")

ax[0].set_title("Trend")
ax[1].set_title("Seasonal")
ax[2].set_title("Residual")

plt.tight_layout()
st.pyplot(fig)

# ======================
# SECTION 3 – ANOMALIES
# ======================
st.subheader("Detected anomalies (Residual-based)")

fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(residual, label="Residual")
ax2.scatter(
    residual[anomalies].index,
    residual[anomalies],
    color="red",
    s=10,
    label="Anomaly"
)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.legend()
st.pyplot(fig2)

# ======================
# SECTION 4 – METRICS
# ======================
st.subheader("Model performance")

st.write("MAE increases with horizon, consistent with uncertainty propagation.")
