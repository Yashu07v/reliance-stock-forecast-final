import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Reliance Stock Forecast",
    layout="wide",
    page_icon="📈"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

h1 {
    color: #38bdf8;
    text-align: center;
    font-size: 50px;
    font-weight: bold;
}

h2, h3 {
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="metric-container"] {
    background-color: #1e293b;
    border: 1px solid #38bdf8;
    padding: 15px;
    border-radius: 15px;
    color: white;
    box-shadow: 0px 0px 15px rgba(56,189,248,0.4);
}

div.stButton > button {
    background-color: #38bdf8;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

div.stDownloadButton > button {
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.title("📈 Reliance Industries Stock Forecast")

st.markdown("""
<div style="
background: linear-gradient(to right,#0f172a,#1e40af);
padding:25px;
border-radius:15px;
margin-bottom:25px;
">

<h2 style="color:white;text-align:center;">
📈 Stock Forecasting Dashboard
</h2>

<p style="
color:#dbeafe;
text-align:center;
font-size:18px;
">
Advanced Analysis with Market Trend Forecasting
</p>

</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📊 Dashboard")

st.sidebar.info("""
Reliance Industries Stock Analysis
using LSTM Deep Learning
""")

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    df = pd.read_excel("Company stock prices.xlsx")
    return df

df = load_data()

# =========================
# PREPROCESSING
# =========================

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')

df['MA10'] = df['Close'].rolling(10).mean()

df['MA50'] = df['Close'].rolling(50).mean()

# =========================
# KPI CARDS
# =========================

latest_close = round(df['Close'].iloc[-1], 2)

highest_price = round(df['High'].max(), 2)

lowest_price = round(df['Low'].min(), 2)

col1, col2, col3 = st.columns(3)

col1.metric(
    "📌 Latest Close",
    f"₹ {latest_close}"
)

col2.metric(
    "📈 Highest Price",
    f"₹ {highest_price}"
)

col3.metric(
    "📉 Lowest Price",
    f"₹ {lowest_price}"
)

# =========================
# PREPARE DATA
# =========================

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

close_data = df[['Close']].values

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(close_data)

x_train = []
y_train = []

for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(
    x_train,
    (x_train.shape[0], x_train.shape[1], 1)
)

# =========================
# BUILD MODEL
# =========================

model = Sequential()

model.add(LSTM(
    50,
    return_sequences=True,
    input_shape=(60,1)
))

model.add(LSTM(
    50,
    return_sequences=False
))

model.add(Dense(25))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=1,
    verbose=0
)

# =========================
# FUTURE FORECAST
# =========================

future_days = 30

future_predictions = []

last_60_days = scaled_data[-60:]

current_batch = last_60_days.reshape(1, 60, 1)

for i in range(future_days):

    pred = model.predict(
        current_batch,
        verbose=0
    )[0]

    future_predictions.append(pred)

    current_batch = np.append(
        current_batch[:,1:,:],
        [[pred]],
        axis=1
    )

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

# =========================
# FINAL PREDICTION
# =========================

final_prediction = round(
    future_predictions[-1][0],
    2
)

prediction_change = round(
    (
        (
            final_prediction -
            latest_close
        ) / latest_close
    ) * 100,
    2
)

# =========================
# SIDEBAR MODEL INSIGHTS
# =========================

st.sidebar.markdown("## 🤖 Model Insights")

st.sidebar.success(f"""
Predicted 30-Day Price

₹ {final_prediction}
""")

st.sidebar.metric(
    "Current Price",
    f"₹ {latest_close}"
)

st.sidebar.metric(
    "Forecast Change",
    f"{prediction_change}%"
)

st.sidebar.markdown("---")

st.sidebar.info("""
Model Used:
• LSTM Deep Learning

Features:
• Open
• High
• Low
• Close
• Moving Averages

Forecast:
• Next 30 Days
""")

st.sidebar.warning("""
Stock predictions are probabilistic
and affected by market conditions.
""")

# =========================
# HISTORICAL DATA
# =========================

st.subheader("📋 Historical Stock Data")

st.dataframe(
    df.tail(),
    use_container_width=True
)

# =========================
# CLOSE PRICE GRAPH
# =========================

st.subheader("📊 Reliance Close Price Trend")

fig1, ax1 = plt.subplots(figsize=(9,3.5))

fig1.patch.set_facecolor('#0f172a')

ax1.set_facecolor('#0f172a')

ax1.plot(
    df['Date'],
    df['Close'],
    color='#38bdf8',
    linewidth=2.5
)

ax1.grid(alpha=0.2)

ax1.tick_params(colors='white')

ax1.set_title(
    "Historical Close Price",
    color='white'
)

ax1.set_xlabel(
    "Date",
    color='white'
)

ax1.set_ylabel(
    "Close Price",
    color='white'
)

st.pyplot(fig1)

# =========================
# MOVING AVERAGE GRAPH
# =========================

st.subheader("📉 Market Trends")

fig2, ax2 = plt.subplots(figsize=(9,3.5))

fig2.patch.set_facecolor('#0f172a')

ax2.set_facecolor('#0f172a')

ax2.plot(
    df['Date'],
    df['Close'],
    label='Close Price',
    linewidth=2,
    color='#38bdf8'
)

ax2.plot(
    df['Date'],
    df['MA10'],
    label='10-Day MA',
    linewidth=2,
    color='#22c55e'
)

ax2.plot(
    df['Date'],
    df['MA50'],
    label='50-Day MA',
    linewidth=2,
    color='#f97316'
)

ax2.legend()

ax2.grid(alpha=0.2)

ax2.tick_params(colors='white')

ax2.set_title(
    "Moving Average Trends",
    color='white'
)

st.pyplot(fig2)

# =========================
# FORECAST CARD
# =========================

st.markdown(f"""
<div style="
background: linear-gradient(to right,#22c55e,#16a34a);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">

<h2 style="color:white;">
🚀 Predicted Stock Price After 30 Days
</h2>

<h1 style="color:white;">
₹ {final_prediction}
</h1>

<h3 style="color:white;">
Expected Change: {prediction_change}%
</h3>

</div>
""", unsafe_allow_html=True)

# =========================
# FORECAST GRAPH
# =========================

st.subheader("🔮 30-Day Future Forecast")

future_days_array = np.arange(1,31)

fig3, ax3 = plt.subplots(figsize=(9,3.5))

fig3.patch.set_facecolor('#0f172a')

ax3.set_facecolor('#0f172a')

ax3.plot(
    future_days_array,
    future_predictions,
    color='#22c55e',
    linewidth=3
)

ax3.fill_between(
    future_days_array,
    future_predictions.flatten(),
    alpha=0.3
)

ax3.grid(alpha=0.2)

ax3.tick_params(colors='white')

ax3.set_title(
    "30-Day Forecast",
    color='white'
)

ax3.set_xlabel(
    "Future Days",
    color='white'
)

ax3.set_ylabel(
    "Predicted Price",
    color='white'
)

st.pyplot(fig3)

# =========================
# FORECAST TABLE
# =========================

forecast_df = pd.DataFrame({
    "Day": future_days_array,
    "Predicted Price": future_predictions.flatten()
})

st.subheader("📄 Forecast Table")

st.dataframe(
    forecast_df,
    use_container_width=True
)

# =========================
# DOWNLOAD CSV
# =========================

csv = forecast_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Forecast CSV",
    data=csv,
    file_name="Reliance_30_Day_Forecast.csv",
    mime="text/csv"
)

# =========================
# FOOTER
# =========================

st.markdown("""
<hr style='border:1px solid #334155'>

<div style='text-align:center;color:lightgray;'>

Built using Streamlit, TensorFlow, LSTM Deep Learning & Python

</div>
""", unsafe_allow_html=True)