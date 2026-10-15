# app.py — Air Passengers Forecasting App
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model and the historical data
model = joblib.load("forecast_model.pkl")
df = pd.read_csv("airline_passengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df = df.set_index("Month").asfreq("MS")

st.title("Air Passengers Forecasting")
st.write("Forecast future monthly passenger numbers using the trained model.")

# User picks how many months to forecast
n_months = st.slider("Months to forecast", 1, 24, 12)

if st.button("Forecast"):
    history = df["Passengers"].copy()  # actual data, extended with predictions
    future_dates = pd.date_range(start="1961-01-01", periods=n_months, freq="MS")
    forecasts = []

    for date in future_dates:
        lag_1 = history.iloc[-1]  # previous month (actual or already predicted)
        lag_12 = history[date - pd.DateOffset(years=1)]  # same month last year
        features = pd.DataFrame([{
            "Lag_1": lag_1,
            "Lag_12": lag_12,
            "Month_Num": date.month,
            "Year": date.year,
        }])
        pred = model.predict(features)[0]
        forecasts.append(pred)
        history[date] = pred  # feed the prediction back for the next month

    forecast_df = pd.DataFrame({"Month": future_dates, "Forecast": forecasts})

    # Plot history + forecast
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Passengers"], label="History (1949-1960)")
    ax.plot(forecast_df["Month"], forecast_df["Forecast"], label="Forecast", linestyle="--")
    ax.set_title("Air Passengers: History and Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Passengers")
    ax.legend()
    st.pyplot(fig)

    # Show the forecast table and offer a download
    st.write(forecast_df)
    csv = forecast_df.to_csv(index=False)
    st.download_button("Download forecast as CSV", csv, "forecast.csv", "text/csv")