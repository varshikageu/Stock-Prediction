import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from model import load_data, prepare_data, train_model, predict_future, get_buy_sell_signal

st.title("📈 Stock Price Prediction App (ML Based)")
st.write("Predict stock trends using an LSTM model and visualize data between two dates.")

# User inputs
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, TCS.NS):", "AAPL")
start_date = st.date_input("Start Date", date(2023, 1, 1))
end_date = st.date_input("End Date", date(2024, 1, 1))

if st.button("Run Prediction"):
    with st.spinner("Fetching data and training model... Please wait ⏳"):
        data = load_data(stock_symbol, start_date, end_date)
        st.subheader(f"Stock Data for {stock_symbol}")
        st.line_chart(data['Close'])

        scaled_data, scaler = prepare_data(data)
        model = train_model(scaled_data)

        predicted_price = predict_future(model, scaled_data, scaler)
        current_price = data['Close'].iloc[-1]
        signal = get_buy_sell_signal(current_price, predicted_price)

        st.success(f"📊 Predicted Next Day Price: ${predicted_price:.2f}")
        st.info(f"💡 Current Price: ${current_price:.2f}")
        st.warning(f"📈 Suggested Action: {signal}")

        # Plot comparison
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Close'], label="Actual Prices")
        ax.axhline(y=predicted_price, color='r', linestyle='--', label="Predicted Next Price")
        plt.legend()
        st.pyplot(fig)
