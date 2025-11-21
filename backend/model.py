import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def load_data(stock_symbol, start, end):
    data = yf.download(stock_symbol, start=start, end=end)
    return data

def prepare_data(data):
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def train_model(data):
    x_train, y_train = [], []
    for i in range(60, len(data)):
        x_train.append(data[i-60:i, 0])
        y_train.append(data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

def predict_future(model, data, scaler):
    last_60_days = data[-60:]
    X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price[0][0]

def get_buy_sell_signal(current_price, predicted_price):
    if predicted_price > current_price:
        return "BUY"
    elif predicted_price < current_price:
        return "SELL"
    else:
        return "HOLD"
