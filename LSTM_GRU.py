import os
import warnings
from itertools import product
from random import seed

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from constants import tickers, metals, metal_pairs, BASE, analysis_end_date

warnings.filterwarnings('ignore')
seed(42)

df = pd.read_csv(f'{BASE}.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df = df.iloc[::-1]
df_all = df[df.index < pd.to_datetime(analysis_end_date)]
df_future = df[df.index >= pd.to_datetime(analysis_end_date)]
os.makedirs(os.path.join(BASE, "pred"), exist_ok=True)
BATCH_SIZE = 8
EPOCHS = 50

def createXY(dataset, n_past):
    dataX = []
    dataY = []
    data = dataset.values
    for i in range(len(data) - n_past):
        dataX.append(data[i:i + n_past, :])
        dataY.append(data[i + n_past, :])
    return np.array(dataX), np.array(dataY)


def forecast_future(model, input_seq, n_future, target_scaler):
    predictions = []
    original_input = input_seq[-1, np.newaxis, :, :]
    next_input = original_input
    for _ in range(n_future):
        pred_scaled = model.predict(next_input, verbose=0)
        predictions.append(pred_scaled)
        temp_next_input = np.zeros_like(original_input[-1])
        temp_next_input[:-1] = next_input[-1][1:]
        temp_next_input[-1] = pred_scaled
        next_input = temp_next_input[np.newaxis, :, :]
    p = np.array(predictions)[:, 0, :]
    return target_scaler.inverse_transform(p)[:, -1]


def plot_results(time, m, ax, original, pred):
    ax.plot(time, original, color='red')
    ax.plot(time, pred, color='blue')
    ax.set_title(m)
    ax.set_xlabel('Data')
    ax.set_ylabel('Uždarymo kaina')
    ax.grid()


def prepare_gru(input_shape, output_features):
    grid_model = Sequential()
    grid_model.add(GRU(150, return_sequences=True, input_shape=input_shape))
    grid_model.add(GRU(200, return_sequences=True))
    grid_model.add(GRU(100))
    grid_model.add(Dropout(0.1))
    grid_model.add(Dense(output_features))
    grid_model.compile(loss='mse', optimizer='adam')
    return grid_model


def prepare_lstm(input_shape, output_features):
    grid_model = Sequential()
    grid_model.add(LSTM(150, return_sequences=True, input_shape=input_shape))
    grid_model.add(LSTM(200, return_sequences=True))
    grid_model.add(LSTM(100))
    grid_model.add(Dropout(0.1))
    grid_model.add(Dense(output_features))
    grid_model.compile(loss='mse', optimizer='adam')
    return grid_model


def run(diff: bool, model: str, test_opt: bool):
    global df_all, df_future
    print(f"Options:\n\tdiff -> {diff}\tmodel -> {model}\ttest_opt -> {test_opt}")
    fig, axs = plt.subplots(3, 2, figsize=(18, 10))
    train_split = round(len(df_all) * 0.80)
    n_past = 30
    future = len(df_future)
    for m, ax in tqdm(zip(metals, axs.flatten())):
        selected = metal_pairs[m] + [m] if diff else tickers + [m]
        combined = df_all[selected]
        X, Y = createXY(combined, n_past)
        if test_opt:
            X_train, X_test = (X[:train_split, :, :], X[train_split:, :, :])
            Y_train, Y_test = (Y[:train_split], Y[train_split:])
        else:
            X_train, Y_train = X, Y
            X_test, Y_test = createXY(df_future[selected], n_past)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        Y_train_scaled = target_scaler.fit_transform(Y_train.reshape(-1, Y_train.shape[1]))

        grid_model = prepare_lstm((n_past, len(combined.columns)), len(combined.columns)) \
            if model.lower() == 'lstm' else prepare_gru((n_past, len(combined.columns)), len(combined.columns))
        grid_model.fit(X_train_scaled, Y_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE)
        if test_opt:
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
            Y_test_scaled = target_scaler.transform(Y_test.reshape(-1, Y_test.shape[1]))
            prediction = grid_model.predict(X_test_scaled)
            pred = target_scaler.inverse_transform(prediction)[:, -1]
            original = target_scaler.inverse_transform(Y_test_scaled)[:, -1]
            plot_results(combined.index[(n_past + train_split):], m, ax, original, pred)
        else:
            pred = forecast_future(grid_model, X_train_scaled, future, target_scaler)
            original = df_future[m].values
            plot_results(df_future.index, m, ax, original, pred)
        print(f"Metalas {m} -> RMSE: {np.sqrt(np.sum((pred - original) ** 2))}")

    fig.suptitle(f"{model.upper()} prognozės testiniai imčiai" if test_opt else f"{model.upper()} ateities prognozės")

    handle1, = ax.plot([], [], color='red', label='Tikra uždarymo kaina')  # Empty plot for legend
    handle2, = ax.plot([], [], color='blue' if test_opt else 'green',
                       label='Prognozuota uždarymo kaina')  # Empty plot for legend
    handles = [handle1, handle2]
    labels = ['Tikra uždarymo kaina', 'Prognozuota uždarymo kaina']
    fig.legend(handles, labels, loc='upper right', ncol=4)

    fig.tight_layout()
    save_path = os.path.join(BASE, "pred", f"{model.lower()}_test.png") if test_opt else os.path.join(BASE, "pred",
                                                                                                      f"{model.lower()}_forecast.png")
    plt.savefig(save_path)
    plt.show()


difs = [True, False]
models = ['lstm', 'gru']
test = [True, False]

for opts in product(difs, models, test):
    run(*opts)
