import os
import sys
import warnings
from itertools import product
from random import seed

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from constants import tickers, metals, metal_pairs, analysis_end_date

warnings.filterwarnings('ignore')
seed(42)

if len(sys.argv) > 1:
    BASE = sys.argv[1]
    print(f"Received constant: {BASE}")
else:
    from constants import BASE

df = pd.read_csv(f'{BASE}.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df_all = df[df.index < pd.to_datetime(analysis_end_date)]
df_future = df[df.index >= pd.to_datetime(analysis_end_date)]
os.makedirs(os.path.join(BASE, "pred"), exist_ok=True)
BATCH_SIZE = 256
EPOCHS = 1


def rmse_metric(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    data = dataset.values
    if data.ndim == 1:
        data = data.reshape(-1, 1)
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


def plot_results(time, m, ax, original, pred, rmse, future=False):
    ax.plot(time, original, color='red')
    ax.plot(time, pred, color='blue' if not future else 'green')
    ax.set_title(f"{m}. RMSE = {rmse:.4f}")
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
    grid_model.compile(loss='mse', optimizer='adam', metrics=[rmse_metric])
    return grid_model


def prepare_lstm(input_shape, output_features):
    grid_model = Sequential()
    grid_model.add(LSTM(150, return_sequences=True, input_shape=input_shape))
    grid_model.add(LSTM(200, return_sequences=True))
    grid_model.add(LSTM(100))
    grid_model.add(Dropout(0.1))
    grid_model.add(Dense(output_features))
    grid_model.compile(loss='mse', optimizer='adam', metrics=[rmse_metric])
    return grid_model


def run(sel: str, model: str, test_opt: str):
    global df_all, df_future
    print(f"Options:\n\tsel -> {sel}\tmodel -> {model}\ttest_opt -> {test_opt}")
    fig, axs = plt.subplots(3, 2, figsize=(18, 10))
    n_past = 30
    train_split = round((len(df_all)-n_past) * 0.80)
    future = len(df_future)
    for m, ax in tqdm(zip(metals, axs.flatten())):
        selected = tickers + [m] if sel == 'all' else metal_pairs[m] + [m] if sel == 'selected' else m
        combined = df_all[selected]
        if isinstance(combined, pd.Series):
            combined = combined.to_frame()
        X, Y = createXY(combined, n_past)
        if test_opt == 'test':
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
        grid_model.fit(X_train_scaled, Y_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        if test_opt == 'test':
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
            Y_test_scaled = target_scaler.transform(Y_test.reshape(-1, Y_test.shape[1]))
            prediction = grid_model.predict(X_test_scaled, verbose=0)
            pred = target_scaler.inverse_transform(prediction)[:, -1]
            original = target_scaler.inverse_transform(Y_test_scaled)[:, -1]
            rmse = np.sqrt(np.mean((pred - original) ** 2))
            plot_results(combined.index[(n_past + train_split):], m, ax, original, pred, rmse)
        else:
            pred = forecast_future(grid_model, X_train_scaled, future, target_scaler)
            original = df_future[m].values
            rmse = np.sqrt(np.mean((pred - original) ** 2))
            plot_results(df_future.index, m, ax, original, pred, rmse, future=True)
        print(f"Metalas {m} -> RMSE: {rmse}")

    fig.suptitle(
        f"{model.upper()} prognozės testiniai imčiai" if test_opt == 'test' else f"{model.upper()} ateities prognozės")

    handle1, = ax.plot([], [], color='red', label='Tikra uždarymo kaina')  # Empty plot for legend
    handle2, = ax.plot([], [], color='blue' if test_opt else 'green',
                       label='Prognozuota uždarymo kaina')  # Empty plot for legend
    handles = [handle1, handle2]
    labels = ['Tikra uždarymo kaina', 'Prognozuota uždarymo kaina']
    fig.legend(handles, labels, loc='upper right', ncol=4)

    fig.tight_layout()
    plt.savefig(os.path.join(BASE, "pred", f"{model.lower()}_{sel}_{test_opt}.png"))
    plt.show()


orgs = ['all', 'selected', 'solo']
models = ['lstm', 'gru']
test = ['test', 'future']

for opts in product(orgs, models, test):
    run(*opts)
