import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from constants import tickers, metals, metal_pairs
from random import seed
import warnings
warnings.filterwarnings('ignore')
seed(42)

df = pd.read_csv('all.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df_all = df.iloc[::-1]
df_stocks = df_all[tickers]
df_metals = df_all[metals]

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(np.array(dataset.iloc[i - n_past:i, 0:dataset.shape[1]]))
        dataY.append(dataset.iloc[i, -1])
    return np.array(dataX), np.array(dataY)

def plot_results(time, m, ax, original, pred):
    ax.plot(time, original, color='red')
    ax.plot(time, pred, color='blue')
    ax.set_title(m)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.grid()

fig, axs = plt.subplots(3,2, figsize=(18, 10))
train_split=round(len(df)*0.80)
n_past=30
for m, ax in tqdm(zip(metals, axs.flatten())):
    combined = df_stocks[metal_pairs[m]]
    combined[m] = df_metals[m]
    X, Y = createXY(combined, n_past)
    X_train, X_test = X[:train_split, :, :], X[train_split:, :, :]
    Y_train, Y_test = Y[:train_split], Y[train_split:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_train_scaled = target_scaler.fit_transform(Y_train.reshape(-1, 1))
    Y_test_scaled = target_scaler.transform(Y_test.reshape(-1, 1))

    grid_model = Sequential()
    grid_model.add(LSTM(150, return_sequences=True, input_shape=(n_past, len(combined.columns))))
    grid_model.add(LSTM(200, return_sequences=True))
    grid_model.add(LSTM(100))
    grid_model.add(Dropout(0.1))
    grid_model.add(Dense(1))
    grid_model.compile(loss='mse', optimizer='adam')

    grid_model.fit(X_train_scaled, Y_train_scaled, epochs=50, batch_size=16)
    prediction = grid_model.predict(X_test_scaled)
    pred = target_scaler.inverse_transform(prediction)
    original = target_scaler.inverse_transform(Y_test_scaled)
    plot_results(combined.index[(n_past+train_split):], m, ax, original, pred)
    grid_model.save(f'lstm_model_{m}.h5')
    print(f"Metal {m} -> RMSE: {np.sqrt(np.sum((pred-original)**2))}")
fig.suptitle("LSTM predictions")

handle1, = ax.plot([], [], color='red', label='Real Stock Price')  # Empty plot for legend
handle2, = ax.plot([], [], color='blue', label='Predicted Stock Price')  # Empty plot for legend
handles=[handle1, handle2]
labels=['Real Stock Price', 'Predicted Stock Price']
fig.legend(handles, labels, loc='upper right', ncol=4)

fig.tight_layout()
plt.savefig("lstm_rez2.png")
plt.show()
