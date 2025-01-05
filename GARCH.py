import os
import sys
import warnings
from random import seed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
from tqdm import tqdm

from constants import metals, analysis_end_date

warnings.filterwarnings('ignore')
seed(42)

if len(sys.argv) > 1:
    BASE = sys.argv[1]
    print(f"Received constant: {BASE}")
else:
    from constants import BASE

os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, 'GARCH'), exist_ok=True)

GARCH_PARAMS = {"p": 1, "q": 1, "mean": 'zero', "vol": 'GARCH', "dist": 'normal'}

# Load and scale data
df = pd.read_csv(f'{BASE}.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df_all = df[df.index < pd.to_datetime(analysis_end_date)]
df_future = df[df.index >= pd.to_datetime(analysis_end_date)]


def analyze_pacf():
    fig, axs = plt.subplots(3, 2, figsize=(16, 9))
    # Loop over metals to generate PACF subplots
    for m, ax in tqdm(zip(metals, axs.flatten())):
        plot_pacf(df[m], ax=ax)
        ax.set_title(m)
    plt.suptitle('Metalų rinkų PACF grafikai')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'GARCH', 'pacf.png'))
    plt.show()


def evaluate_model(data, future_data, ax, ax2, forecast_period=60):
    log_returns = 100 * data.pct_change()
    log_returns.dropna(inplace=True)
    log_returns_future = 100 * future_data.pct_change()
    log_returns_future.dropna(inplace=True)

    # Fit GARCH model
    am = arch_model(log_returns, **GARCH_PARAMS)
    res = am.fit(disp='off')
    future_m = res.forecast(horizon=forecast_period)
    future = np.sqrt(future_m.variance.values[-1, :])
    # Calculate AIC and BIC
    aic = res.aic
    bic = res.bic

    # Perform backtesting
    residuals = log_returns - res.conditional_volatility
    res_t = residuals / res.conditional_volatility
    backtest = (res_t ** 2).sum()

    # Out-of-sample testing
    data_length = len(log_returns)
    train_size = int(0.8 * data_length)
    test_size = data_length - train_size
    train_data = log_returns[:train_size]
    test_data = log_returns[train_size:]
    res_oos = am.fit(last_obs=train_data[-1], disp='off')
    forecast = res_oos.forecast(horizon=len(test_data))
    # Calculate out-of-sample forecast error
    forecast_vol = forecast.residual_variance.values[-1, :]
    error = test_data - forecast_vol

    # PLOT 1: Predictability on test data: whole and rolling
    rolling_predictions = []
    for i in range(test_size):
        train = log_returns[:-(test_size - i)]
        model = arch_model(train, **GARCH_PARAMS)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))
    rolling_predictions = pd.Series(rolling_predictions, index=data.index[-test_size:])

    ax.plot(test_data, color='blue', label='Testinės imties grąžos')
    ax.plot(test_data.index, np.sqrt(forecast.variance.values[-1, :]),
            color='orange', label='Grąžos ilgalaikė prognozė')
    ax.plot(test_data.index, rolling_predictions, color='red',
            label='Grąžų kasdienė prognozė')
    ax.set_xlabel('Data')
    ax.set_ylabel('Grąža')
    ax.set_title(data.name)
    ax.grid()

    # PLOT 2: Forecasting future volatility
    # Plot actual log returns and forecasted data
    ax2.plot(log_returns_future, color='blue', label='Tikrosios grąžos')
    # Plot forecasted log returns
    ax2.plot(future_data.index, future, color='green', linestyle=':',
             label='Prognozuojamos grąžos')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Grąža')
    ax2.set_title(data.name)
    ax2.grid()

    return aic, bic, backtest, error


analyze_pacf()

# Evaluate the GARCH model using log returns data
rez = {}
fig, axs = plt.subplots(3, 2, figsize=(24, 9))
fig2, axs2 = plt.subplots(3, 2, figsize=(24, 9))

with open(os.path.join(BASE, 'GARCH', "rez.txt"), "w") as file:
    file.truncate(0)
# Loop over metals to generate subplots
for m, ax, ax2 in tqdm(zip(metals, axs.flatten(), axs2.flatten())):
    aic, bic, backtest, forecast_error = evaluate_model(df_all[m], df_future[m], ax, ax2,
                                                        forecast_period=len(df_future))
    rez[m] = [aic, bic, backtest, forecast_error]
    result_txt = f'Metalas: {m}\tAIC: {aic}\tBIC: {bic}\tAtgalinio testavimo rezultatas: {backtest}'
    with open(os.path.join(BASE, 'GARCH', "rez.txt"), "a") as file:
        file.write(result_txt + "\n")

handle0_1, = ax.plot([], [], color='blue', label='Testinės imties grąžos')  # Empty plot for legend
handle0_2, = ax.plot([], [], color='orange', label='Grąžos ilgalaikė prognozė)')  # Empty plot for legend
handle0_3, = ax.plot([], [], color='red', label='Grąžos kasdienė prognozė)')  # Empty plot for legend

handles = [handle0_1, handle0_2, handle0_3]
labels = ['Testinės imties grąžos', 'Grąžos ilgalaikė prognozė',
          'Grąžos kasdienė prognozė']
fig.suptitle("GARCH prognozių tikrinimas")
fig.legend(handles, labels, loc='upper right', ncol=4)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'GARCH', "garch_forecast.png"))
fig.show()

handle1, = ax2.plot([], [], color='blue', label='Tikrosios grąžos')  # Empty plot for legend
handle2, = ax2.plot([], [], color='green', linestyle=':', label='Prognozuojamos grąžos')  # Empty plot for legend

handles = [handle1, handle2]
labels = ['Tikrosios grąžos', 'Prognozuojamos grąžos']
fig2.suptitle("GARCH prognozavimas")
fig2.legend(handles, labels, loc='upper right', ncol=4)
fig2.tight_layout()
fig2.savefig(os.path.join(BASE, 'GARCH', "garch_forecast.png"))
fig2.show()
