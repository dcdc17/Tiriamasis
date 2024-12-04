import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from arch import arch_model
import matplotlib.dates as mdates
from constants import tickers, metals, metal_pairs
ts_order = {'^GSPC': (2, 1, 2),
            '^DJI': (3, 1, 3),
            '^IXIC': (2, 1, 2),
            '^FTSE': (0, 1, 0),
            '^GDAXI': (2, 1, 2),
            '^FCHI': (0, 1, 0),
            '^N100': (0, 1, 0),
            'EURUSD=X': (0, 1, 0),
            '^HSI': (0, 1, 0),
            '^DXS': (2, 1, 4),
            'GD=F': (2, 1, 1),
            'EURRUB=X': (1, 1, 3),
            'GC=F': (3, 1, 3),
            'SI=F': (2, 1, 0),
            'PL=F': (0, 1, 0),
            'PA=F': (2, 1, 1),
            'HG=F': (0, 1, 0),
            'ALI=F': (0, 1, 1)}
warnings.simplefilter('ignore')

# Load and scale data
df = pd.read_csv('all.csv', index_col=0)
scaler = MinMaxScaler(feature_range=(1, 2))
df_all = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
df_all.index = pd.to_datetime(df_all.index)
df_all = df.iloc[::-1]
df_stocks = df_all[tickers]
df_metals = df_all[metals]


def evaluate_model(ind, data, ax, forecast_period=30):
    log_returns = np.log(data / data.shift(1))
    log_returns = log_returns.dropna()

    # Fit GARCH model
    am = arch_model(log_returns, mean='Zero', vol='GARCH', p=ts_order[ind][0] if ts_order[ind][0]>0 else 1, q=ts_order[ind][-1] if ts_order[ind][-1]>0 else 1, rescale=True)
    res = am.fit(disp='final')
    future_m = res.forecast(start=log_returns[-1], horizon=forecast_period)
    future = future_m.residual_variance.iloc[-1, :]
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
    train_data = log_returns[:train_size]
    test_data = log_returns[train_size:]

    res_oos = am.fit(last_obs=train_data[-1], disp='final')
    forecast = res_oos.forecast(start=train_data[-1], horizon=len(test_data))

    # Calculate out-of-sample forecast error
    forecast_vol = forecast.residual_variance.iloc[-1, :]
    error = (test_data - forecast_vol).dropna()

    # Plot actual log returns and forecasted data
    ax.plot(mdates.date2num(test_data.index.tolist()), test_data, color='blue', label='Actual Log Returns')
    ax.set_ylabel('Log Return')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='y', labelcolor='blue')

    ax2 = ax.twinx()
    ax2.plot(mdates.date2num(test_data.index.tolist()), np.sqrt(forecast_vol), color='red', linestyle='--', label='Forecasted Volatility')
    ax2.set_ylabel('Volatility', color='red')
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='y', labelcolor='red')

    # Plot forecasted log returns
    future_dates = pd.date_range(test_data.index[-1], periods=forecast_period+1, freq='D')[1:]
    ax2.plot(mdates.date2num(future_dates.tolist()), np.sqrt(future), color='green', linestyle=':', label='Forecasted Log Returns')
    ax2.set_title(data.name)
    ax2.set_xlabel('Date')
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    return aic, bic, backtest, error


# Evaluate the GARCH model using log returns data
rez = {}
fig, axs = plt.subplots(3, 2, figsize=(16, 9))

# Loop over metals to generate subplots
for m, ax in tqdm(zip(metals, axs.flatten())):
    aic, bic, backtest, forecast_error = evaluate_model(m, df_all[m], ax)
    rez[m] = [aic, bic, backtest, forecast_error]
    print(f'Metal: {m}\tAIC: {aic}\tBIC: {bic}\tBacktesting Result: {backtest}')

handle1, = ax.plot([], [], color='blue', label='Actual Log Returns')  # Empty plot for legend
handle2, = ax.plot([], [], color='red', linestyle='--', label='Forecasted Volatility')  # Empty plot for legend
handle3, = ax.plot([], [], color='green', linestyle=':',
                   label='Future volatility')  # Empty plot for legend
handles = [handle1, handle2, handle3]
labels = ['Actual Log Returns', 'Forecasted Volatility', 'Forecasted Log Returns']

# Set the suptitle and manually add the legend
fig.suptitle("GARCH Predictions with Forecast")
fig.legend(handles, labels, loc='upper right', ncol=4)

# Adjust layout and save figure
fig.tight_layout()  # Adjust layout to avoid overlap with suptitle
plt.savefig("garch_forecast.png")
plt.show()
