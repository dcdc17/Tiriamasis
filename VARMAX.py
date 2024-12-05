import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.varmax import VARMAX
from constants import tickers, metals, ts_order, metal_pairs
import pickle

# Load your dataset
df = pd.read_csv('all.csv', index_col=0)  # Replace with your actual file path
df = df.iloc[::-1]
# Define the column names for indexes and metal markets
indexes_cols = tickers
metal_market_cols = metals
# VAR Model - Loop over each metal market
for metal_market in tqdm(metal_market_cols):
    # Prepare the data for the current metal market
    metal_market_data = df[metal_pairs[metal_market] + [metal_market]]

    # Fit the VAR model
    model = VARMAX(metal_market_data, order=(1, 2))
    results = model.fit(maxlags=15, ic='aic')  # You can tune the lags parameter

    # Print the summary of VAR results
    print(f"VAR Results for Metal Market {metal_market}")
    print(results.summary())


    # VARMAX Model - Using indexes as exogenous variables
    exog = df[[i for i in indexes_cols if i not in metal_pairs[metal_market]]]  # Exogenous variables (indexes)
    endog = metal_market_data  # Endogenous variable (metal market)

    # Fit the VARMAX model
    varmax_model = VARMAX(endog, exog=exog, order=(ts_order[metal_market][0] if ts_order[metal_market][0] > 0 else 1,
                                                   ts_order[metal_market][-1] if ts_order[metal_market][
                                                                                     -1] > 0 else 1))  # You can tune the (p, q) order
    varmax_results = varmax_model.fit(disp=False)

    # Print the summary of VARMAX results
    print(f"VARMAX Results for Metal Market {metal_market}")
    print(varmax_results.summary())

    with open(f'var_{metal_market}.pkl', 'wb') as f:
        pickle.dump({'var': model, 'var_rez': results,
                     'varmax': varmax_model, 'varmax_results': varmax_results}, f)

rez = {}
fig, axs = plt.subplots(2, 3, figsize=(18, 8))
for metal_market, ax in zip(metals, axs.flatten()):
    with open(f'var_{metal_market}.pkl', 'rb') as f:
        r = pickle.load(f)
        rez[metal_market] = {'var': r['var'], 'var_rez': r['var_rez'],
                             'varmax': r['varmax'], 'varmax_results': r['varmax_results']}

    metal_market_data = df[indexes_cols + [f'{metal_market}']]
    exog = metal_market_data[[i for i in indexes_cols if i != '^GSPC']]  # Exogenous variables (indexes)
    endog = metal_market_data[[f'{metal_market}', '^GSPC']]

    forecast_steps = 365
    forecast_v = rez[metal_market]['var_rez'].get_forecast(steps=forecast_steps)
    forecast = rez[metal_market]['varmax_results'].get_forecast(steps=forecast_steps, exog=exog[-forecast_steps:])

    # Get the predicted values
    predicted_values = forecast.predicted_mean[metal_market]
    predicted_values_v = forecast_v.predicted_mean[metal_market]
    xtime = mdates.date2num(df.index.tolist())
    predtime = mdates.date2num(
        pd.Index(pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='D')[1:].strftime('%Y-%m-%d')).tolist())
    ax.plot(xtime, df[metal_market], color='blue')
    ax.plot(predtime, predicted_values, color='red', linestyle='--')
    ax.plot(predtime, predicted_values_v, color='green', linestyle='--')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_title(metal_market)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)

handle1, = ax.plot([], [], color='blue', label='True data')  # Empty plot for legend
handle2, = ax.plot([], [], color='red', linestyle='--', label='Forecasted price (VARMA)')  # Empty plot for legend
handle2, = ax.plot([], [], color='green', linestyle='--', label='Forecasted price (VARMAX)')  # Empty plot for legend
handles = [handle1, handle2]
labels = ['True data', 'Forecasted price (VARMA)', 'Forecasted price (VARMAX)']
plt.tight_layout()
plt.savefig('pred/varma_varmax.png')
plt.show()
