from joblib import Parallel, delayed
import os
import sys
import warnings
import pickle
from random import seed

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX

from constants import tickers, metals, ts_order, metal_pairs, analysis_end_date

warnings.filterwarnings('ignore')
seed(42)

if len(sys.argv) > 1:
    BASE = str(sys.argv[1])
    print(f"Received constant: {BASE}")
else:
    from constants import BASE

# Load your dataset
df = pd.read_csv(f'{BASE}.csv', index_col=0)  # Replace with your actual file path
df.index = pd.to_datetime(df.index)
df_all = df[df.index < pd.to_datetime(analysis_end_date)]
df_future = df[df.index >= pd.to_datetime(analysis_end_date)]
os.makedirs(os.path.join(BASE, "varmax"), exist_ok=True)


def fit_var_models_parallel(metal_market, df_all, metal_pairs, tickers, ts_order, BASE):
    selected = tickers + [metal_market]
    selected_pairs = metal_pairs[metal_market] + [metal_market]
    metal_market_data_whole = df_all[selected]
    metal_market_data_part = df_all[selected_pairs]
    p, s, q = ts_order[metal_market]
    if p == 0:
        print(f"Warning: p is zero for {metal_market}, setting to default 1")
        p = 1
    if q == 0:
        print(f"Warning: q is zero for {metal_market}, setting to default 1")
        q = 1
    print(f"Fitting VAR for whole Metal Market {metal_market}")
    model = VARMAX(metal_market_data_whole, order=(p, q), enforce_stationarity=True)
    results = model.fit(disp=False)  # You can tune the lags parameter
    print(f"VAR Results for whole Metal Market {metal_market}")
    print(f"Fitting VAR for part Metal Market {metal_market}")
    model_part = VARMAX(metal_market_data_part, order=(p, q), enforce_stationarity=True)
    results_part = model_part.fit(disp=False)  # You can tune the lags parameter
    print(f"VAR Results for part Metal Market {metal_market}")
    print(f"Fitting VARMAX for Metal Market {metal_market}")
    exog = df_all[list(set(tickers) - set(selected_pairs))]  # Exogenous variables (indexes)
    endog = metal_market_data_part  # Endogenous variable (metal market)
    varmax_model = VARMAX(endog, exog=exog, order=(p, q), enforce_stationarity=True)  # You can tune the (p, q) order
    varmax_results = varmax_model.fit(disp=False)
    print(f"VARMAX Results for Metal Market {metal_market}")
    try:
        with open(os.path.join(BASE, 'varmax', f'var_{metal_market}.pkl'), 'wb') as f:
            pickle.dump({'var': model, 'var_rez': results,
                         'var_part': model_part, 'var_rez_part': results_part,
                         'varmax': varmax_model, 'varmax_results': varmax_results}, f)
        print(f"Successfully saved results to {os.path.join(BASE, 'varmax', f'var_{metal_market}.pkl')}")
    except Exception as e:
        print(f"Error saving pickle: {e}")


def parallelize_varmax(df_all, metal_pairs, tickers, ts_order, BASE, metals):
    Parallel(n_jobs=-1, verbose=10)(delayed(fit_var_models_parallel)(metal_market, df_all, metal_pairs, tickers, ts_order, BASE) for metal_market in metals)


def run():
    global df_all, df_future
    parallelize_varmax(df_all, metal_pairs, tickers, ts_order, BASE, metals)

    rez = {}
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    for metal_market, ax in zip(metals, axs.flatten()):
        with open(os.path.join(BASE, 'varmax', f'var_{metal_market}.pkl'), 'rb') as f:
            r = pickle.load(f)
            rez[metal_market] = {'var': r['var'], 'var_rez': r['var_rez'],
                                 'var_part': r['model_part'], 'var_rez_part': r['results_part'],
                                 'varmax': r['varmax'], 'varmax_results': r['varmax_results']}

        forecast_steps = len(df_future)
        forecast_v = rez[metal_market]['var_rez'].get_forecast(steps=forecast_steps)
        forecast_v_part = rez[metal_market]['var_rez_part'].get_forecast(steps=forecast_steps)
        forecast = rez[metal_market]['varmax_results'].get_forecast(steps=forecast_steps,
                                                                    exog=df_future.drop(columns=[metal_market]))

        # Get the predicted values
        predicted_values = forecast.predicted_mean[metal_market]
        predicted_values_v = forecast_v.predicted_mean[metal_market]
        predicted_values_v_part = forecast_v_part.predicted_mean[metal_market]
        rmses = {'varma': np.sqrt(np.mean((df_future[metal_market].values - predicted_values_v) ** 2)),
                 'varma_part': np.sqrt(np.mean((df_future[metal_market].values - predicted_values_v_part) ** 2)),
                 'varmax': np.sqrt(np.mean((df_future[metal_market].values - predicted_values) ** 2))
                 }
        xtime = mdates.date2num(df_future.index.tolist())
        predtime = mdates.date2num(
            pd.Index(
                pd.date_range(df_future.index[-1], periods=forecast_steps + 1, freq='D')[1:].strftime(
                    '%Y-%m-%d')).tolist())
        ax.plot(xtime, df_future[metal_market], color='blue')
        ax.plot(predtime, predicted_values, color='red', linestyle='--')
        ax.plot(predtime, predicted_values_v, color='green', linestyle='--')
        ax.plot(predtime, predicted_values_v_part, color='orange', linestyle='--')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_title(
            f"{metal_market}. RMSE:\n VARMA -> {rmses['varma']}\n VARMA su poromis -> {rmses['varma_part']}\n VARMAX -> {rmses['varmax']}")
        ax.set_xlabel('Data')
        ax.set_ylabel('Uždarymo kaina')
        ax.grid(True)

    handle1, = ax.plot([], [], color='blue', label='Tikra kaina')  # Empty plot for legend
    handle2, = ax.plot([], [], color='red', linestyle='--', label='Prognozuota kaina\n(VARMA)')  # Empty plot for legend
    handle3, = ax.plot([], [], color='orange', linestyle='--',
                       label='Prognozuota kaina\n(VARMA su poromis)')  # Empty plot for legend
    handle4, = ax.plot([], [], color='green', linestyle='--',
                       label='Prognozuota kaina\n(VARMAX)')  # Empty plot for legend
    handles = [handle1, handle2, handle3, handle4]
    labels = ['Tikra kaina', 'Prognozuota kaina\n(VARMA)', 'Prognozuota kaina\n(VARMA su poromis)',
              'Prognozuota kaina\n(VARMAX)']
    fig.legend(handles, labels, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'varmax', 'varma_varmax.png'))
    plt.show()


orgs = ['all', 'selected']
for opt in orgs:
    run()
