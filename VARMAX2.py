import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from constants import tickers, metals, ts_order
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
    metal_market_data = df[indexes_cols + [f'{metal_market}']]

    # Fit the VAR model
    model = VAR(metal_market_data)
    results = model.fit(maxlags=15, ic='aic')  # You can tune the lags parameter

    # Print the summary of VAR results
    print(f"VAR Results for Metal Market {metal_market}")
    #print(results.summary())

    # Example forecasting for the next 10 steps
    forecast = results.forecast(metal_market_data.values[-15:], steps=10)
    print(f"Forecast for Metal Market {metal_market} (next 10 steps):")
    #print(forecast)

    # VARMAX Model - Using indexes as exogenous variables
    exog = metal_market_data[[i for i in indexes_cols if i != '^GSPC']]  # Exogenous variables (indexes)
    endog = metal_market_data[[f'{metal_market}', '^GSPC']]  # Endogenous variable (metal market)

    # Fit the VARMAX model
    varmax_model = VARMAX(endog, exog=exog, order=(ts_order[metal_market][0] if ts_order[metal_market][0]>0 else 1, ts_order[metal_market][-1] if ts_order[metal_market][-1]>0 else 1))  # You can tune the (p, q) order
    varmax_results = varmax_model.fit(disp=False)

    # Print the summary of VARMAX results
    print(f"VARMAX Results for Metal Market {metal_market}")
    print(varmax_results.summary())

    # Example forecasting with exogenous variables for the next 10 steps
    forecast_varmax = varmax_results.forecast(steps=10, exog=exog.tail(10))
    print(f"VARMAX Forecast for Metal Market {metal_market} (next 10 steps):")
    print(forecast_varmax)
    with open(f'var_{metal_market}.pkl', 'wb') as f:
        pickle.dump({'var': model, 'var_rez': results,
                          'varmax': varmax_model, 'varmax_results': varmax_results}, f)
