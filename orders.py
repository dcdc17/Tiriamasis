import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pmdarima as pm
from constants import tickers, metals

warnings.simplefilter('ignore')

df = pd.read_csv('all.csv', index_col=0)
df_all = df
df_all = df_all.iloc[::-1]
df_all.index = pd.to_datetime(df_all.index)
df_stocks = df_all[tickers]
df_metals = df_all[metals]

for col in df_all.columns:
    model = pm.auto_arima(df_all[col], seasonal=True, stepwise=True, trace=False, n_jobs=-1)
    print(f"Best ARIMA model parameters for {col}: {model.order}")
