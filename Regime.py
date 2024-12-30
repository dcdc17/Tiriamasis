import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from constants import tickers, metals, BASE

os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, 'regime'), exist_ok=True)

df_all = pd.read_csv(f'{BASE}.csv', index_col=0)
df_all.index = pd.to_datetime(df_all.index)
df_stocks = df_all[tickers]
df_metals = df_all[metals]

for col in df_all.columns:
    df_all[f'{col}_returns'] = np.log(df_all[col] / df_all[col].shift(1))  # Lognormuotos grąžos

df_all = df_all.dropna()

fig, ax = plt.subplots(4, 4, figsize=(16, 16))
for a, col in zip(ax.flatten(), tickers + metals):
    a.plot(df_all.index, df_all[f'{col}_returns'], label='Lognormuotos grąžos')
    a.set_title(f'{col}')
    a.set_xlabel('Data')
    a.set_ylabel('Lognormuotos grąžos')
    a.xaxis.set_major_locator(mdates.YearLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.suptitle('Lognormuotos grąžos')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'regime', 'log_returns.png'))
plt.show()

results = {}
for col in [i for i in df_all.columns if i.endswith('_returns')]:
    model = MarkovRegression(df_all[col], k_regimes=2, switching_variance=True)
    results[col.split('_')[0]] = model.fit()

fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for ax, col in zip(axs.flatten(), [i for i in df_all.columns if i.endswith('_returns')]):
    ax.plot(df_all.index, df_all[col], label='Lognormuotos grąžos', color='black')
    ax.fill_between(df_all.index, 0, results[col.split('_')[0]].smoothed_marginal_probabilities[0], color='blue',
                    alpha=0.3, label='Mažo nepastovumo režimas')
    ax.fill_between(df_all.index, 0, results[col.split('_')[0]].smoothed_marginal_probabilities[1], color='red',
                    alpha=0.3, label='Didelio nepastovumo režimas')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_title(f"{col.split('_')[0]}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Lognormuotos grąžos')
    ax.legend()
plt.suptitle("Grąžos su režimų tikimybėmis")
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'regime', "regimes.png"))
plt.show()

low_volatility_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Mažo nepastovumo režimas')
high_volatility_line = plt.Line2D([0], [0], color='red', linewidth=2, label='Didelio nepastovumo režimas')
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for ax, col in zip(axs.flatten(), [i for i in df_all.columns if i.endswith('_returns')]):
    returns = df_all[col]
    probabilities = results[col.split('_')[0]].smoothed_marginal_probabilities
    regimes = np.argmax(probabilities, axis=1)
    for i in range(len(returns) - 1):
        ax.plot(
            df_all.index[i:i + 2], returns.iloc[i:i + 2],
            color='blue' if regimes[i] == 0 else 'red', linewidth=1
        )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_title(f"{col.split('_')[0]}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Lognormuotos grąžos')

fig.legend(
    handles=[low_volatility_line, high_volatility_line],
    loc='upper right',
    bbox_to_anchor=(0.9, 0.95),  # Adjust as needed
    fontsize='medium'
)
plt.suptitle("Grąžos su pažymėtais režimais")
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'regime', "regimes_line_color.png"))
plt.show()
