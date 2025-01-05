import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from constants import tickers, metals, war_date, BASE

os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, 'AB'), exist_ok=True)

df = pd.read_csv(f'{BASE}.csv', index_col=0)
scaler = MinMaxScaler(feature_range=(1, 2))
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
df.index = pd.to_datetime(df.index)


# Function to calculate Alpha and Beta for a stock
def calculate_alpha_beta(market, benchmark):
    x = benchmark.pct_change().dropna()
    x = np.vstack([x, np.ones(len(x))]).T
    y = market.pct_change().dropna()
    alpha, beta = np.linalg.lstsq(x, y, rcond=None)[0]

    # Grade for Alpha
    if alpha > 0.05:
        alpha_grade = "Puiku"
    elif 0.01 <= alpha <= 0.05:
        alpha_grade = "Gerai"
    else:
        alpha_grade = "Žemiau vidurkio"

    # Grade for Beta
    if beta < 0.8:
        beta_grade = "Maža rizika (gynybinė)"
    elif 0.8 <= beta <= 1.2:
        beta_grade = "Vidutinė rizika"
    else:
        beta_grade = "Didelė rizika (agresyvi)"

    return alpha, beta, alpha_grade, beta_grade


WAR = [None, True, False]
for war in WAR:
    print(f"Analyzing WAR = {war}")
    KARAS = 'full'
    df_all = df
    if war is not None:
        df_all = df_all[df_all.index >= pd.to_datetime(war_date) if war else df_all.index < pd.to_datetime(war_date)]
        KARAS = 'po' if war else 'pries'
    df_stocks = df_all[tickers]
    df_metals = df_all[metals]

    results = {m: {} for m in metals}

    for m in metals:
        for t in tickers:
            alpha, beta, alpha_grade, beta_grade = calculate_alpha_beta(df_all[m], df_all[t])
            results[m][t] = {
                'Alpha': alpha,
                'Beta': beta,
                'Alpha Grade': alpha_grade,
                'Beta Grade': beta_grade
            }

    # Loop through each metal to create line charts
    fig, axs = plt.subplots(3,2, figsize=(16,12))
    for ax, (metal, tickers_data) in zip(axs.flatten(), results.items()):
        tickers = list(tickers_data.keys())
        alphas = [tickers_data[t]['Alpha'] for t in tickers]
        betas = [tickers_data[t]['Beta'] for t in tickers]

        # Primary y-axis for Alpha
        ax.plot(tickers, alphas, label='Alpha', marker='o', color='blue')
        ax.set_ylabel('Alpha', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Secondary y-axis for Beta
        ax2 = ax.twinx()
        ax2.plot(tickers, betas, label='Beta', marker='s', color='red')
        ax2.set_ylabel('Beta', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Customizations
        ax.set_title(metal)
        ax.set_xlabel('Tickers')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Reference line
        ax.grid(True)
        ax.tick_params(axis='x', rotation=60)  # Rotate x-axis tick labels

    # Add a shared title for the entire figure
    plt.suptitle('Alpha ir Beta reikšmės kiekvienam metalo indeksui', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'AB', f"{KARAS}_alpha_beta_dual_axes.png"))
    plt.show()

    table_data = []

    for metal, tickers_data in results.items():
        for ticker, grades in tickers_data.items():
            table_data.append({
                "Metal": metal,
                "Ticker": ticker,
                "Alpha Grade": grades['Alpha Grade'],
                "Beta Grade": grades['Beta Grade']
            })

    # Convert list to a DataFrame
    grades_table = pd.DataFrame(table_data)

    # Display the table
    print(grades_table)
    grades_table.to_excel(os.path.join(BASE, 'AB', f"{KARAS}_alpha_beta_grades.xlsx"), index=False)