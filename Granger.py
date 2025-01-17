import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

from constants import tickers, metals, BASE, war_date, analysis_end_date

warnings.simplefilter('ignore')
os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, 'granger'), exist_ok=True)

df = pd.read_csv(f'{BASE}.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df = df[df.index < pd.to_datetime(analysis_end_date)]

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

    # Granger ------------------
    ml = 4
    gc_dfs = {m: [grangercausalitytests(df_all[[t, m]], maxlag=ml, verbose=False) for t in tickers] for m in metals}
    p_value_matrix = {lag: pd.DataFrame(index=tickers, columns=metals) for lag in range(1, ml + 1)}
    for m in metals:
        for t in tickers:
            for lag in range(1, ml + 1):
                test_results = gc_dfs[m][tickers.index(t)]
                p_value = test_results[lag][0]['ssr_chi2test'][1]
                p_value_matrix[lag].at[t, m] = p_value
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # Adjust as needed for more lags
    axes = axes.flatten()

    for lag in range(1, ml + 1):
        ax = axes[lag - 1]
        sns.heatmap(p_value_matrix[lag].astype(float), annot=True, cmap='coolwarm', fmt=".2f",
                    cbar_kws={'label': 'p-reikšmė'}, ax=ax)
        ax.set_title(f"Grangerio priežastingumo testo p-reikšmės (delsa {lag})")

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'granger', f"{KARAS}_granger_causality_lags.png"))
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # Adjust for the number of lags (e.g., 4 lags here)
    axes = axes.flatten()
    p_threshold = 0.05  # Significance threshold for causality

    # Loop through each lag to create the network graph
    for lag in range(1, ml + 1):
        ax = axes[lag - 1]
        G = nx.DiGraph()  # Create a directed graph for each lag

        # Build the graph with significant edges
        for m in metals:
            for t in tickers:
                p_value = p_value_matrix[lag].at[t, m]
                if p_value < p_threshold:
                    # If causal, add an edge (from ticker to metal)
                    G.add_edge(t, m)

        # Draw the graph on the current subplot axis
        pos = nx.spring_layout(G, seed=42)  # Layout for the graph
        labels = nx.get_edge_attributes(G, 'label')

        # Plot the network on the current subplot
        nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=12, font_weight='bold',
                edge_color='blue', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10, font_color='red', ax=ax)

        # Set title for each lag's network plot
        ax.set_title(f"Grangerio priežastingumo testo rezultatų tinklas (delsa {lag})")

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, 'granger', f"{KARAS}_granger_causality_network_lags.png"))
    plt.show()
