import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from constants import tickers, metals


df_all = pd.read_csv('all.csv', index_col=0)
df_stocks = df_all[tickers]
df_metals = df_all[metals]

# Granger ------------------
ml = 4
gc_dfs = {m: [grangercausalitytests(df_all[[t, m]], maxlag=ml, verbose=False) for t in tickers] for m in metals}
p_value_matrix = {lag: pd.DataFrame(index=tickers, columns=metals) for lag in range(1, ml+1)}
for m in metals:
    for t in tickers:
        for lag in range(1, ml+1):
            test_results = gc_dfs[m][tickers.index(t)]
            p_value = test_results[lag][0]['ssr_chi2test'][1]
            p_value_matrix[lag].at[t, m] = p_value
fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # Adjust as needed for more lags
axes = axes.flatten()

for lag in range(1, ml+1):
    ax = axes[lag-1]
    sns.heatmap(p_value_matrix[lag].astype(float), annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'P-Value'}, ax=ax)
    ax.set_title(f"Granger Causality P-Values for Lag {lag}")

plt.tight_layout()
plt.savefig(os.path.join("corr", "granger_causality_lags.png"))
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
                G.add_edge(t, m, label=f"Lag {lag}")

    # Draw the graph on the current subplot axis
    pos = nx.spring_layout(G, seed=42)  # Layout for the graph
    labels = nx.get_edge_attributes(G, 'label')

    # Plot the network on the current subplot
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=12, font_weight='bold',
            edge_color='blue', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10, font_color='red', ax=ax)

    # Set title for each lag's network plot
    ax.set_title(f"Granger Causality Network for Lag {lag}")

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig(os.path.join("corr", "granger_causality_network_lags.png"))
plt.show()
