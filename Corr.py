import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import ccf
import networkx as nx
from constants import tickers, metals


df_all = pd.read_csv('all.csv', index_col=0)
df_stocks = df_all[tickers]
df_metals = df_all[metals]

# CORR -----------------------------------
cor_matrix = df_all.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson correlation Heatmap")
plt.savefig(os.path.join("corr", "pearson_correlation.png"))
plt.show()

scor_matrix = df_all.corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(scor_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Spearman correlation Heatmap")
plt.savefig(os.path.join("corr", "spearman_correlation.png"))
plt.show()

# Cross-correlation ------------
for m in metals:
    i = 0
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(nrows=len(tickers) + 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    combined_data = []
    for t in tickers:
        cross_corr = ccf(df_all[t], df_all[m])
        combined_data.append(cross_corr)
        ax[i].plot(cross_corr)
        ax[i].text(1.02, 0.5, t,
                   transform=ax[i].transAxes, rotation='horizontal', va='center', ha='left')
        ax[i].set_ylabel('CCF')
        i += 1
    for cross_corr in combined_data:
        ax[-1].plot(cross_corr)
    ax[-1].text(1.02, 0.5, "Combined",
                transform=ax[i].transAxes, rotation='horizontal', va='center', ha='left')
    ax[-1].set_ylabel('CCF')
    ax[-1].set_xlabel('Lag')
    fig.suptitle(f"Cross correlations for {m}")
    fig.subplots_adjust(right=0.8)
    fig.savefig(os.path.join("corr", f"cross_correlations_{m}.png"))

# Rolling corr ------------------------
window_size = 30  # Rolling window size
for m in metals:
    i = 0
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(nrows=len(tickers) + 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    combined_data = []
    for t in tickers:
        rolling_corr = df_all[t].rolling(window_size).corr(df_all[m])
        combined_data.append(rolling_corr)
        ax[i].plot(rolling_corr)
        ax[i].text(1.02, 0.5, t,
                   transform=ax[i].transAxes, rotation='horizontal', va='center', ha='left')
        ax[i].set_ylabel('RCF')
        i += 1
    for cross_corr in combined_data:
        ax[-1].plot(cross_corr)
    ax[-1].text(1.02, 0.5, "Combined",
                transform=ax[i].transAxes, rotation='horizontal', va='center', ha='left')
    ax[-1].set_ylabel('CCF')
    ax[-1].set_xlabel('Date')
    fig.suptitle(f"Rolling correlations for {m}")
    fig.subplots_adjust(right=0.8)
    fig.savefig(os.path.join("corr", f"roll_correlations_{m}.png"))

# Network --------
threshold = 0.8
graph = nx.Graph()
c = cor_matrix
for i in c.columns:
    for j in c.columns:
        if i != j and abs(c[i][j]) > threshold:
            graph.add_edge(i, j, weight=c[i][j])
nx.draw(graph, with_labels=True, node_color="lightblue")
plt.title("Network in regards of Pearson correlation")
plt.savefig(os.path.join("corr", "pearson_correlation_network.png"))
plt.show()

threshold = 0.8
graph = nx.Graph()
c = scor_matrix
for i in c.columns:
    for j in c.columns:
        if i != j and abs(c[i][j]) > threshold:
            graph.add_edge(i, j, weight=c[i][j])
nx.draw(graph, with_labels=True, node_color="lightblue")
plt.title("Network in regards of Spearman correlation")
plt.savefig(os.path.join("corr", "spearman_correlation_network.png"))
plt.show()




