import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import ccf

from constants import tickers, metals, BASE

SCALE = True
os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, 'corr'), exist_ok=True)

df = pd.read_csv(f'{BASE}.csv', index_col=0)
if SCALE:
    scaler = MinMaxScaler(feature_range=(1, 2))
    df_all = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
else:
    df_all = df
df_all = df_all.iloc[::-1]
df_stocks = df_all[tickers]
df_metals = df_all[metals]

# CORR -----------------------------------
cor_matrix = df_all.corr(method='pearson')
mask = np.triu(np.ones_like(cor_matrix, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, cbar_kws={"shrink": 0.8})
plt.title("Pearson correlation Heatmap")
plt.savefig(os.path.join(BASE, "corr", "pearson_correlation.png"))
plt.show()

scor_matrix = df_all.corr(method='spearman')
mask = np.triu(np.ones_like(scor_matrix, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(scor_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, cbar_kws={"shrink": 0.8})
plt.title("Spearman correlation Heatmap")
plt.savefig(os.path.join(BASE, "corr", "spearman_correlation.png"))
plt.show()

# Cross-correlation ------------
for m in metals:
    i = 0
    fig = plt.figure(figsize=(8, 14))
    gs = fig.add_gridspec(nrows=len(tickers) + 1, hspace=0)
    ax = gs.subplots(sharex=True, sharey=True)
    combined_data = []
    for t in tickers:
        cross_corr = ccf(df_all[t], df_all[m])
        combined_data.append(cross_corr)
        ax[i].plot(cross_corr)
        ax[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax[i].text(1.02, 0.5, t,
                   transform=ax[i].transAxes, rotation='horizontal', va='center', ha='left')
        ax[i].set_ylabel('CCF')
        ax[i].grid()
        i += 1
    for cross_corr in combined_data:
        ax[-1].plot(cross_corr)
    ax[-1].text(1.02, 0.5, "Combined",
                transform=ax[i].transAxes, rotation='horizontal', va='center', ha='left')
    ax[-1].set_ylabel('CCF')
    ax[-1].set_xlabel('Lag')
    ax[-1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax[-1].grid()
    fig.suptitle(f"Cross correlations for {m}")
    fig.subplots_adjust(right=0.8)
    fig.savefig(os.path.join(BASE, "corr", f"cross_correlations_{m}.png"))

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
        ax[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax[i].text(1.02, 0.5, t,
                   transform=ax[i].transAxes, rotation='horizontal', va='center', ha='left')
        ax[i].set_ylabel('RCF')
        ax[i].grid()
        i += 1
    for cross_corr in combined_data:
        ax[-1].plot(cross_corr)
    ax[-1].text(1.02, 0.5, "Combined",
                transform=ax[i].transAxes, rotation='horizontal', va='center', ha='left')
    ax[-1].set_ylabel('CCF')
    ax[-1].set_xlabel('Date')
    ax[-1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax[-1].grid()
    fig.suptitle(f"Rolling correlations for {m}")
    fig.subplots_adjust(right=0.8)
    fig.savefig(os.path.join(BASE, "corr", f"roll_correlations_{m}.png"))

# Network --------
threshold = 0.8
graph = nx.Graph()
c = cor_matrix
fig = plt.figure(figsize=(8, 8))
for i in c.columns:
    for j in c.columns:
        if i != j and abs(c[i][j]) > threshold:
            graph.add_edge(i, j, weight=c[i][j])
nx.draw(graph, with_labels=True, node_color="lightblue")
plt.title("Network in regards of Pearson correlation")
plt.savefig(os.path.join(BASE, "corr", "pearson_correlation_network.png"))
plt.show()

threshold = 0.8
graph = nx.Graph()
c = scor_matrix
fig = plt.figure(figsize=(8, 8))
for i in c.columns:
    for j in c.columns:
        if i != j and abs(c[i][j]) > threshold:
            graph.add_edge(i, j, weight=c[i][j])
nx.draw(graph, with_labels=True, node_color="lightblue")
plt.title("Network in regards of Spearman correlation")
plt.savefig(os.path.join(BASE, "corr", "spearman_correlation_network.png"))
plt.show()
