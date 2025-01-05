import os
from fitter import Fitter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from constants import tickers, metals, BASE, war_date, analysis_end_date

SCALE = False
os.makedirs(BASE, exist_ok=True)
os.makedirs(os.path.join(BASE, 'distr'), exist_ok=True)

df = pd.read_csv(f'{BASE}.csv', index_col=0)
df.index = pd.to_datetime(df.index)
if SCALE:
    scaler = MinMaxScaler(feature_range=(1, 2))
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
else:
    df = df
df = df[df.index < pd.to_datetime(analysis_end_date)]
df = df.pct_change().dropna()

recommended_distributions = [
    "norm",         # Normal distribution
    "t",            # Student's t-distribution
    "lognorm",      # Log-Normal distribution
    "expon",        # Exponential distribution
    "gamma",        # Gamma distribution
    "weibull_min",  # Weibull distribution (minimum)
    "weibull_max",  # Weibull distribution (maximum)
    "gennorm",      # Generalized Normal distribution
    "pareto",       # Pareto distribution
    "cauchy"        # Cauchy distribution
]

WAR = [None, True, False]

# Path for the consolidated Excel file
output_file = os.path.join(BASE, 'distr', 'distributions_summary.xlsx')

# Using ExcelWriter to handle multiple sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for t in tickers + metals:
        print(f"Analyzing index = {t}")
        data = df[t].copy()
        for war in WAR:
            print(f"Analyzing WAR = {war}")
            KARAS = 'full'
            df_all = data
            if war is not None:
                df_all = data[data.index >= pd.to_datetime(war_date) if war else data.index < pd.to_datetime(war_date)]
                KARAS = 'po' if war else 'pries'

            fitter = Fitter(df_all, distributions=recommended_distributions)
            fitter.fit()
            summary_df = fitter.df_errors  # Access the full DataFrame
            ordered_summary = summary_df.sort_values(by="ks_pvalue", ascending=False)
            # Save the summary to a new sheet
            sheet_name = f"{t}_{KARAS}"[:31]  # Sheet name should be <= 31 characters
            ordered_summary.to_excel(writer, sheet_name=sheet_name)

print(f"All results have been saved to {output_file}.")


