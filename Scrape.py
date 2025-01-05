import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from constants import indices_path, commodities_path, tickers, metals, start_date, end_date, analysis_end_date, war_date

os.makedirs(indices_path, exist_ok=True)
os.makedirs(commodities_path, exist_ok=True)


def normalize_data_01(series):
    return (series - series.min()) / (series.max() - series.min())


def plot_historical_data(csv_path, title):
    # Load the historical data from CSV
    data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)

    # Plot the closing price
    plt.figure(figsize=(10, 6))
    plt.plot(normalize_data_01(data['Close']), label="Close Price", color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_on_one_plot(directory, title):
    plt.figure(figsize=(18, 6))

    # Loop over all CSV files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv"):
            # Load each CSV file
            file_path = os.path.join(directory, file_name)
            data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

            # Extract the name from the file name and use it as the label
            label = file_name.split('_')[0]

            # Plot the 'Close' price for each file
            plt.plot(normalize_data_01(data['Close']), label=label)

    # Customize the plot
    plt.axvline(x=pd.to_datetime(war_date), color='red', linestyle='--', label='Ukrainos-Rusijos\nkaro pradžia')
    plt.axvline(x=pd.to_datetime(analysis_end_date), color='green', linestyle='--', label='Analizės/Ateities\nduomenų riba')
    plt.title(title)
    plt.xlim(pd.to_datetime(start_date)-pd.Timedelta(weeks=1), pd.to_datetime(end_date)+pd.Timedelta(weeks=1))
    plt.xlabel('Data')
    plt.ylabel('Kaina, $')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Show the legend with the names
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"{directory}.png")
    plt.show()


def scrape_data(path_to_save, list_of_indices):
    for ticker in list_of_indices:
        ticker_data = yf.Ticker(ticker)
        hist_data = ticker_data.history(start=start_date, end=pd.to_datetime(end_date)+pd.Timedelta(days=3))
        file_name = os.path.join(path_to_save, f"{ticker.replace('^', '').split('=')[0]}_historical_data.csv")
        hist_data['Date'] = pd.to_datetime(hist_data.index)
        hist_data['Date'] = hist_data['Date'].dt.date
        hist_data = hist_data.set_index('Date')
        hist_data = hist_data.iloc[::-1]
        hist_data.to_csv(file_name)
        print(f"Data has been exported to {file_name}.")


def one_df(indices_path, commodities_path, tickers, metals):
    df = None
    for ticker in tickers:
        temp = pd.read_csv(os.path.join(indices_path, f"{ticker.replace('^', '').split('=')[0]}_historical_data.csv"))
        temp = temp[['Date', 'Close']].rename(columns={'Close': ticker})
        df = temp if df is None else pd.merge(df, temp, on='Date', how='outer')
    for metal in metals:
        temp = pd.read_csv(os.path.join(commodities_path, f"{metal.split('=')[0]}_historical_data.csv"))
        temp = temp[['Date', 'Close']].rename(columns={'Close': metal})
        df = pd.merge(df, temp, on='Date', how='outer')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index(ascending=False)
    df = df.interpolate(method='linear')
    df = df.sort_index()
    df = df[df.index < pd.to_datetime(end_date)]
    return df


scrape_data(indices_path, tickers)
scrape_data(commodities_path, metals)


df_all = one_df(indices_path,commodities_path,tickers,metals)
df_all.to_csv('all.csv')
print(f"Analysis period: {start_date} to {analysis_end_date}. Future period: {analysis_end_date} to {end_date}")
print("Whole data saved: all.csv")
df_all_analysis = df_all[df_all.index < pd.to_datetime(analysis_end_date)]
df_all_future = df_all[df_all.index >= pd.to_datetime(analysis_end_date)]


# Aggregation
df_weekly = df_all.resample('W').mean().sort_index()      # Weekly average
df_bi_weekly = df_all.resample('2W').mean().sort_index()  # Bi-weekly average

# Save the combined datasets
df_weekly.to_csv('weekly.csv')
df_bi_weekly.to_csv('bi_weekly.csv')

print("Aggregations saved:")
print("- Weekly: weekly.csv")
print("- Bi-weekly: bi_weekly.csv")

# Plots
plot_all_on_one_plot(indices_path, "Pasaulio indeksai")
plot_all_on_one_plot(commodities_path, "Metalų indeksai")