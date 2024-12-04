import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

indices_path = "indices"
commodities_path = "commodities"
tickers = ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI", "^FCHI", "^N100", "EURUSD=X", "^HSI", "^DXS", "GD=F", "EURRUB=X"]
metals = ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "ALI=F"]

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
    plt.figure(figsize=(12, 8))

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
    plt.axvline(x=pd.to_datetime("2022-02-24"), color='red', linestyle='--', label='Ukrainos-Rusijos karo pradžia')
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Kaina, $')
    plt.legend(loc='best')  # Show the legend with the names
    plt.grid(True)

    # Show the plot
    plt.show()

plot_all_on_one_plot(indices_path, "Pasaulio indeksai")
plot_all_on_one_plot(commodities_path, "Metalų indeksai")


for ticker in tickers:
    ticker_data = yf.Ticker(ticker)
    hist_data = ticker_data.history(period="5y")
    file_name = os.path.join(indices_path, f"{ticker.replace('^', '')}_historical_data.csv")
    hist_data.to_csv(file_name)
    print(f"Data has been exported to {file_name}.")

for metal in metals:
    ticker_data = yf.Ticker(metal)
    hist_data = ticker_data.history(period="5y")
    file_name = os.path.join(commodities_path, f"{metal.split('=')[0]}_historical_data.csv")
    hist_data.to_csv(file_name)
    print(f"Data has been exported to {file_name}.")


def one_df(indices_path, commodities_path, tickers, metals):
    df = None
    for ticker in tickers:
        temp = pd.read_csv(os.path.join(indices_path, f"{ticker.replace('^', '')}_historical_data.csv"))
        temp['Date'] = pd.to_datetime(temp['Date'])
        temp['Date']=temp['Date'].apply(lambda x: x.date())
        temp = temp[['Date', 'Close']].rename(columns={'Close': f'{ticker}_Close'})
        df = temp if df is None else pd.merge(df, temp, on='Date', how='outer')
    for metal in metals:
        temp = pd.read_csv(os.path.join(commodities_path, f"{metal.split('=')[0]}_historical_data.csv"))
        temp['Date'] = pd.to_datetime(temp['Date'])
        temp['Date']=temp['Date'].apply(lambda x: x.date())
        temp = temp[['Date', 'Close']].rename(columns={'Close': f'{metal}_Close'})
        df = pd.merge(df, temp, on='Date', how='outer')
    return df.sort_values(by='Date', ascending=True).reset_index(drop=True)

df_all=one_df(indices_path,commodities_path,tickers,metals)
df_all.set_index('Date', inplace=True)
df_all.index = pd.to_datetime(df_all.index)
df_all.to_csv('all.csv')


