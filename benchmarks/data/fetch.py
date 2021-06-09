"""Function to fetch the data when they cna be retrieved from the internet."""
import pandas as pd


def fetch_nasdaq_close():
    url = 'https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=959817600&period2=1622505600&interval=1d&events=history'
    data = pd.read_csv(url)
    close = data['Close']
    close.to_csv('stochastic_volatility.csv', header=['CLOSING_PRICE'], index=False)


if __name__ == "__main__":
    fetch_nasdaq_close()
