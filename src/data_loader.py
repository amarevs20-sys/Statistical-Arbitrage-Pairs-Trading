import yfinance as yf
import pandas as pd

def load_prices(tickers, start, end):
    # 1. Download data with group_by='column' to ensure standard formatting
    data = yf.download(tickers, start=start, end=end)
    
    # 2. Check if data is empty
    if data.empty:
        raise ValueError(f"No data found for tickers {tickers} between {start} and {end}")

    # 3. Handle 'Adj Close' safely
    # In newer yfinance versions, it might be 'Adj Close' or just 'Close'
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        raise KeyError(f"Could not find price data in columns: {data.columns}")

    # 4. If only one ticker was passed, prices is a Series. Convert to DataFrame.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        
    return prices
