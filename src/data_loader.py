import yfinance as yf

def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]

    # Ensure DataFrame even for single ticker
    if isinstance(data, dict):
        data = data.to_frame()

    return data.dropna(how="all")
