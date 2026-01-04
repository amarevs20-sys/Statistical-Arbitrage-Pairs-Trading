import matplotlib.pyplot as plt
from src.data_loader import load_prices
from src.pair_selection import find_cointegrated_pairs
from src.strategy import compute_spread, zscore
from src.backtest import backtest
from src.metrics import sharpe_ratio, max_drawdown

tickers = ["KO", "PEP", "MCD", "YUM", "DPZ"]
prices = load_prices(tickers, "2018-01-01", "2024-01-01")

pairs = find_cointegrated_pairs(prices)
if len(pairs) == 0:
    raise ValueError("No cointegrated pairs found. Try different tickers or date range.")

pair = pairs[0][:2]
print("Selected pair:", pair)


s1 = prices[pair[0]]
s2 = prices[pair[1]]

spread, hedge = compute_spread(s1, s2)
z = zscore(spread)

returns = backtest(spread, z)
cum_returns = (1 + returns).cumprod()

print("Sharpe:", sharpe_ratio(returns))
print("Max Drawdown:", max_drawdown(cum_returns))

cum_returns.plot(title="Cumulative PnL")
plt.show()
print(prices.head())
print(prices.columns)
split_date = "2021-01-01"

train = prices.loc[:split_date]
test = prices.loc[split_date:]

pairs = find_cointegrated_pairs(train)
pair = pairs[0][:2]

s1_train = train[pair[0]]
s2_train = train[pair[1]]

spread_train, hedge = compute_spread(s1_train, s2_train)

s1_test = test[pair[0]]
s2_test = test[pair[1]]
spread_test = s1_test - hedge * s2_test

z_test = zscore(spread_test)

returns = backtest(spread_test, z_test)
cum_returns = (1 + returns).cumprod()
