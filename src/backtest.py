import pandas as pd

def backtest(spread, z, entry=2.0, exit=0.5, cost=0.0005):
    position = 0
    pnl = []

    for i in range(1, len(spread)):
        prev_position = position

        if z.iloc[i] > entry:
            position = -1
        elif z.iloc[i] < -entry:
            position = 1
        elif abs(z.iloc[i]) < exit:
            position = 0

        trade_cost = cost if position != prev_position else 0
        daily_pnl = position * (spread.iloc[i] - spread.iloc[i-1]) - trade_cost
        pnl.append(daily_pnl)

    return pd.Series(pnl, index=spread.index[1:])
