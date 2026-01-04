import numpy as np

def sharpe_ratio(returns):
    if returns.std() == 0:
        return 0.0
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(returns):
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min()
