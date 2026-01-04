import numpy as np
import pandas as pd
import statsmodels.api as sm

def compute_spread(series1, series2):
    model = sm.OLS(series1, sm.add_constant(series2)).fit()
    hedge_ratio = model.params[1]
    spread = series1 - hedge_ratio * series2
    return spread, hedge_ratio

def zscore(series, window=30):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std
