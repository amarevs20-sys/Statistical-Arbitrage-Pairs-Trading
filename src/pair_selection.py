from statsmodels.tsa.stattools import coint
import numpy as np

def find_cointegrated_pairs(prices, pval_threshold=0.05, min_obs=504):
    tickers = prices.columns
    results = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]

            s1 = prices[t1]
            s2 = prices[t2]

            data = s1.dropna().to_frame("s1").join(s2.dropna().to_frame("s2"), how="inner")

            if len(data) < min_obs:
                continue

            score, pvalue, _ = coint(data["s1"], data["s2"])
            results.append((t1, t2, pvalue))

    # sort by p-value
    results.sort(key=lambda x: x[2])

    # return both strict + ranked results
    tradable = [r for r in results if r[2] <= pval_threshold]

    return tradable if len(tradable) > 0 else results
