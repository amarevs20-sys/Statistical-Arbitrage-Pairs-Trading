import streamlit as st
import matplotlib.pyplot as plt
import datetime

from src.data_loader import load_prices
from src.pair_selection import find_cointegrated_pairs
from src.strategy import compute_spread, zscore
from src.backtest import backtest
from src.metrics import sharpe_ratio, max_drawdown

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Statistical Arbitrage Research Platform",
    layout="wide"
)

st.title("📈 Statistical Arbitrage – Pairs Trading Research")
st.caption("Cointegration-based market-neutral strategy")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Strategy Parameters")

tickers = st.sidebar.multiselect(
    "Select Tickers",
    [
        "AAPL","MSFT","GOOG","META","AMZN","NVDA",
        "AMD","INTC","AVGO","QCOM",
        "JPM","BAC","WFC","C","GS",
        "XOM","CVX","COP","OXY"
    ],
    default=["AAPL", "MSFT", "GOOG", "META"]
)

start = st.sidebar.date_input(
    "Start Date",
    value=datetime.date(2014, 1, 1)
)

end = st.sidebar.date_input(
    "End Date",
    value=datetime.date(2024, 1, 1)
)

entry_z = st.sidebar.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)
exit_z = st.sidebar.slider("Exit Z-Score", 0.0, 1.5, 0.5, 0.1)

run = st.sidebar.button("Run Backtest")

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if run:

    if len(tickers) < 2:
        st.error("Please select at least two tickers.")
        st.stop()

    with st.spinner("Running analysis..."):

        prices = load_prices(tickers, start, end)

        if prices.isnull().all().any():
            st.error("One or more tickers returned no usable price data.")
            st.stop()

        pairs = find_cointegrated_pairs(prices)

        if len(pairs) == 0:
            st.error("No valid pairs could be evaluated (insufficient overlapping data).")
            st.stop()

        # Ranked pairs – always safe
        t1, t2, pval = pairs[0]

        if pval <= 0.05:
            st.success(f"Tradable cointegrated pair found (p-value = {pval:.4f})")
        else:
            st.warning(
                f"No statistically tradable pairs found.\n\n"
                f"Showing best research candidate instead (p-value = {pval:.4f})."
            )

        # --------------------------------------------------
        # STRATEGY
        # --------------------------------------------------
        s1 = prices[t1]
        s2 = prices[t2]

        spread, hedge = compute_spread(s1, s2)
        z = zscore(spread)

        # 🔴 MATCHES BACKTEST SIGNATURE EXACTLY
        returns = backtest(
            spread,
            z,
            entry=entry_z,
            exit=exit_z
        )

        cum_returns = returns.cumsum()

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------
        st.subheader("Selected Pair")
        st.write(f"**{t1} / {t2}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe Ratio", f"{sharpe_ratio(returns):.2f}")
        col2.metric("Max Drawdown", f"{max_drawdown(cum_returns):.2%}")
        col3.metric("Hedge Ratio", f"{hedge:.2f}")

        # --------------------------------------------------
        # PLOTS
        # --------------------------------------------------
        st.subheader("Cumulative Strategy PnL")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cum_returns)
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL")
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("Spread Z-Score")

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(z)
        ax2.axhline(entry_z, linestyle="--")
        ax2.axhline(-entry_z, linestyle="--")
        ax2.axhline(exit_z, linestyle=":")
        ax2.axhline(-exit_z, linestyle=":")
        ax2.grid(True)
        st.pyplot(fig2)

        # --------------------------------------------------
        # TOP CANDIDATES
        # --------------------------------------------------
        st.subheader("Top Candidate Pairs (Lowest p-values)")
        for a, b, p in pairs[:5]:
            st.write(f"{a} / {b} — p-value: {p:.4f}")

else:
    st.info("Select tickers and click **Run Backtest**.")
