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

    with st.spinner("Fetching data and running cointegration tests..."):
        # 1. Load data
        prices = load_prices(tickers, start, end)

        # 2. Cleanup: Remove columns with more than 20% missing values
        # This prevents the cointegration test from crashing on "bad" tickers
        prices = prices.dropna(thresh=len(prices) * 0.8, axis=1).dropna()

        if prices.empty or prices.shape[1] < 2:
            st.error("Not enough valid data points found after cleaning. Try a different date range.")
            st.stop()

        # 3. Find Cointegrated Pairs
        pairs = find_cointegrated_pairs(prices)

        if not pairs:
            st.error("No valid pairs could be evaluated with the current selection.")
            st.stop()

        # 4. Get the best pair
        t1, t2, pval = pairs[0]

        # --- FIX FOR MULTI-INDEXING ---
        # We ensure we are grabbing the column correctly regardless of DF structure
        s1 = prices[t1]
        s2 = prices[t2]

        if pval <= 0.05:
            st.success(f"Tradable cointegrated pair found: {t1} & {t2} (p-value: {pval:.4f})")
        else:
            st.warning(f"No statistically significant pairs (p < 0.05). Showing best candidate: {t1}/{t2}")

        # --------------------------------------------------
        # STRATEGY & BACKTEST
        # --------------------------------------------------
        spread, hedge = compute_spread(s1, s2)
        z = zscore(spread)

        returns = backtest(spread, z, entry=entry_z, exit=exit_z)
        cum_returns = returns.cumsum()

        # --------------------------------------------------
        # METRICS DISPLAY
        # --------------------------------------------------
        st.subheader(f"Strategy Performance: {t1} vs {t2}")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Sharpe Ratio", f"{sharpe_ratio(returns):.2f}")
        m_col2.metric("Max Drawdown", f"{max_drawdown(cum_returns):.2%}")
        m_col3.metric("Hedge Ratio (Beta)", f"{hedge:.3f}")

        # --------------------------------------------------
        # PLOTS
        # --------------------------------------------------
        tab1, tab2 = st.tabs(["PnL Curve", "Z-Score Signals"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(cum_returns, color='green', label='Cumulative Returns')
            ax.set_ylabel("PnL")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(z, label='Spread Z-Score', color='blue')
            ax2.axhline(entry_z, color='red', linestyle="--", label='Entry')
            ax2.axhline(-entry_z, color='red', linestyle="--")
            ax2.axhline(exit_z, color='orange', linestyle=":", label='Exit')
            ax2.axhline(-exit_z, color='orange', linestyle=":")
            ax2.legend()
            st.pyplot(fig2)

        # --------------------------------------------------
        # TOP CANDIDATES TABLE
        # --------------------------------------------------
        st.subheader("Top Research Candidates")
        st.table([{"Asset 1": p[0], "Asset 2": p[1], "P-Value": round(p[2], 4)} for p in pairs[:5]])

else:
    st.info("Adjust the sidebar parameters and click **Run Backtest** to begin.")
