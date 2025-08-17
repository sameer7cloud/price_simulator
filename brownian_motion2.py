import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm
import streamlit as st

st.set_page_config(layout="wide")
st.title("📈 Simulation for Cryptos")

# --- UI ---
tickers = st.text_input("Enter Tickers (comma-separated):", value="ETH-USD, BTC-USD, LTC-USD, SOL-USD, DOGE-USD, NEO-USD")
start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
end_date = st.date_input("End Date", value=datetime(2025, 1, 1))
future_date = st.date_input("Future Date", value=datetime.today())
num_instances = st.number_input("Number of Simulations", value=1000, step=100)

simulate_btn = st.button("Run Simulation")

# --- Logic Functions ---

def fetch_asset_data(ticker, start_date, end_date):
    asset_data = yf.download(ticker, start=start_date, end=end_date)
    if asset_data.empty:
        raise ValueError(f"No data for {ticker}")
    return asset_data

def generate_brownian_motion(asset_data, future_days, num_instances):
    log_returns = np.log(asset_data['Close'] / asset_data['Close'].shift(1)).dropna()
    mean = log_returns.mean()
    std_dev = log_returns.std()
    last_price = asset_data['Close'].iloc[-1]

    simulations = []
    for _ in range(num_instances):
        prices = [last_price]
        for _ in range(future_days):
            drift = mean - 0.5 * std_dev ** 2
            shock = std_dev * np.random.normal()
            change = prices[-1] * np.exp(drift + shock)
            prices.append(change)
        simulations.append(prices)
    return simulations, mean, std_dev

# --- Main App Logic ---

if simulate_btn:
    try:
        st.subheader("🔄 Running Simulations...")
        tickers = [t.strip().upper() for t in tickers.split(",")]
        future_days = abs((future_date - end_date).days)

        final_prices_dict = {}
        for ticker in tickers:
            st.write(f"Fetching data for: `{ticker}`")
            asset_data = fetch_asset_data(ticker, start_date, end_date)
            simulations, mean, std_dev = generate_brownian_motion(asset_data, future_days, num_instances)
            final_prices = np.array([sim[-1] for sim in simulations])
            final_prices_dict[ticker] = final_prices.ravel()


        result_df = pd.DataFrame(final_prices_dict)
        percentiles = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25,
                       0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 1.0]
        result = result_df.quantile(percentiles)
        result.index = result.index * 100

        st.subheader("📊 Simulated Price Percentiles")
        st.markdown("#### Prices below 10th percentile and 90th percentile can be considered as entry and exit points for long position")
        st.dataframe(result.style.format("{:.2f}"))

    except Exception as e:
        st.error(f"🚫 Error: {e}")
