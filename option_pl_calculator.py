import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from pytz import timezone  # For timezone conversion

# Set page layout
st.set_page_config(page_title="Cool Options P/L Calculator", layout="wide")
st.title("Options P/L Calculator with Live Chain, Greeks, and Insights")

# Sidebar settings
st.sidebar.header("Settings")
ticker_symbol = st.sidebar.text_input("📈 Ticker Symbol", value="CHWY")
ticker = yf.Ticker(ticker_symbol)

# Refresh button
if st.sidebar.button("🔄 Refresh Options Chain"):
    ticker = yf.Ticker(ticker_symbol)

# Fetch options chain data
try:
    expirations = ticker.options
    expiration_date = st.sidebar.selectbox("🗓 Expiration Date", expirations)
    options_chain = ticker.option_chain(expiration_date)
except IndexError:
    st.write("Error fetching options data. Check the ticker symbol.")
    options_chain = None

# Function to format options chain data
def format_options_chain(df, option_type):
    # Rename columns and format "Last Trade Date" in Eastern Time
    df = df.rename(columns={
        "contractSymbol": "Contract Symbol",
        "lastTradeDate": "Last Trade Date (EST)"
    })
    # Convert "Last Trade Date" to EST and reformat
    df["Last Trade Date (EST)"] = pd.to_datetime(df["Last Trade Date (EST)"]).dt.tz_convert(timezone("America/New_York"))
    df["Last Trade Date (EST)"] = df["Last Trade Date (EST)"].dt.strftime('%m/%d/%Y %I:%M %p')
    
    # Highlight in-the-money options based on option type
    current_price = ticker.history(period='1d')['Close'].iloc[0]
    def highlight_in_the_money(val):
        if option_type == 'call' and val < current_price:
            return 'background-color: #c6efce; color: #006100'
        elif option_type == 'put' and val > current_price:
            return 'background-color: #ffc7ce; color: #9c0006'
        return ''

    # Apply formatting to the DataFrame
    styled_df = df.style.format({
        'strike': "{:.2f}",
        'lastPrice': "{:.2f}",
        'bid': "{:.2f}",
        'ask': "{:.2f}",
        'volume': "{:.0f}",
        'openInterest': "{:.0f}"
    }).applymap(highlight_in_the_money, subset=['strike'])
    
    return styled_df, df

# Display options chain with selection
if options_chain:
    st.subheader("Live Options Chain")

    # Format calls and puts tables
    calls_table, calls_df = format_options_chain(options_chain.calls, 'call')
    puts_table, puts_df = format_options_chain(options_chain.puts, 'put')

    # Option to select an individual call or put for P/L calculation
    selected_option = st.selectbox(
        "Select an Option for P/L Calculation",
        calls_df['Contract Symbol'].tolist() + puts_df['Contract Symbol'].tolist()
    )

    # Display formatted Calls table
    st.write("**📊 Calls**")
    st.dataframe(calls_table)
    
    # Display formatted Puts table
    st.write("**📉 Puts**")
    st.dataframe(puts_table)

    # Pre-fill the P/L model inputs based on selected option
    selected_option_data = calls_df[calls_df['Contract Symbol'] == selected_option]
    if selected_option_data.empty:
        selected_option_data = puts_df[puts_df['Contract Symbol'] == selected_option]
    if not selected_option_data.empty:
        strike_price = selected_option_data['strike'].values[0]
        premium = selected_option_data['lastPrice'].values[0]
        option_type = 'call' if selected_option in calls_df['Contract Symbol'].tolist() else 'put'
    else:
        strike_price = 100
        premium = 5

# Sidebar inputs for P/L model
position_type = st.sidebar.selectbox("📑 Position Type", ["long", "short"])
strike_price = st.sidebar.slider("Strike Price", min_value=1.0, max_value=200.0, value=float(strike_price), step=1.0)
premium = st.sidebar.slider("Premium (Cost of Option)", min_value=0.0, max_value=50.0, value=float(premium), step=0.1)

# Define the stock price range for P/L calculation
stock_price_min = st.sidebar.slider("Stock Price Min", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
stock_price_max = st.sidebar.slider("Stock Price Max", min_value=0.0, max_value=200.0, value=100.0, step=1.0)
stock_price_range = np.arange(stock_price_min, stock_price_max + 1, 1)

# Black-Scholes Greeks Calculation
def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    return delta, gamma, theta, vega, rho

# Function to calculate P/L for different option scenarios
def calculate_option_pl(option_type, position_type, strike_price, stock_price_at_expiration, premium):
    if option_type == 'call':
        if position_type == 'long':
            return max(stock_price_at_expiration - strike_price, 0) - premium
        elif position_type == 'short':
            return premium - max(stock_price_at_expiration - strike_price, 0)
    elif option_type == 'put':
        if position_type == 'long':
            return max(strike_price - stock_price_at_expiration, 0) - premium
        elif position_type == 'short':
            return premium - max(strike_price - stock_price_at_expiration, 0)

# Calculate P/L across different expiration stock prices
pl_values = [calculate_option_pl(option_type, position_type, strike_price, sp, premium) for sp in stock_price_range]

# Calculate Greeks for the selected option
S = ticker.history(period='1d')['Close'].iloc[0]
T = (pd.to_datetime(expiration_date) - pd.Timestamp.now()).days / 365
r = 0.01  # Assume 1% risk-free rate
sigma = 0.2  # Assume 20% volatility
delta, gamma, theta, vega, rho = calculate_greeks(option_type, S, strike_price, T, r, sigma)

# Display Greeks and performance metrics
st.sidebar.markdown("### Option Metrics")
st.sidebar.write(f"**Delta**: {delta:.2f}")
st.sidebar.write(f"**Gamma**: {gamma:.4f}")
st.sidebar.write(f"**Theta**: {theta:.2f}")
st.sidebar.write(f"**Vega**: {vega:.2f}")
st.sidebar.write(f"**Rho**: {rho:.2f}")

# Display summary of selected option
st.sidebar.markdown("### Selected Option Summary")
st.sidebar.write(f"**Option Type**: {option_type.capitalize()}")
st.sidebar.write(f"**Strike Price**: {strike_price}")
st.sidebar.write(f"**Premium**: {premium}")
st.sidebar.write(f"**Position**: {position_type.capitalize()}")

# Plotting the P/L profile
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(stock_price_range, pl_values, label=f"{position_type.capitalize()} {option_type.capitalize()}")
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Stock Price at Expiration')
ax.set_ylabel('Profit / Loss')
ax.set_title(f'P/L Profile for {position_type.capitalize()} {option_type.capitalize()} Option')
ax.legend()

# Show plot in Streamlit app
st.pyplot(fig)
