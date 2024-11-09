import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from scipy.stats import norm
from pytz import timezone  # For timezone conversion
import datetime

# Set page layout
st.set_page_config(page_title="Options P/L Calculator", layout="wide")
st.title("Options P/L Calculator with Multi-Leg Strategies")

# Sidebar settings
st.sidebar.header("Settings")
ticker_symbol = st.sidebar.text_input("📈 Ticker Symbol", value="CHWY")

# Validate ticker symbol and get current price
ticker = yf.Ticker(ticker_symbol)
try:
    current_price = ticker.info.get('regularMarketPrice')
    if current_price is None:
        # Try to get current price from history
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
except Exception:
    st.error("Unable to fetch current price. Please enter a valid ticker symbol.")
    st.stop()

# Fetch options chain data
try:
    expirations = ticker.options
    if not expirations:
        st.error("No options data available for this ticker.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching options data: {e}")
    st.stop()

# Convert expiration dates to datetime.date objects
expiration_dates = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in expirations]
expiration_set = set(expiration_dates)

# Select expiration date using date_input
st.sidebar.markdown("### Select Expiration Date")
selected_expiration_date = st.sidebar.date_input(
    "Expiration Date",
    min_value=min(expiration_dates),
    max_value=max(expiration_dates),
    value=min(expiration_dates)
)

# Check if selected date is in expiration dates
if selected_expiration_date not in expiration_set:
    st.warning("Selected date does not have options expirations. Please select a valid expiration date.")
    st.stop()
else:
    # Convert selected date back to string format
    expiration_date = selected_expiration_date.strftime('%Y-%m-%d')

# Cache the options chain data fetching
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_options_chain(ticker_symbol, expiration_date):
    ticker = yf.Ticker(ticker_symbol)
    options = ticker.option_chain(expiration_date)
    # Convert to DataFrame to make it serializable
    calls_df = options.calls
    puts_df = options.puts
    return calls_df, puts_df

calls_df, puts_df = get_options_chain(ticker_symbol, expiration_date)

# Combine calls and puts for easier lookup
options_chain = pd.concat([calls_df, puts_df])

# Initialize session state for option legs
if 'option_legs' not in st.session_state:
    st.session_state['option_legs'] = []

# Sidebar for adding option legs
st.sidebar.markdown("### Option Legs")
if st.sidebar.button('➕ Add Option Leg'):
    st.session_state['option_legs'].append({})

# Function to remove an option leg
def remove_leg(i):
    st.session_state['option_legs'].pop(i)

# Display option legs
for i, leg in enumerate(st.session_state['option_legs']):
    with st.sidebar.expander(f"Option Leg {i+1}", expanded=True):
        # Option Type selection
        option_type = st.selectbox(f"Option Type", ['Call', 'Put'], key=f'option_type_{i}')
        # Position Type selection
        position_type = st.selectbox("Position Type", ["Long", "Short"], key=f'position_type_{i}')
        # Options chain selection
        options_df = calls_df if option_type == 'Call' else puts_df
        contract_symbol = st.selectbox("Contract Symbol", options_df['contractSymbol'].tolist(), key=f'contract_symbol_{i}')
        # Store the leg data
        st.session_state['option_legs'][i]['option_type'] = option_type
        st.session_state['option_legs'][i]['position_type'] = position_type
        st.session_state['option_legs'][i]['contract_symbol'] = contract_symbol
        # Remove leg button
        if st.button(f'Remove Leg {i+1}', key=f'remove_leg_{i}'):
            remove_leg(i)
            st.experimental_rerun()

# Allow user to input risk-free rate
r = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Risk-free interest rate as a percentage.") / 100

# Define the stock price range for P/L calculation
stock_price_min = st.sidebar.number_input("Stock Price Min", min_value=0.0, max_value=100000.0, value=max(0.0, current_price - current_price * 0.5), step=1.0)
stock_price_max = st.sidebar.number_input("Stock Price Max", min_value=0.0, max_value=100000.0, value=current_price + current_price * 0.5, step=1.0)
stock_price_range = np.arange(stock_price_min, stock_price_max + 1, 1)

# Calculate time to expiration
expiration_datetime = datetime.datetime.combine(selected_expiration_date, datetime.datetime.min.time())
current_datetime = datetime.datetime.now()
T = (expiration_datetime - current_datetime).total_seconds() / (365 * 24 * 3600)
T = max(T, 0.0001)  # Ensure T is positive

# Gather all selected legs data
legs = []

for leg in st.session_state['option_legs']:
    option_type = leg.get('option_type')
    position_type = leg.get('position_type')
    contract_symbol = leg.get('contract_symbol')
    if not contract_symbol:
        continue
    option_data = options_chain[options_chain['contractSymbol'] == contract_symbol]
    if not option_data.empty:
        strike_price = option_data['strike'].values[0]
        premium = option_data['lastPrice'].values[0]
        implied_volatility = option_data['impliedVolatility'].values[0]
        leg_data = {
            'option_type': option_type,
            'position_type': position_type,
            'strike_price': strike_price,
            'premium': premium,
            'implied_volatility': implied_volatility,
            'contract_symbol': contract_symbol
        }
        legs.append(leg_data)
    else:
        st.error(f"Option data not found for contract symbol {contract_symbol}.")
        st.stop()

if not legs:
    st.warning("Please add at least one option leg.")
    st.stop()

# Function to calculate P/L for a leg
def calculate_option_pl_leg(leg, stock_price_range):
    option_type = leg['option_type']
    position_type = leg['position_type']
    strike_price = leg['strike_price']
    premium = leg['premium']
    if option_type.lower() == 'call':
        intrinsic_value = np.maximum(stock_price_range - strike_price, 0)
    else:
        intrinsic_value = np.maximum(strike_price - stock_price_range, 0)
    if position_type.lower() == 'long':
        return intrinsic_value - premium
    else:
        return premium - intrinsic_value

# Calculate total P/L across different stock prices
total_pl_values = np.zeros_like(stock_price_range)
for leg in legs:
    pl_values_leg = calculate_option_pl_leg(leg, stock_price_range)
    total_pl_values += pl_values_leg

# Calculate breakeven points
breakeven_indices = np.where(np.diff(np.sign(total_pl_values)))[0]
breakeven_prices = stock_price_range[breakeven_indices]

# Plotting the P/L profile using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_price_range, y=total_pl_values, mode='lines', name='Total P/L'))
fig.update_layout(title='P/L Profile',
                  xaxis_title='Stock Price at Expiration',
                  yaxis_title='Profit / Loss',
                  hovermode='x unified',
                  template='plotly_dark')
fig.add_hline(y=0, line_dash="dash", line_color="gray")

# Add breakeven points to the graph
for price in breakeven_prices:
    fig.add_vline(x=price, line_dash="dot", line_color="yellow", annotation_text=f"Breakeven: {price:.2f}", annotation_position="top left")

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Display time to expiration
time_to_expiration = expiration_datetime - current_datetime
days = time_to_expiration.days
hours, remainder = divmod(time_to_expiration.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
st.sidebar.write(f"**Time to Expiration**: {max(days,0)}d {max(hours,0)}h {max(minutes,0)}m")

# Option Strategy Insights
st.sidebar.markdown("### Strategy Insights")
if len(legs) == 2:
    leg1 = legs[0]
    leg2 = legs[1]
    # Example insight for Bull Call Spread
    if (leg1['option_type'] == 'Call' and leg1['position_type'] == 'Long' and
        leg2['option_type'] == 'Call' and leg2['position_type'] == 'Short' and
        leg1['strike_price'] < leg2['strike_price']):
        st.sidebar.info("💡 This appears to be a Bull Call Spread strategy.")
    # Additional strategy checks can be added here
else:
    st.sidebar.write("Add more option legs to form a strategy.")

# Function to format options chain data
def format_options_chain(df, option_type, current_price):
    # Rename columns and format "Last Trade Date" in Eastern Time
    df = df.rename(columns={
        "contractSymbol": "Contract Symbol",
        "lastTradeDate": "Last Trade Date (EST)"
    })
    # Convert "Last Trade Date" to EST and reformat
    df["Last Trade Date (EST)"] = pd.to_datetime(df["Last Trade Date (EST)"]).dt.tz_convert(timezone("America/New_York"))
    df["Last Trade Date (EST)"] = df["Last Trade Date (EST)"].dt.strftime('%m/%d/%Y %I:%M %p')

    # Highlight in-the-money options based on option type
    def highlight_in_the_money(val):
        if option_type.lower() == 'call' and val < current_price:
            return 'background-color: #c6efce; color: #006100'
        elif option_type.lower() == 'put' and val > current_price:
            return 'background-color: #ffc7ce; color: #9c0006'
        return ''

    # Apply formatting to the DataFrame
    styled_df = df.style.format({
        'strike': "{:.2f}",
        'lastPrice': "{:.2f}",
        'bid': "{:.2f}",
        'ask': "{:.2f}",
        'volume': "{:.0f}",
        'openInterest': "{:.0f}",
        'impliedVolatility': "{:.2%}"
    }).applymap(highlight_in_the_money, subset=['strike'])

    return styled_df

# Display options chain with selection
st.subheader("Live Options Chain")

# Format calls and puts tables
calls_table = format_options_chain(calls_df, 'call', current_price)
puts_table = format_options_chain(puts_df, 'put', current_price)

# Display formatted Calls table
st.write("**📊 Calls**")
st.dataframe(calls_table, height=300)

# Display formatted Puts table
st.write("**📉 Puts**")
st.dataframe(puts_table, height=300)
