from prophet import Prophet
import streamlit as st
from datetime import date
import yfinance as yf
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# create dates for historical data
start_date = "2018-01-01"
todays_date = date.today().strftime("%Y-%m-%d")

# using streamlit to create title for the application in the UI
st.title("Stock Price Prediction Web Application")

# stocks that will be selectable in the UI
stocks = ("ABBV", "ABT", "ACN", "AMT", "AXP", "BA", "BABA", "BAC", "BHP", "BLK", "BMY", "BP", "BRK-B", "BSX", "BUD", "BX", "C", "CAT", "CB", "CI", "CNI", "COP", "CRM", "CVS", "CVX", "DE", "DEO", "DHR", "DIS", "ELV", "EQNR", "ETN", "FI", "FMX", "GE", "GS", "HCA", "HD", "HDB", "HSBC", "IBM" "IBN", "ITW", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDT", "MMC", "MO", "MRK", "MS", "MUFG", "NEE", "NKE", "NOW", "NVO", "NVS", "ORCL", "PBR-A", "PBR", "PFE", "PG", "PLD", "PM", "RIO", "RTX", "RY", "SAP", "SCHW", "SHEL", "SHOP", "SLB", "SO", "SONY", "SPGI", "SYK", "T", "T", "TD", "TJX", "TM", "TMO", "TSM", "TTE", "UBER", "UL", "UNH", "UNP", "V", "VZ", "WFC", "WMT", "XOM", "ZTS")
stock_selected = st.selectbox("Select the stock you would like to analyze", stocks)

# a slider that allows the user to select years of prediction
years_of_prediction = st.slider("Years of prediction:", 1, 4)
time_period = years_of_prediction * 365

# uses yfinance to get the data of specific ticker
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start_date, todays_date)
    data.reset_index(inplace=True)
    return data

data = load_data(stock_selected)

st.subheader('Raw data for: ' + stock_selected)
st.write(data.tail())

# creates a scatter chart that uses the selected stocks data
def raw_data_plot_scatter():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    figure.layout.update(title_text='Scatter Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

# creates a rsi chart that uses the selected stocks data
def raw_data_plot_rsi():
    figure = go.Figure()
    figure.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']))
    figure.layout.update(title_text='RSI candlestick chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

raw_data_plot_scatter()
raw_data_plot_rsi()

# Forecasting
train = data[['Date', 'Close']]
train = train.rename(columns={"Date": "ds", "Close": "y"})

# uses prophet library to train the dataframe
m = Prophet()
m.fit(train)
future_data_frame = m.make_future_dataframe(periods=time_period)
stock_forecast = m.predict(future_data_frame)

# visualizes the forecast with a line chart
st.subheader('Forecast data for ' + stock_selected)
st.write(stock_forecast.tail())

st.write('forecast data for ' + stock_selected)
figure1 = plot_plotly(m, stock_forecast)
st.plotly_chart(figure1)

st.write('forecast component for - Line Charts' + stock_selected)
figure2 = m.plot_components(stock_forecast)
st.write(figure2)