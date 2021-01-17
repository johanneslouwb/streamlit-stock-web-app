import streamlit as st
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import plotly.graph_objs as go
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
st.set_page_config(layout='wide')

def convert_ip_yf_period(input):
    d_per = {'days': 'd', 'weeks': 'wk', 'months': 'mo', 'years': 'y'}
    return d_per[input]


def convert_ip_yf_interval(input):
    d_int = {
        '1 minute': '1m', '2 minutes': '2m', '5 minutes': '5m', '15 minutes': '15m',
        '30 minutes': '30m', '90 minutes': '90m', '1 hour': '1h', '1 day': '1d',
        '5 days': '5d', '1 week': '1wk', '1 month': '1mo', '3 months': '3mo'
    }
    return d_int[input]


def get_input():
    stock_symbol = st.sidebar.text_input('Stock Symbol', 'AAPL')
    period_num = st.sidebar.number_input('Numeric Period', min_value=1, step=1)
    period_len = st.sidebar.selectbox('Period Length', 
        options=('days', 'weeks', 'months', 'years'),
        index=3  
    )
    
    interval_len = st.sidebar.selectbox('Interval Length', options=(
        '1 minute', '2 minutes', '5 minutes', '15 minutes', '30 minutes', '90 minutes',
        '1 hour', '1 day', '5 days', '1 week', '1 month', '3 months'
        ),
        index=7
    )

    period = str(f'{period_num}'+f'{convert_ip_yf_period(period_len)}')
    interval = str(convert_ip_yf_interval(interval_len))
    return [stock_symbol, period, interval]


def get_data(symbol, period, interval):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    return data


def get_company_name(symbol):
    try:
        name = constituents.loc[symbol]['Name']
    except:
        name = symbol
    return name

st.sidebar.header('Preferences : Choose Period, Interval and Stock Symbol you would like to investigate')
inpt = get_input()
data = get_data(symbol=inpt[0], period=inpt[1], interval=inpt[2])

constituents = pd.read_csv('data/constituents_csv.csv', index_col='Symbol')
comp_name = get_company_name(inpt[0])

# ------------------------------ SENTIMENT ANALYSIS --------------------------------------
news_tables = {}
finviz_url = 'https://finviz.com/quote.ashx?t='+inpt[0]
req = Request(url=finviz_url, headers={'user-agent': 'my-app'})
response = urlopen(req)

html = BeautifulSoup(response, 'html.parser')
news = html.find(id='news-table')
news_tables[inpt[0]] = news
rows = news_tables[inpt[0]].findAll('tr')

list_news = []
for row in rows:
    list_news.append(row.a.text)
df_news = pd.DataFrame(list_news, columns=['Recent Article'])

vader = SentimentIntensityAnalyzer()
f_pol = lambda title: vader.polarity_scores(title)['compound']
df_news['compound'] = df_news['Recent Article'].apply(f_pol)
compound_mean = df_news['compound'].mean()


# --------------------------- LAYOUT ------------------------------------------------


st.write(f'# Stock Market Data for {comp_name}')
# -------------------------- PLOTLY GRAPHING
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=data.index, 
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='market data'
    )
)
# titles:
fig.update_layout(
    title='Share Price Over User Chosen Period/Interval : Graph',
    yaxis_title='Stock Price (USD / Share)'            
)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.beta_columns(2)

with col1:
    st.write('**Stock Price Colormap** : green=High, red=Low (relative to stock evolution)')
    st.dataframe(data.style.background_gradient(cmap='RdYlGn')) 

with col2:
    st.write(f"""
        **Most Recent Headlines for {comp_name}** : Compound Scores from Sentiment Analysis: 
        **mean: {round(compound_mean, 6)}**
    """)
    st.dataframe(df_news.style.set_properties(**{'text-align': 'left'}))



