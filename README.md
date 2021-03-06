# Basic Stock Web Application

This application uses Streamlit to simply create a web application that allows the user to view any chosen S&P500 Stock symbol. The user can view the stock data in a 'heatmapped' Pandas Dataframe over various different time periods, broken up into various different time intervals. 

Additionally, a candlestick graph is given that updates dynamically as user input changes (symbol, period, interval) and provides all standard plotly features (resizing, fullscreen, hovermodes etc.)

Lastly; latest news articles pertaining to the user chosen stock symbol are collected and placed in a dataframe to read, and a compound score of sentiment analysis is given. 

Imports:
- Streamlit
- Pandas for use of dataframes and easy data manipulation
- yahoo finance API for up to date S&P500 stock prices
- plotly for graphing stock prices using candlestick plotting
- BeautifulSoup for parsin stock symbol latest headlines from 'finviz'
- nltk / vader for sentiment analysis on latest headlines scraped. 

## The Application
![stock-web-app image](./appImgs/stockwebapp.png)



