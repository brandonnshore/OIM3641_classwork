import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sb
import yfinance as yf  # Assuming yfinance for data API

sb.set_theme()

"""
STUDENT CHANGE LOG & AI DISCLOSURE:
----------------------------------
1. Did you use an LLM (ChatGPT/Claude/etc.)? [Yes/No]
Yes I tried to do as much as I could myself and using snippets of code from other classes, 
but then i fed it to gemini and asked its opinion. the main change it made is it said we should have 3 graphs instead of two and added
the entire pllot return dist at the bottom. I am unsure if that is too much or okay but I figured I would rather keep it.
2. If yes, what was your primary prompt?
"You are a developer whos only coding language is python and you write it the most efficent and simpified code, with no grandeur or LLM style writing, 
please take this code and clean it up, ensure it works, and make any changes you feel are ABSOULTLEY needed"
----------------------------------
"""

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(365))
DEFAULT_END = dt.date.isoformat(dt.date.today())


class Stock:
    def __init__(self, symbol, start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """Downloads data from yfinance and triggers return calculation."""
        data = yf.download(self.symbol, start=self.start, end=self.end)
        data.columns = data.columns.get_level_values(0)
        data.index = pd.to_datetime(data.index)
        self.calc_returns(data)
        return data

    def calc_returns(self, df):
        """Adds 'Change', close to close and 'Instant_Return' columns to the dataframe."""
        df['Change'] = df['Close'].pct_change()
        df['Instant_Return'] = np.log(df['Close']).diff().round(4)
    
    def add_technical_indicators(self, windows=[20, 50]):
        """
        Add Simple Moving Averages (SMA) for the given windows
        to the internal DataFrame. Produce a plot showing the closing price and SMAs. 
        """
        for w in windows:
            self.data['SMA_' + str(w)] = self.data['Close'].rolling(window=w).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(self.data['Close'], label='Close')
        for w in windows:
            plt.plot(self.data['SMA_' + str(w)], label='SMA ' + str(w))
        plt.title(self.symbol + ' - Close with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_performance(self):
        """Plots cumulative growth of $1 investment."""
        cumulative = (1 + self.data['Change']).cumprod()
        pct_gain = (cumulative - 1) * 100

        plt.figure(figsize=(10, 6))
        plt.plot(pct_gain)
        plt.title(self.symbol + ' - Performance (% Gain/Loss)')
        plt.xlabel('Date')
        plt.ylabel('% Gain/Loss')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.tight_layout()
        plt.show()

    def plot_return_dist(self):
        """Plots histogram of instantaneous returns."""
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['Instant_Return'].dropna(), bins=50)
        plt.title(self.symbol + ' - Distribution of Instant Returns')
        plt.xlabel('Instant Return')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


def main():
    aapl = Stock("AAPL")
    print(aapl.data)
    aapl.plot_performance()
    aapl.plot_return_dist()
    aapl.add_technical_indicators()

if __name__ == "__main__":
    main()