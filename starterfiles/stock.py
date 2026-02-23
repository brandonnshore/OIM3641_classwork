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
2. If yes, what was your primary prompt?
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
        # TODO: Use yf.download(self.symbol, start=self.start, end=self.end)
        # data = ...

        # self.calc_returns(data)
        # return data
        pass

    def calc_returns(self, df):
        """Adds 'Change', close to close and 'Instant_Return' columns to the dataframe."""
        # Requirement: Use vectorized pandas operations, not loops.
        pass
    
    def add_technical_indicators(self, windows=[20, 50]):
        """
        Add Simple Moving Averages (SMA) for the given windows
        to the internal DataFrame. Produce a plot showing the closing price and SMAs. 
        """
        pass

    def plot_performance(self):
        """Plots cumulative growth of $1 investment."""
        pass


def main():
    # Example usage:
    # aapl = Stock("AAPL")
    # aapl.plot_performance()
    # appl.add_technical_indicators()
    pass


if __name__ == "__main__":
    main()