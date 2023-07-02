import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the stock ticker symbol and date range
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2021-12-31'

# Fetch the historical stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Clean the data
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()  # Keep only relevant columns and remove missing values

# Calculate additional features
data['Return'] = data['Close'].pct_change()  # Calculate daily returns
data['LogReturn'] = np.log(1 + data['Return'])  # Calculate log returns
data['Volatility'] = data['LogReturn'].rolling(window=252).std() * np.sqrt(252)  # Calculate annualized volatility

# Perform exploratory data analysis
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
data['Close'].plot(ax=axes[0])
axes[0].set_ylabel('Price')
axes[0].set_title('Stock Price Analysis - {} ({} to {})'.format(ticker, start_date, end_date))

sns.histplot(data['Return'].dropna(), ax=axes[1], kde=True)
axes[1].set_xlabel('Daily Returns')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Daily Returns')

plt.tight_layout()
plt.show()

# Calculate basic statistics
mean_return = data['Return'].mean()
std_return = data['Return'].std()
annualized_return = (1 + mean_return) ** 252 - 1  # Assuming 252 trading days in a year
annualized_volatility = data['Volatility'][-1]

# Print the results
print('Stock: {}'.format(ticker))
print('Period: {} to {}'.format(start_date, end_date))
print('Mean Daily Return: {:.4f}'.format(mean_return))
print('Standard Deviation of Daily Return: {:.4f}'.format(std_return))
print('Annualized Return: {:.4f}'.format(annualized_return))
print('Annualized Volatility: {:.4f}'.format(annualized_volatility))
