# Initial imports
import pandas as pd
import numpy as np
import seaborn as sns


# Reading whale returns
whale_returns = pd.read_csv("Resources/whale_returns.csv", index_col="Date", parse_dates=True, infer_datetime_format=True)
whale_returns.sort_index(inplace = True)

# Drop nulls
whale_returns.dropna(inplace=True)


# Reading algorithmic returns
algo_returns = pd.read_csv("Resources/algo_returns.csv", index_col="Date", parse_dates=True, infer_datetime_format=True)
algo_returns.sort_index(inplace=True)


# Drop nulls
algo_returns.dropna(inplace=True)
algo_returns.isnull().sum()

# Reading S&P 500 Closing Prices
sp500_history = pd.read_csv("Resources/sp500_history.csv", index_col="Date", parse_dates=True, infer_datetime_format=True)
sp500_history.sort_index(inplace = True)


# Fix Data Types
sp500_history["Close"] = sp500_history["Close"].str.replace("$", " ", regex=True).astype(float)
sp500_history.head()

# Calculate Daily Returns
sp500_returns = sp500_history.pct_change()

# Drop nulls
sp500_returns.dropna(inplace=True)
sp500_returns.isnull().sum()

# Rename `Close` Column to be specific to this portfolio.
sp500_returns.columns = ["S&P 500"]

# Join Whale Returns, Algorithmic Returns, and the S&P 500 Returns into a single DataFrame with columns for each portfolio's returns.
portfolio_returns = pd.concat([whale_returns, algo_returns, sp500_returns], axis=1, join='inner')


# Plot daily returns of all portfolios
portfolio_returns.plot(figsize = (20,10))

# Calculate cumulative returns of all portfolios
cumulative_returns = (1 + portfolio_returns).cumprod()

# Plot cumulative returns
cumulative_returns.plot(figsize=(20,10))

# Box plot to visually show risk
portfolio_returns.plot(kind='box', figsize=(20,10))

# Calculate the daily standard deviations of all portfolios
portfolio_std = pd.DataFrame(portfolio_returns.std())
portfolio_std.columns=['STD']

# Calculate  the daily standard deviation of S&P 500
sp500_std = portfolio_std['STD']['S&P 500']

# Determine which portfolios are riskier than the S&P 500
print(portfolio_std.loc[portfolio_std['STD']>sp500_std])

# Calculate the annualized standard deviation (252 trading days)
ann_std = portfolio_std * np.sqrt(252)
ann_std.columns = ['Annualized STD']

# Calculate the rolling standard deviation for all portfolios using a 21-day window
portfolio_rolling_std = portfolio_returns.rolling(window=21).std()

# Plot the rolling standard deviation
portfolio_rolling_std['S&P 500'].plot(title='21-Day Rolling', figsize=(20,10))

# Calculate the correlation
correlation = portfolio_returns.corr()
correlation

# Display de correlation matrix
sns.heatmap(correlation, vmin=-1, vmax=1)

# Calculate covariance of a single portfolio
rolling_covariance_BH = portfolio_returns['BERKSHIRE HATHAWAY INC'].rolling(window=60).cov(portfolio_returns['S&P 500'])


# Calculate variance of S&P 500
rolling_variance = portfolio_returns['S&P 500'].rolling(window=60).var()


# Computing beta
rolling_beta = rolling_covariance_BH / rolling_variance


# Plot beta trend
rolling_beta.plot(title='Rolling 60-Day Beta of BERKSHIRE HATHAWAY INC', figsize=(20,10))

# Use `ewm` to calculate the rolling window
exponential_ma = portfolio_returns.ewm(halflife=21, adjust=False).mean()

# plot ewm
exponential_ma.plot(figsize=(20,10))


# Annualized Sharpe Ratios
sharpe_ratios = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))

# Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot(kind='bar', title='Sharpe Ratios')

# Reading data from 1st stock
tsla_history = pd.read_csv("Resources/tsla_history.csv", index_col = "Date", parse_dates=True, infer_datetime_format=True)

# Reading data from 2nd stock
roku_history = pd.read_csv("Resources/roku_history.csv", index_col = "Date", parse_dates=True, infer_datetime_format=True)

# Reading data from 3rd stock
fb_history = pd.read_csv("Resources/fb_history.csv", index_col = "Date", parse_dates=True, infer_datetime_format=True)

# Combine all stocks in a single DataFrame
my_portfolio = pd.concat([tsla_history, roku_history, fb_history], axis=1, join='inner')


# Reset Date index
my_portfolio.index = my_portfolio.index.normalize()


# Reorganize portfolio data by having a column per symbol
my_portfolio.columns=['TSLA', 'ROKU', 'FB']


# Calculate daily returns
daily_returns = my_portfolio.pct_change()

# Drop NAs
daily_returns.dropna(inplace=True)

# Display sample data
daily_returns.head()

# Set weights
weights = [1/3, 1/3, 1/3]

# Calculate portfolio return
my_returns = daily_returns.dot(weights)

# Display sample data
my_returns.head()

# Join your returns DataFrame to the original returns DataFrame
combined_returns = pd.concat([my_returns, portfolio_returns], axis=1, join='inner')
combined_returns.rename(columns={0:"My Portfolio Returns"}, inplace=True) 


# Only compare dates where return data exists for all the stocks (drop NaNs)
combined_returns.dropna(inplace=True)

# Calculate the annualized `std`
combined_ann_std = combined_returns.std() * np.sqrt(252)

# Calculate rolling standard deviation
combined_rolling_std = combined_returns.rolling(window=21).std()

# Plot rolling standard deviation
combined_rolling_std.plot(figsize=(20,10))

# Calculate and plot the correlation
combined_correlation = combined_returns.corr()
sns.heatmap(combined_correlation,vmin=-1, vmax=1)


# Calculate and plot Beta
combined_roll_cov_my_portfolio = combined_returns['My Portfolio Returns'].rolling(window=60).cov(combined_returns['S&P 500'])
combined_roll_var = combined_returns['S&P 500'].rolling(window=60).var()
combined_roll_beta = combined_roll_cov_my_portfolio / combined_roll_var
combined_roll_beta.plot(title='Rolling 60-Day Beta of My Portfolio Returns', figsize=(20,10))


# Calculate Annualzied Sharpe Ratios
combined_sharpe_ratios = (combined_returns.mean() * 252) / (combined_returns.std() * np.sqrt(252))
combined_sharpe_ratios


# Visualize the sharpe ratios as a bar plot
combined_sharpe_ratios.plot(kind='bar', title='Sharpe Ratios')