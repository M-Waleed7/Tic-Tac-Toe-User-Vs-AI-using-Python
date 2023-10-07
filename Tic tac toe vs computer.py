import ccxt
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialize the Binance exchange object
binance = ccxt.binance()

# Retrieve historical price data for BTC in USD
btc_ohlcv = binance.fetch_ohlcv('BTC/USDT')

# Extract the close prices and convert to a numpy array
close_prices = np.array([x[4] for x in btc_ohlcv])

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the historical data
X = np.array(range(len(close_prices))).reshape(-1, 1)
model.fit(X, close_prices)

# Predict the next price
future_price = model.predict(np.array([len(close_prices)]).reshape(-1, 1))

print("The predicted future price of BTC is:", future_price[0], "USD")
