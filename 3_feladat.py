import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_ewma_weights(decay_factor, window_size):
    weights = np.zeros(window_size)
    weights[0] = 1.0
    for i in range(1, window_size):
        weights[i] = weights[i-1] * decay_factor
    return weights / np.sum(weights)


decay_factor_1 = 0.94
window_size_1 = 100
ewma_weights_1 = calculate_ewma_weights(decay_factor_1, window_size_1)


decay_factor_2 = 0.97
window_size_2 = 252
ewma_weights_2 = calculate_ewma_weights(decay_factor_2, window_size_2)

# Plotting
lag_1 = np.arange(window_size_1)
lag_2 = np.arange(window_size_2)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(lag_1, ewma_weights_1, marker='o')
plt.xlabel('Days')
plt.ylabel('Weight')
plt.title('EWMA Weights for Decay Factor 0.94 and Window Size 100')
plt.ylim(0, 0.061)
plt.xlim(0, window_size_1)

plt.subplot(1, 2, 2)
plt.plot(lag_2, ewma_weights_2, marker='o')
plt.xlabel('Days')
plt.ylabel('Weight')
plt.title('EWMA Weights for Decay Factor 0.97 and Window Size 252')
plt.ylim(0, 0.061)
plt.xlim(0, window_size_2)

plt.tight_layout()
plt.show()



data = pd.read_csv("VOO.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate daily log returns
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

def ewma_forecast(data, decay_factor, window_size):
    variance_forecast = []
    volatility_forecast = []
    ewma_weights = calculate_ewma_weights(decay_factor, window_size)

    for i in range(window_size, len(data)):
        returns = data['Log_Returns'].values[i - window_size:i]
        weighted_returns = ewma_weights * returns
        variance = np.sum(weighted_returns ** 2)
        volatility = np.sqrt(variance)

        variance_forecast.append(variance)
        volatility_forecast.append(volatility)

    return variance_forecast, volatility_forecast


def calculate_ewma_variance(data, decay_factor, window_size):
    variance_forecast = []
    ewma_weights = calculate_ewma_weights(decay_factor, window_size)

    for i in range(window_size, len(data)):
        returns = data['Log_Returns'].values[i - window_size:i]
        weighted_returns = ewma_weights * returns
        variance = np.sum(weighted_returns ** 2)

        variance_forecast.append(variance)

    return variance_forecast


decay_factor_1 = 0.94
window_size_1 = 100
variance_forecast_1 = calculate_ewma_variance(data, decay_factor_1, window_size_1)


decay_factor_2 = 0.97
window_size_2 = 100
variance_forecast_2 = calculate_ewma_variance(data, decay_factor_2, window_size_2)

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(data.index[window_size_1:], variance_forecast_1)
plt.xlabel('Time')
plt.ylabel('Variance')
plt.title('EWMA Variance Forecast (Decay Factor 0.94)')
plt.ylim(0, 0.015)


plt.subplot(1, 2, 2)
plt.plot(data.index[window_size_2:], variance_forecast_2)
plt.xlabel('Time')
plt.ylabel('Variance')
plt.title('EWMA Variance Forecast (Decay Factor 0.97)')
plt.ylim(0, 0.015)


plt.tight_layout()
plt.show()
