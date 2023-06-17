
import pandas as pd
import numpy as np

# 1.feladat


def calc_historical_var(df_portfolio_returns, l_conf_levels):
    df_portfolio_returns = pd.DataFrame(df_portfolio_returns)  # Convert to DataFrame
    l_quantiles = [1 - x for x in l_conf_levels]
    df_result = df_portfolio_returns.quantile(l_quantiles)
    df_result.index = l_conf_levels
    return df_result


confidence_levels = [0.95]
x = calc_historical_var(np.arange(-0.05, 0.06, 0.01), confidence_levels)
print(x)


# 2.feladat


def calculate_covariance_matrix(std_x, std_y, corr):
    cov_xy = std_x * std_y * corr
    cov_matrix = np.array([[std_x ** 2, cov_xy], [cov_xy, std_y**2]])
    return cov_matrix


def calc_asset_returns(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)


def simulated_returns(expected_return, volatility, correlation, num_of_simulations):
    cov_matrix = calculate_covariance_matrix(volatility[0], volatility[1], correlation)
    return calc_asset_returns(expected_return, cov_matrix, num_of_simulations)


# 3.feladat


def calculate_ewma_weights(decay_factor, window_size):
    weights = np.zeros(window_size)
    weights[0] = 1.0
    for i in range(1, window_size):
        weights[i] = weights[i-1] * decay_factor
    return weights / np.sum(weights)


def calculate_ewma_variance(data, decay_factor, window_size):
    variance_forecast = []
    ewma_weights = calculate_ewma_weights(decay_factor, window_size)

    for i in range(window_size, len(data)):
        returns = data['Log_Returns'].values[i - window_size:i]
        weighted_returns = ewma_weights * returns
        variance = np.sum(weighted_returns ** 2)

        variance_forecast.append(variance)

    return variance_forecast
