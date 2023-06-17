import pandas as pd
import numpy as np

df_voo = pd.read_csv('voo.csv')
df_spy = pd.read_csv('spy.csv')

df_voo['Return'] = df_voo['Adj Close'] / df_voo['Adj Close'].shift(1) - 1
df_spy['Return'] = df_spy['Adj Close'] / df_spy['Adj Close'].shift(1) - 1
df_combined = pd.merge(df_voo[['Date', 'Return']], df_spy[['Date', 'Return']], on='Date', suffixes=('_voo', '_spy'))
df_combined.dropna(inplace=True)

def calculate_covariance_matrix(std_x, std_y, corr):
    cov_xy = std_x * std_y * corr
    cov_matrix = np.array([[std_x ** 2, cov_xy], [cov_xy, std_y**2]])
    return cov_matrix


def calc_asset_returns(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)


def simulated_returns(expected_return, std_voo, std_spy, correlation, num_of_simulations):
    cov_matrix = calculate_covariance_matrix(std_voo, std_spy, correlation)
    return calc_asset_returns(expected_return, cov_matrix, num_of_simulations)


Voo_std = np.std(df_voo['Return'])
Voo_mean = np.mean(df_voo['Return'])
Spy_std = np.std(df_spy['Return'])
Spy_mean = np.mean(df_spy['Return'])
total_volatility = Voo_std + Spy_std
expected_return = [Voo_mean, Spy_mean]
weight_voo = Voo_std / total_volatility
weight_spy = Spy_std / total_volatility


def calc_historical_var(portf_ret, confidence_level):
    sorted_returns = np.sort(portf_ret)
    n = portf_ret.size
    var_index = int(np.floor(n * (1 - confidence_level)))
    var = abs(sorted_returns[var_index])
    return var


nsim = 1000
correlations = [0.2, 0.4, 0.6, 0.8, 1.0]
confidence_level = 0.95

for correlation in correlations:
    price_path = simulated_returns(expected_return, Voo_std, Spy_std, correlation, nsim)
    price_path = np.exp(price_path) - 1
    weighted_portf_return = weight_voo * price_path[:, 0] + weight_spy * price_path[:, 1]

    VAR = calc_historical_var(weighted_portf_return, confidence_level)
    print(f"Correlation: {correlation}, VaR: {VAR}")