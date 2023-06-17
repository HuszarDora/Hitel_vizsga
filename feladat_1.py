import pandas as pd
import numpy as np


def get_portfolio_return(d_weights):

    df_voo = pd.read_csv('voo.csv')
    df_spy = pd.read_csv('spy.csv')

    df_voo['Return'] = df_voo['Adj Close'] / df_voo['Adj Close'].shift(1) - 1
    df_spy['Return'] = df_spy['Adj Close'] / df_spy['Adj Close'].shift(1) - 1

    df_combined = pd.merge(df_voo[['Date', 'Return']], df_spy[['Date', 'Return']], on='Date', suffixes=('_voo', '_spy'))

    df_combined['Portfolio Return'] = d_weights['voo'] * df_combined['Return_voo'] + d_weights['spy'] * df_combined[
        'Return_spy']

    df_combined.dropna(inplace=True)

    return df_combined['Portfolio Return']


def calc_historical_var(df_weights, l_conf_levels):
    l_quantiles = [1 - x for x in l_conf_levels]
    df_result = pd.DataFrame(columns=['Weight_voo', 'Weight_spy'] + l_conf_levels)

    for index, weights in df_weights.iterrows():
        df_pf = get_portfolio_return(weights)
        df_result.loc[index, 'Weight_voo'] = weights['voo']
        df_result.loc[index, 'Weight_spy'] = weights['spy']
        df_result.loc[index, l_conf_levels] = np.abs(df_pf.quantile(l_quantiles).values)

    return df_result


weight_range = np.arange(0, 1.1, 0.1)
df_weights = pd.DataFrame({
    'voo': weight_range,
    'spy': 1 - weight_range,
})
confidence_levels = [0.95]

var_result = calc_historical_var(df_weights, confidence_levels)
print(var_result)