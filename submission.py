# import feladat_1 as e
# import feladat_2 as k
# import feladat_3 as h
# import feladat_4 as n
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

# 3.feladat
# 4.feladat