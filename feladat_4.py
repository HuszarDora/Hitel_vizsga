import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


data = pd.read_csv('VOO.csv')


data['Returns'] = data['Adj Close']/data['Adj Close'].shift(1)-1
data['Squared Returns'] = data['Returns']**2


lags = 20
for lag in range(1, lags+1):
    data[f'Lagged Squared Returns {lag}'] = data['Squared Returns'].shift(lag)


data.dropna(inplace=True)


X = data[[f'Lagged Squared Returns {lag}' for lag in range(1, lags+1)]]
y = data['Squared Returns']


tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)


mean_mse = np.mean(mse_scores)

print(f"Mean Squared Error (MSE): {mean_mse}")
