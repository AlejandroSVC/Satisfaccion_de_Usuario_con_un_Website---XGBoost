# Website user satisfaction
Predicting user satisfaction using xgBoost Regression

### XGBOOST Machine Learning model
XGBoost stands for eXtreme Gradient Boosting. It is focused on
computational speed and model performance. 

### Python code:

### Set working directory and load data

import os
import pandas as pd

os.chdir('C:/Users/Alejandro/Documents/')

df = pd.read_csv('website365.csv')

df.info()
### Import libraries

import xgboost as xgb

import numpy as np

import seaborn as sns

from numpy import asarray

from numpy import mean

from numpy import std

from sklearn.datasets import make_regression

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedKFold

from matplotlib import pyplot

### Display correlation matrix

sns.heatmap(df.corr(), cmap='coolwarm')
### Dataset: extract features and target

X = df.drop('Satisfaction',axis=1)
y = df['Satisfaction']
### Split the data into Train and Test datasets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                          shuffle=False,
                                                          random_state = 1234)
### Initialize the XGBoost regressor

model = XGBRegressor(n_estimators=100, random_state=42)

### Fit the xgBoost model to the data

model.fit(X_train, y_train)

### Make predictions on the test set

y_pred = model.predict(X_test)

### Calculate evaluation metrics

from sklearn.metrics import root_mean_squared_error

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

mape = mean_absolute_percentage_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

explained_var = explained_variance_score(y_test, y_pred)

### Print the evaluation metrics

print("MAPE:", mape)

print("Mean squared error:", mse)

print("Root mean squared error:", rmse)

print("Mean absolute error:", mae)

print("R-squared:", r2)

print("Explained variance:", explained_var)

### Feature importance

importance = model.feature_importances_
print(importance)

### Plot feature importance

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

xgb.plot_importance(model, ax=ax, importance_type='gain', grid=False,
                    show_values=True, values_format='{v:.2f}')

plt.title('Feature Importance')

plt.show()

### Plot actual vs predicted values, and actual vs predicted residuals

import matplotlib.pyplot as plt

from sklearn.metrics import PredictionErrorDisplay

from sklearn.pipeline import make_pipeline

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)

X = rng.random(size=(200, 2)) * 10

y = X[:, 0]**2 + 5 * X[:, 1] + 10 + rng.normal(loc=0.0, scale=0.1, size=(200,))

reg = make_pipeline(StandardScaler(), SVR(kernel='linear', C=10))

reg.fit(X, y)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

PredictionErrorDisplay.from_estimator(reg, X, y, ax=axes[0], kind="actual_vs_predicted")

PredictionErrorDisplay.from_estimator(reg, X, y, ax=axes[1], kind="residual_vs_predicted")

plt.show()
