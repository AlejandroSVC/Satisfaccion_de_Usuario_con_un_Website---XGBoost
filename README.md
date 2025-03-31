# Predicting website user satisfaction
## XGBoost Machine Learning Regression optimized with Optuna
## Python code

![Banner delgado](docs/assets/images/Internet_users.jpg)

### XGBOOST Machine Learning model

XGBoost stands for eXtreme Gradient Boosting. It is focused on
computational speed and model performance. 

### Python code:

### 1. Set working directory and load data
```
import os
os.chdir('C:/Users/Alejandro/Documents/')
import pandas as pd
data = pd.read_csv('website365.csv')
data.info()
```
### 2. Import libraries
```
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
```
### 3. Separate features and target
```
X = data.drop('Satisfaction', axis=1)
y = data['Satisfaction']
```
### 4. Split data into train and test sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 5. Define objective function for Optuna
```
from sklearn.metrics import root_mean_squared_error

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    return rmse
```
### 6. Run Optuna optimization
```
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
```
### 7. Train final model with best hyperparameters
```
best_params = study.best_params
best_params['objective'] = 'reg:squarederror'
best_params['random_state'] = 42

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
```
### 8. Evaluate the model
```
y_pred = final_model.predict(X_test)

print("\nEvaluation Metrics:")
```
### Calculate evaluation metrics
```
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)
```
### Print the evaluation metrics
```
print(f"MAPE, mean absolute percentage error:", round(mape,5))
print(f"MSE, Mean squared error:", round(mse,5))
print(f"RMSE, Root mean squared error:", round(rmse,5))
print(f"MAE, Mean absolute error:", round(mae,5))
print(f"R2, R-squared:", round(r2,5))
print(f"Explained variance:", round(explained_var,5))
```
### 9. Plots

### Plot feature importance
```
feature_importance = final_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.xticks(range(X.shape[1]), sorted_idx)
plt.xlabel('Feature index')
plt.ylabel('Feature importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()
```
![1_Feature_importance](docs/assets/images/1_Feature_importance.png)

### Plot actual vs predicted values
```
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
```
![2_Actual_vs_predicted_values](docs/assets/images/2_Actual_vs_predicted_values.png)

### Plot residuals
```
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```
![3_Residual_plot](docs/assets/images/3_Residual_plot.png)

### Plot optimization history
```
optuna.visualization.plot_optimization_history(study).show()
```
![4_Optimization_history_plot](docs/assets/images/4_Optimization_history_plot.png)

### Plot parameter importance
```
optuna.visualization.plot_param_importances(study).show()
```
![5_Parameter_importance_plot](docs/assets/images/5_Parameter_importance_plot.png)

