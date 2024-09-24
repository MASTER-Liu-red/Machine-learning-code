
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
import optuna

def xgb_objective(trial):
    _n_estimators = trial.suggest_int("n_estimators", 1, 100)
    _max_depth = trial.suggest_int("max_depth", 1, 20)
    _learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    _gamma = trial.suggest_float("gamma", 0.01, 1)
    _min_child_weight = trial.suggest_float("min_child_weight", 0.1, 10)
    _subsample = trial.suggest_float('subsample', 0.01, 1)
    _reg_alpha = trial.suggest_float('reg_alpha', 0.01, 10)
    _reg_lambda = trial.suggest_float('reg_lambda', 0.01, 10)

    xgbr = xgb.XGBRegressor(
        n_estimators=_n_estimators,
        max_depth=_max_depth,
        learning_rate=_learning_rate,
        gamma=_gamma,
        min_child_weight=_min_child_weight,
        subsample=_subsample,
        reg_alpha=_reg_alpha,
        reg_lambda=_reg_lambda,
        random_state=1,
    )

    xgbr.fit(X_train, y_train)
    score = np.sqrt(metrics.mean_squared_error(y_train, xgbr.predict(X_train)))
    return score



# Generate some random data for demonstration
np.random.seed(1)

# Read data
data=pd.read_excel("database.xlsx")
print(data)
# Input feature X
X=data.iloc[:,2:]

X=X.to_numpy()
print(X)
Y=data.iloc[:,1]
Y=Y.to_numpy()
# print(Y)
#Data standardization (may improve model performance, can be skipped)
X=StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
study = optuna.create_study()
study.optimize(xgb_objective, n_trials=200)

print(study.best_params)  # Output the best combination of hyperparameters
print(study.best_value)

