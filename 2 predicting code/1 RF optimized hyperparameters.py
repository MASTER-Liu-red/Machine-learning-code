
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
import optuna

def rf_objective(trial):
    _n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    _max_depth = trial.suggest_int('max_depth', 1, 10)
    _min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    _min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    _min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0, 0.5)
    _min_impurity_decrease = trial.suggest_int('min_impurity_decrease', 0, 5)


    rfr =RandomForestRegressor(
        n_estimators=_n_estimators,
        max_depth=_max_depth,
        min_samples_split=_min_samples_split,
        min_samples_leaf=_min_samples_leaf,
        min_weight_fraction_leaf=_min_weight_fraction_leaf,
        min_impurity_decrease=_min_impurity_decrease,
    )

    rfr.fit(X_train, y_train)
    score = np.sqrt(metrics.mean_squared_error(y_train, rfr.predict(X_train)))
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
study.optimize(rf_objective, n_trials=200)

# Output the best combination of hyperparameters
print(study.best_params)  
print(study.best_value)