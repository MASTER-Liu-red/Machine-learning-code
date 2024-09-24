
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
import optuna

def SVR_objective(trial):
    _degree = trial.suggest_int("degree", 1, 10)
    _C = trial.suggest_int("C", 1,5)
    _gamma = trial.suggest_float("gamma", 0.00001, 1)
    _tol = trial.suggest_float("tol",0.00001,1)
    _epsilon = trial.suggest_float("epsilon",0.0001,1)
    _max_iter = trial.suggest_int("max_iter",50,500)

    SVRr = SVR(
        degree=_degree,
        gamma=_gamma,
        C=_C,
        tol=_tol,
        epsilon=_epsilon,
        max_iter=_max_iter,

    )

    SVRr.fit(X_train, y_train)
    score = np.sqrt(metrics.mean_squared_error(y_train, SVRr.predict(X_train)))
    return score

# Generate some random data for demonstration
np.random.seed(60)

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
study.optimize(SVR_objective, n_trials=200)


print(study.best_params)  # 输出最佳的超参数组合
print(study.best_value)