import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
import optuna

def knn_objective(trial):
    _n_neighbors = trial.suggest_int("n_neighbors", 2, 10)
    _leaf_size = trial.suggest_int("leaf_size", 1, 100)
    _p = trial.suggest_int("p", 1, 10)


    knnr = KNeighborsRegressor(
        n_neighbors=_n_neighbors,
        leaf_size=_leaf_size,
        p=_p,
    )

    knnr.fit(X_train, y_train)
    score = np.sqrt(metrics.mean_squared_error(y_train, knnr.predict(X_train)))
    return score

# Generate some random data for demonstration
np.random.seed(60)

# Read data
data=pd.read_excel("database.xlsx")
print(data)

X=data.iloc[:,2:]

#Formation descriptor
X=X.to_numpy()
print(X)
Y=data.iloc[:,1]
Y=Y.to_numpy()
# print(Y)

#Data standardization (may improve model performance, can be skipped)
X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
study = optuna.create_study()
study.optimize(knn_objective, n_trials=200)

# Output the best combination of hyperparameters
print(study.best_params)  
print(study.best_value)