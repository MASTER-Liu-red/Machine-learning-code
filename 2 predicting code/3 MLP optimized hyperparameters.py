import numpy as np
import optuna
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# Generate some random data for demonstration
np.random.seed(1)

# Read data
data=pd.read_excel("database.xlsx")
print(data)

# Input feature X
X=data.iloc[:,2:]

X=X.to_numpy()
print(X)
y=data.iloc[:,1]
y=y.to_numpy()
# print(y)
# #Data standardization (may improve model performance, can be skipped)
# X=StandardScaler().fit_transform(X)

import warnings
warnings.filterwarnings("ignore")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# study = optuna.create_study()
# study.optimize(mlp_objective, n_trials=200)

# MLP model
mlp = MLPRegressor()

# Set the hyperparameter range
tuned_parameters = {
    'hidden_layer_sizes': [(50,50), (100,100), (50,100),(50,50,100)],
    'activation': ['logistic', 'tanh', 'relu'],
    'alpha': np.logspace(-4, 0, num=5),
    'learning_rate_init': np.logspace(-4, 0, num=5),

}

# Random search for hyperparameter optimization
clf = GridSearchCV(mlp,tuned_parameters,
                   scoring="neg_mean_absolute_error", cv=5)
#clf.fit(X,y)
clf.fit(X_train, y_train)

# Output optimal hyperparameter combinations and scores
print("Best Parameters:", clf.best_params_)
print("Best Score:", clf.best_score_)

# 获取随机搜索结果
means = clf.cv_results_

# Extract learning rate and score
learning_rates = means['param_learning_rate_init']
scores = means['mean_test_score']

# Output Results (MAE)
print("The optimal model for the current data set is set:")
print(clf.best_params_)
print("score：")
print("%0.3f"%(clf.best_score_))
print()
print("Setting of each hyperparameter and corresponding model score (score /± SD)")
means=-clf.cv_results_['mean_test_score']
stds=clf.cv_results_['std_test_score']
for mean,std,params in zip(means,stds,clf.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2,params))

# # Visualize the relationship between learning rate and scoring
# plt.figure(figsize=(10, 6))
# plt.scatter(learning_rates, scores,color="r",)
# plt.xscale('log')
# plt.xlabel('Learning Rate')
# plt.ylabel('Mean Test Score')
# plt.title('Learning Rate vs Mean Test Score')
# plt.show()