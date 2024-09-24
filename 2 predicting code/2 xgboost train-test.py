
import numpy as np
import time
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


beg=time.time()
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
# X=StandardScaler().fit_transform(X)

# Convert data to DMatrix format
dtrain = xgb.DMatrix(X, label=Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

xgb_params = {'n_estimators': 25, 'max_depth': 9, 'learning_rate': 0.9669902767182301, 'gamma': 0.05902669654040922, 'min_child_weight': 0.8007187794278572, 'subsample': 0.9201026986317694, 'reg_alpha': 0.011082628765489785, 'reg_lambda': 0.19050485879365056

}
model = xgb.XGBRegressor(random_state=1, **xgb_params)


model.fit(X_train, y_train)

y_pred = model.predict(X)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("y_test_pred",y_test_pred)
print("y_test",y_test)

#Training set R2 fractions
print("TRAIN")
r2 = r2_score(y_train,y_train_pred)
print("R2score:", r2)
mse = mean_squared_error(y_train,y_train_pred)
print("mean squared error:", mse)
mae = mean_absolute_error(y_train,y_train_pred)
print("mean absolute error:", mae)

#Test set R2 fractions
print("TEST")
r2 = r2_score(y_test, y_test_pred)
print("R2score:", r2)
mse = mean_squared_error(y_test, y_test_pred)
print("mean squared error:", mse)
mae = mean_absolute_error(y_test, y_test_pred)
print("mean absolute error:", mae)

end=time.time()
time=end-beg
print(f"process time: {time} s")



#draw
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] ="in"
plt.rcParams["ytick.direction"] ="in"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

w = 10
h = w / 4.85
plt.figure(figsize = (w,h),dpi = 300)
fig,ax=plt.subplots()
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

#Plot prediction - truth against scatter
ax.scatter(y_train,y_train_pred,color="cornflowerblue",label='Train',alpha=0.5)
ax.scatter(y_test,y_test_pred,color="mediumorchid",label='Test',alpha=0.5)
# draw y = x
ax.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linestyle='--')  
ax.set_ylabel("Predicted PCE (%)",fontsize=16)
ax.set_xlabel("Experiment PCE (%)",fontsize=16)
ax.set_title("XGBoost Regression performance of the best model (PCE$_{MAX}$)")


ax.legend(loc='lower right', fontsize=14,frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
          ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=1.5)
# plt.savefig('E:xxx.tif', dpi=300, bbox_inches='tight')
plt.show()


