
import numpy
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

beg=time.time()
# Generate some random data for demonstration
np.random.seed(60)

# Generate some random data for demonstration
data=pd.read_excel("database.xlsx")
print(data)
#Input feature X (from 0, second column and beyond)
X=data.iloc[:,2:]
#Automatic concatenation into descriptors
X=X.to_numpy()
print(X)
Y=data.iloc[:,1]
Y=Y.to_numpy()


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)

rf_params = {'n_estimators': 429, 'min_samples_split': 2, 'min_samples_leaf': 1}
model = RandomForestRegressor( **rf_params)

# predicting
model.fit(X_train, y_train)
y_pred = model.predict(X)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("y_test_pred",y_test_pred)
print("y_test",y_test)

# Calculate the R2 fraction TRAIN
print("TRAIN")
r2 = r2_score(y_train,y_train_pred)
print("R2 score", r2)
mse = mean_squared_error(y_train,y_train_pred)
print("mean squared error:", mse)
mae = mean_absolute_error(y_train,y_train_pred)
print("mean absolute error:", mae)

# Calculate the R2 score TEST
print("TEST")
r2 = r2_score(y_test, y_test_pred)
print("R2 score:", r2)
mse = mean_squared_error(y_test, y_test_pred)
print("mean squared error:", mse)
mae = mean_absolute_error(y_test, y_test_pred)
print("mean absolute error:", mae)

end=time.time()
time=end-beg
print(f"process time: {time} s")
results = []
max_y=1
max_x_values=None

for i in np.arange(25,85,5):
    for j in np.arange(20,80,20):
        for k in np.arange(1.6,2.5,0.2):
            for l in np.arange(21,24.9,0.1):
                for m in np.arange(0.1,1.6,0.1):
                    y_pre_r=model.predict(np.array([[i,j,k,l,m]]))
                    print("parameters", i, j, k, l, m)
                    print("predicting", y_pre_r)
                    result = pd.DataFrame([[i, j, k, l, m, y_pre_r]], columns=['i', 'j', 'k', 'l', 'm', 'predicted_value'])
                    results.append(result)
                    if y_pre_r>max_y:
                        max_y=y_pre_r
                        max_x_values=(i,j,k,l,m)
print("maximum parameter",max_x_values)
print("maximum PCE",max_y)



df = pd.concat(results, ignore_index=True)
df.to_excel('predicted_values_RF_all.xlsx', index=False)




