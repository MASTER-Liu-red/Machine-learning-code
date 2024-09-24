# 导入所需的库
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

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
X=StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)

# #Data standardization
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Create MLP regression models
mlp = MLPRegressor(
    hidden_layer_sizes=(50, 100),
    activation='tanh',
    alpha=0.001,
    learning_rate_init=0.1
    )

# Training model
mlp.fit(X, Y)

#Prediction
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X)
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)

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

#draw picture
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
ax.set_title("MLPRegression performance of the best model (PCE$_{MAX}$)")

'''loc:图例位置， 
   fontsize：字体大小，
   frameon：是否显示图例边框，
   ncol：图例的列的数量，一般为1,
   title:为图例添加标题
   shadow:为图例边框添加阴影,
   markerfirst:True表示图例标签在句柄右侧，false反之，
   markerscale：图例标记为原图标记中的多少倍大小，
   numpoints:表示图例中的句柄上的标记点的个数，一半设为1,
   fancybox:是否将图例框的边角设为圆形
   framealpha:控制图例框的透明度
   borderpad: 图例框内边距
   labelspacing: 图例中条目之间的距离
   handlelength:图例句柄的长度
   bbox_to_anchor: (横向看右，纵向看下),如果要自定义图例位置或者将图例画在坐标外边，用它，比如bbox_to_anchor=(1.4,0.8)，这个一般配合着ax.get_position()，set_position([box.x0, box.y0, box.width*0.8 , box.height])使用
   用不到的参数可以直接去掉,有的参数没写进去，用得到的话加进去     , bbox_to_anchor=(1.11,0)
'''

ax.legend(loc='lower right', fontsize=14,frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
          ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=1.5)
# plt.savefig('E:/机器学习ML/paper-ML/ML图/1.tif', dpi=300, bbox_inches='tight')
plt.show()
