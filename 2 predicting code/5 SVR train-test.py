# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 生成一些随机数据用于演示
np.random.seed(60)

#读取数据表格
data=pd.read_excel("1,1.3,1.6,1.8,2and2.5-4.xlsx")
print(data)
#输入特征X（从0计，第二列及以后）
X=data.iloc[:,2:]
#自动拼接成描述符
X=X.to_numpy()
# print(X)
Y=data.iloc[:,1]
Y=Y.to_numpy()
# print(Y)
#数据标准化（可能提高模型性能，注释该部分代码可以跳过）
X=StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)

SVR_params = {'degree': 7, 'C': 5, 'gamma': 0.9997693990409532, 'tol': 0.6275278738290937, 'epsilon': 0.1374597917625562, 'max_iter': 103
}
model = SVR( **SVR_params)

# 进行预测
model.fit(X_train, y_train)
y_pred = model.predict(X)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("y_test_pred",y_test_pred)
print("y_test",y_test)

# 计算R2分数TRAIN
print("TRAIN")
r2 = r2_score(y_train,y_train_pred)
print("R2分数:", r2)
mse = mean_squared_error(y_train,y_train_pred)
print("mean squared error:", mse)
mae = mean_absolute_error(y_train,y_train_pred)
print("mean absolute error:", mae)

# 计算R2分数TEST
print("TEST")
r2 = r2_score(y_test, y_test_pred)
print("R2分数:", r2)
mse = mean_squared_error(y_test, y_test_pred)
print("mean squared error:", mse)
mae = mean_absolute_error(y_test, y_test_pred)
print("mean absolute error:", mae)

# 绘制
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

#绘制预测-真值对散点
ax.scatter(y_train,y_train_pred,color="cornflowerblue",label='Train',alpha=0.5)
ax.scatter(y_test,y_test_pred,color="mediumorchid",label='Test',alpha=0.5)
#绘制对角线
ax.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linestyle='--')  # 绘制y=x参考线
ax.set_ylabel("Predicted PCE (%)",fontsize=16)
ax.set_xlabel("Experiment PCE (%)",fontsize=16)
ax.set_title("SVR performance of the best model (PCE$_{MAX}$)")
# 显示图例
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

