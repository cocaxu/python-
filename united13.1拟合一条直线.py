#问题描述： 训练一个能表示特征和目标向量之间线性关系的模型
#使用线性回归算法（scikit-learn LinearRegression）

#加载库
from sklearn.linear_model import  LinearRegression
from  sklearn.datasets import  load_boston
#加载只有两个特征的数据集
boston = load_boston()
features= boston.data[:,0:2]
target =  boston.target

#创建线性回归对象
regression = LinearRegression()
#若在LinearRegression中不加（），那么出现如下错误：
#TypeError: fit() missing 1 required positional argument: 'y'
#拟合线性回归模型
model = regression.fit(features, target)


#查看截距
print(model.intercept_)
#显示特征权重
print(model.coef_)
#目标向量的第一个值
print(target[0]*1000)
#预测第一个样本的目标值，并乘以1000(target的单位是千美元，故而要乘以1000）
print(model.predict(features)[0]*1000)
#预测误差
error= target[0]*1000-model.predict(features)[0]*1000
print(error)
#可解释性