"""
基于树的学习算法是十分流行且应用广泛的一类非参数化的有监督学习算法。
优点：可解释性强，直观。
扩展为随机森林和堆叠（stacking）模型。既可用于回归有可用于分类
训练，处理，调整，可视化， 评估
"""

"""
问题描述： 使用决策树训练回归模型
解决方案： 使用scikit-learn中的DecisionTreeRegressor
"""

#加载库
from sklearn.tree import  DecisionTreeRegressor
from sklearn import  datasets
#加载数据
iris = datasets.load_iris()
features = iris.data
print(features[1,:])
target = iris.target
#创建决策树分类器对象
decisiontree = DecisionTreeRegressor(random_state=0)
#训练模型
model = decisiontree.fit(features,target)

"""
#使用训练好的模型来预测一个样本的值
"""
#创建新样本
#observation = [[0,0,2, 16]]
observation = features[1,:].reshape(1,- 1)
print(observation)
#预测样本的值
print(model.predict(observation))
"""
可以用参数criterion来选择分裂质量（split quality）的度量方式
"""
#用平均绝对误差（MAE）的减少量作为分裂标准来构造决策树回归模型
decisiontree_mae = DecisionTreeRegressor(criterion="mae", random_state=0)
#训练模型
model_mae = decisiontree_mae.fit(features,target)

#创建新样本
#observation1 = [[0,0,2, 16]]
"""
#observation1=features[1,:]
ValueError: Expected 2D array, got 1D array instead:
array=[4.9 3.  1.4 0.2].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
"""

observation1=features[1,:].reshape(1,- 1)
#预测样本的值
print(model_mae.predict(observation1))