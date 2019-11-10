"""
问题描述：用随机森林训练一个回归模型
解决方案：使用scikit-learn中的RandomForestRegressor训练随机森林回归模型
"""

#加载库
from sklearn.ensemble import  RandomForestRegressor
from sklearn import  datasets
#加载仅有两个特征的数据
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target
#创建随机森林回归对象
randomforest = RandomForestRegressor(random_state=0,n_jobs=-1)
#训练模型
model = randomforest.fit(features,target)
#新的样本
observation =  [[5,4]]
#预测
print(model.predict(observation))
