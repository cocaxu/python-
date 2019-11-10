"""
问题描述：数据的特征是连续的，训练一个朴素贝叶斯分类器
解决方案：在scikit-learn中使用高斯朴素贝叶斯分类器
"""
#加载数据库
from sklearn import  datasets
from sklearn.naive_bayes import GaussianNB

#加载数据
iris =  datasets.load_iris()
features= iris.data
target = iris.target

#创建高斯朴素贝叶斯对象
#classifer = GaussianNB()
classifer= GaussianNB(priors=[0.25,0.25,0.5])
#训练模型
model = classifer.fit(features,target)

#创建一个观察值
new_observation =  [[4,4,4,0.4]]
#预测分类
print(model.predict(new_observation))
#预测分类概率
print(model.predict_proba(new_observation))

