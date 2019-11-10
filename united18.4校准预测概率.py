"""
问题描述：校准由朴素贝叶斯分类器得出的预测概率，使他们可以被解释
解决方案：使用CalibratedClassifierCV
"""
#加载数据库
from sklearn import  datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

#加载数据
iris =  datasets.load_iris()
features= iris.data
target = iris.target

#创建高斯朴素贝叶斯对象
classifer = GaussianNB()
#classifer= GaussianNB(priors=[0.25,0.25,0.5])
#创建使用sigmoid校准调校过的交叉验证模型
classifer_sigmoid = CalibratedClassifierCV(classifer,cv=2,method='sigmoid')
#校准概率
classifer_sigmoid.fit(features,target)

#创建一个观察值
#new_observation =  [[4,4,4,0.4]]
new_observation =  [[2.6,2.6,2.6,.4]]
#预测分类
print(classifer_sigmoid.predict(new_observation))
#预测分类概率
print(classifer_sigmoid.predict_proba(new_observation))

