""""
问题描述：在不使用交叉验证的情况下评估随机森林模型
解决方案：计算模型的袋外误差分数
"""

#加载库
from sklearn.ensemble import  RandomForestClassifier
from  sklearn import  datasets
#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
"""
训练模型
"""
#创建随机森林分类器对象
randomforest = RandomForestClassifier(random_state=0,n_estimators=1000,oob_score=True,n_jobs=-1)
#训练模型
model = randomforest.fit(features,target)
#查看袋外误差
print(randomforest.oob_score_)
"""
预测新的样本
"""
#创建新的样本
observation = [[5,4,3,2]]
#预测样本的分类
print(model.predict(observation))

"""
训练改变度量分类质量的模型
"""
#使用熵创建随机森林分类器对象
randomforest_entropy =  RandomForestClassifier(criterion="entropy",random_state=0)
#训练模型
model_entropy = randomforest_entropy.fit(features,target)
"""
预测新模型的样本
"""
print(model_entropy.predict(observation))