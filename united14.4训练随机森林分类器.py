#训练一个随机森林分类器模型
#解决方案： 使用scikit-learn 中的RandomForestClassifier 训练随机森林分类器模型
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
randomforest = RandomForestClassifier(random_state=0,n_jobs=-1)
#训练模型
model = randomforest.fit(features,target)

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