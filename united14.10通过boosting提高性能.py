"""
问题描述： 需要一个比决策树或随机森林性能更好的模型
解决方案： 使用AdaBoostClassifier 或AdaBoostRegressor 训练一个boosting模型
"""


#加载库
from sklearn.tree import  DecisionTreeClassifier
from  sklearn import  datasets
from sklearn.ensemble import  AdaBoostClassifier
#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
#创建adaboost树分类器对象
adaboost =  AdaBoostClassifier(random_state=0)
#训练模型
model = adaboost.fit(features,target)


"""
预测新的样本
"""
#创建新的样本
observation = [[5,4,3,2]]
#预测样本的分类
print(model.predict(observation))
