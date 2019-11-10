"""
基于树的学习算法是十分流行且应用广泛的一类非参数化的有监督学习算法。
优点：可解释性强，直观。
扩展为随机森林和堆叠（stacking）模型。既可用于回归有可用于分类
训练，处理，调整，可视化， 评估
"""

"""
问题描述： 使用决策树训练分类器
解决方案： 使用scikit-learn中的DecisionTreeClassifier
"""

#加载库
from sklearn.tree import  DecisionTreeClassifier
from sklearn import  datasets
#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
#创建决策树分类器对象
decisiontree = DecisionTreeClassifier(random_state=0)
#训练模型
model = decisiontree.fit(features,target)

"""
#使用训练好的模型来预测一个样本的分类
"""
#创建新样本
observation = [[5, 4, 3, 2]]
#预测样本的分类
print(model.predict(observation))
#样本分类的概率
print(model.predict_proba(observation))