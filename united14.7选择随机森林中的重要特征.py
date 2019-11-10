""""
问题描述：在随机森林中进行特征选择
解决方案: 确定重要特征，并使用他们重新训练模型
"""#

#加载库
from  sklearn.ensemble import  RandomForestClassifier
from  sklearn import  datasets
from sklearn.feature_selection import SelectFromModel
import  numpy as np

#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
#创建随机森林分类器对象
randomforest =  RandomForestClassifier(random_state=0,n_jobs=-1)
#创建对象，选择重要性大于或等于阈值的特征
selector = SelectFromModel(randomforest,threshold=0.3)
#使用选择器创建新的特征矩阵
features_important = selector.fit_transform(features,target)
#使用重要特征训练随机森林模型
model = randomforest.fit(features_important,target)
#计算特征的重要性
importances = model.feature_importances_
#查看模型中每个特征的重要程度
print(importances)
#将特征的重要性按降序排列
indices= np.argsort(importances)[::-1]
#按照特征的重要性对特征名称重新排序
names = [iris.feature_names[i] for i in indices]

