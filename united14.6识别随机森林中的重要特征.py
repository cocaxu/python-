"""
问题描述： 想知道随机森林模型中最重要的特征
解决方案： 计算并可视化每个特征的重要性
"""#
#加载库
import numpy as np
import  matplotlib.pyplot as plt
from  sklearn.ensemble import  RandomForestClassifier
from  sklearn import  datasets
#加载数据
boston = datasets.load_iris()
features = boston.data
target = boston.target
#创建随机森林分类器对象
randomforest =  RandomForestClassifier(random_state=0,n_jobs=-1)
#训练模型
model = randomforest.fit(features,target)
#计算特征的重要性
importances = model.feature_importances_
#查看模型中每个特征的重要程度
print(importances)
#将特征的重要性按降序排列
indices= np.argsort(importances)[::-1]
#按照特征的重要性对特征名称重新排序
names = [boston.feature_names[i] for i in indices]

#创建图
plt.figure()
#创建图标题
plt.title("Feature Importance")
#添加数据条
plt.bar(range(features.shape[1]),importances[indices])
#将特征名称添加为X轴标签
plt.xticks(range(features.shape[1]),names,rotation=90)
#显示图
plt.show()