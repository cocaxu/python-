"""
问题描述：在高度不均衡的数据上训练随机森林模型
解决方案：用参数class_weight= "balanced" 训练决策树或随机森林模型
scikit-learn 中，很多学习算法都带有用于纠正不均衡分类的内置方法
"""
#加载库
from  sklearn.ensemble import  RandomForestClassifier
from  sklearn import  datasets
import  numpy as np

#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
#删除前40个样本以获得高度不均衡的数据
features= features[40:,:]
target = target[40:]
#创建目标向量表明分类为0还是1
target = np.where((target==0),0,1)
#创建随机森林分类器对象
randomforest = RandomForestClassifier(random_state=0,n_jobs=-1,class_weight="balanced")
#训练模型
model = randomforest.fit(features,target)