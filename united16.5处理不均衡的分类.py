"""
问题描述：训练一个简单的分类器模型‘
解决方案：在scikit-learn中使用LogisticRegression来训练一个逻辑回归模型

"""
#加载库
from  sklearn.linear_model import  LogisticRegression
from  sklearn import  datasets
import  numpy as np
from  sklearn.preprocessing import  StandardScaler
#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
#删除前40个样本以获得高度不均衡的数据
features= features[40:,:]
target = target[40:]
#创建目标向量表明分类为0还是1
target = np.where((target==0),0,1)
#标准化特征
scaler =StandardScaler()
features_standardized = scaler.fit_transform(features)

#创建随机森林分类器对象
logistic_regression = LogisticRegression(random_state=0,class_weight="balanced")
#训练模型
model = logistic_regression.fit(features,target)