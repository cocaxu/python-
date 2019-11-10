#识别支持向量
#问题描述：找出哪些观察值是决策超平面的支持向量
#解决方案训练模型然后使用support_vectors_方法

#加载库
from sklearn.svm import SVC
from sklearn import  datasets
from  sklearn.preprocessing import StandardScaler
import  numpy as np

#加载数据，数据中只有两个分类
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

#标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
#创建SVC对象
svc = SVC(kernel='linear',random_state=0)
#训练分类器
model = svc.fit(features_standardized,target)

#查看支持向量
#使用support_vectors_方法来输出模型中观察值特征的4个支持向量
print(model.support_vectors_)
#使用support_来查看支持向量在观察值中的索引值
print(model.support_)
#使用n_support_来查看每个分类有几个支持向量
print(model.n_support_)