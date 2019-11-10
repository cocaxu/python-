# 问题：想知道观察值被预测为某个分类的概率
#解决方案：若使用scikit-learn 的SVC， 可以设置probability=True, 然后训练模型，接着可以使用predict_proba来
#来查看校准后的概率

#加载库
from sklearn.svm import  SVC
from sklearn import  datasets
from sklearn.preprocessing import  StandardScaler
import  numpy as np
#加载数据
iris = datasets.load_iris()
features = iris.data
target =  iris.target

#标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#创建SVC对象
svc=SVC(kernel='linear',probability=True,random_state=0)
#训练分类器
model = svc.fit(features_standardized,target)
#创建一个观察值
new_observation= [[.4,.4,.4,.4]]

#查看观察值 被预测为被不同分类的概率
print(model.predict_proba(new_observation))

