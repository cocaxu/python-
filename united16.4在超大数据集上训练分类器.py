"""
问题描述：在一个超大数据集上训练一个简单的分类器模型
解决方案：在scikit-learn中通过LogisticRegression使用随机平均梯度（Stochastic Average Gradient, SAG）solver
来训练一个逻辑回归模型
"""
#加载库
from sklearn.linear_model import  LogisticRegression
from sklearn import  datasets
from  sklearn.preprocessing import  StandardScaler

#加载仅有两个分类的数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

#标准化特征
scaler =  StandardScaler()
features_Standardized = scaler.fit_transform(features)
#创建一个逻辑回归对象
logistic_regression = LogisticRegression(random_state=0,solver="sag")
#训练模型
model = logistic_regression.fit(features_Standardized,target)

#创建一个新的观察值
new_observation =[[.5,.5,.5,.5],[1,1,1,1],[.1,.1,.1,.1]]
#预测分类
print(model.predict(new_observation))
#查看预测概率
print(model.predict_proba(new_observation))

