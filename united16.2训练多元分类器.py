"""
问题描述：数据集中的分类不止两个分类，需要训练处一个合适的分类器模型
解决方案：在scikit-learn中，通过LogisticRegression使用一对多或者多项式方法来训练逻辑回归模型
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
logistic_regression_ovr = LogisticRegression(random_state=0,multi_class="ovr")
#训练模型
model_ovr = logistic_regression_ovr.fit(features_Standardized,target)

#创建一个新的观察值
new_observation =[[.5,.5,.5,.5],[1,1,1,1],[.1,.1,.1,.1]]
#预测分类
print(model_ovr.predict(new_observation))
#查看预测概率
print(model_ovr.predict_proba(new_observation))


#创建一个逻辑回归对象
logistic_regression_MLR = LogisticRegression(random_state=0,multi_class="multinomial")
#训练模型
model_MLR = logistic_regression_MLR.fit(features_Standardized,target)

#创建一个新的观察值
new_observation =[[.5,.5,.5,.5],[1,1,1,1],[.1,.1,.1,.1]]
#预测分类
print(model_MLR.predict(new_observation))
#查看预测概率
print(model_MLR.predict_proba(new_observation))
