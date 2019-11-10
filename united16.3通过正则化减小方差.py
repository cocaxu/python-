"""
问题描述：减小逻辑回归模型的方差
解决方案：调校正则化强度超参数C
"""
#加载库
from sklearn.linear_model import  LogisticRegressionCV
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
logistic_regression = LogisticRegressionCV(penalty='l2',Cs=10,random_state=0,n_jobs=-1)
#训练模型
model_ovr = logistic_regression.fit(features_Standardized,target)

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
