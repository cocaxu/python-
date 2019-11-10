"""
#问题描述：希望通过减少特征的数量来简化线性回归模型
#解决方案： 使用套索回归
套索回归的惩罚项可以将特征的系数减小为零，从而有效减少模型中特征的数量，近一步减少模型方差，
同时提高模型的可解释性。
"""

#加载库
from sklearn.linear_model import Lasso
from  sklearn.datasets import  load_boston
from  sklearn.preprocessing import  StandardScaler

#加载数据
boston = load_boston()
features = boston.data
target = boston.target
#特征标准化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#创建一个包含指定alpha值的岭回归
regression = Lasso(alpha=0.5)
#拟合线性回归模型
model = regression.fit(features_standardized,target)


"""
在scikit-learn库中使用RidgeCV方法，来设置alpha参数
"""
#加载库
from sklearn.linear_model import LassoCV
#创建包含三个alpha值的RidgeCV对象
regr_cv = LassoCV(alphas=[0.1,1.0,10.0])
#拟合线性回归
model_cv = regr_cv.fit(features_standardized,target)

#查看模型的系数
print(model_cv.coef_)
#查看最优模型的alpha值
print(model_cv.alpha_)

