#问题描述：希望减少线性回归模型的方差
#解决方案： 使用包含惩罚项的学习算法，如岭回归（ridge regression）和套索回归（lasso regression）
#岭回归算法：最小二乘法的改进版。
#加载库
from sklearn.linear_model import Ridge
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
regression = Ridge(alpha=0.5)
#拟合线性回归模型
model = regression.fit(features_standardized,target)


"""
在scikit-learn库中使用RidgeCV方法，来设置alpha参数
"""
#加载库
from sklearn.linear_model import RidgeCV
#创建包含三个alpha值的RidgeCV对象
regr_cv = RidgeCV(alphas=[0.1,1.0,10.0])
#拟合线性回归
model_cv = regr_cv.fit(features_standardized,target)

#查看模型的系数
print(model_cv.coef_)
#查看最优模型的alpha值
print(model_cv.alpha_)

