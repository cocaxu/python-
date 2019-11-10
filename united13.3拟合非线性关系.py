#问题描述 ：对一个非线性关系建模
#解决方案：在线性回归模型中纳入多项式特征，以创建多项式回归模型

#加载库
from sklearn.linear_model import  LinearRegression
from  sklearn.datasets import  load_boston
from sklearn.preprocessing import  PolynomialFeatures

#加载只有两个特征的数据集
boston = load_boston()
features= boston.data[:,:1]#仅包含有一个特征
target =  boston.target
"""
相当于是数据预处理。得到的模型的输入应为预处理过后的特征数据
"""
#创多项式特征 x^2 和x^3
polynomial = PolynomialFeatures(degree=3,include_bias=False)
features_polynomial = polynomial.fit_transform(features)
"""""
interaction = PolynomialFeatures(degree=3,include_bias=False, interaction_only=True)#设置交互参数
features_interaction = interaction.fit_transform(features)#将两个特征进行交互,前两列是原始特征，第三列是两个特征相乘
"""
print(features_polynomial)
#创建线性回归对象
regression = LinearRegression()

#拟合线性回归模型
model = regression.fit(features_polynomial, target)
print(model.predict(features_polynomial)[0])
print(target[0])
