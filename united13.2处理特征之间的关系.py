#问题描述： 一个特征对目标变量的影响取决于另一个特征，现在要处理两个特征之间的影响
#解决方案：使用scikit-learn中的PolynomialFeatures创建多项式特征，对这种依赖关系建模

#加载库
from sklearn.linear_model import  LinearRegression
from  sklearn.datasets import  load_boston
from  sklearn.preprocessing import  PolynomialFeatures

#加载只有两个特征的数据集
boston = load_boston()
features= boston.data[:,:2]
target =  boston.target
"""
print(features)
print(target)
"""
#创建交互特征
interaction = PolynomialFeatures(degree=3,include_bias=False, interaction_only=True)#设置交互参数
features_interaction = interaction.fit_transform(features)#将两个特征进行交互,前两列是原始特征，第三列是两个特征相乘
print(features_interaction)
#创建线性回归对象
regression= LinearRegression()
#拟合线性回归
model =  regression.fit(features_interaction, target)
print(features[0])
