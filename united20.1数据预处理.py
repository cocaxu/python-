#为神经网络预处理数据
#问题描述：对数据进行预处理，使其可以用于神经网络
#解决方案;使用scikit-learn 的StandardScaler标准化每个特征

#加载库
from sklearn import preprocessing
import  numpy as np

#创建特征
features = np.array([[-100.1,3240.41],
                     [-200.2,-234.1],
                     [5000.5,150.1],
                     [6000.6,-125.1],
                     [9000.9,-673.1]])

#创建scaler
scaler = preprocessing.StandardScaler()

#转换特征
features_standardized = scaler.fit_transform(features)

#展示特征
print(features_standardized)

#第一个特征的均值和标准差
print("Mean:", round(features_standardized[:,0].mean()))
print("Standard deviation:", round(features_standardized[:,0].std()))


#第二个特征的均值和标准差
print("Mean:", round(features_standardized[:,1].mean()))
print("Standard deviation:", round(features_standardized[:,1].std()))