"""
问题描述：找到一个离观察值最近的k个观察值（邻居）
解决方案：使用scikit-learn 的NearestNeighbors
"""
#加载库
from sklearn import  datasets
from  sklearn.neighbors import NearestNeighbors
from  sklearn.preprocessing import  StandardScaler

#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
#创建standardizer
standardizer =  StandardScaler()
#特征标准化
features_standardized= standardizer.fit_transform(features)
#两个最近的观察值
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)
#创建一个观察值
new_observation=[1,1,1,1]
#获取离观察值最近的两个观察值的索引，以及到这两个点的距离
distances,indices= nearest_neighbors.kneighbors([new_observation])

#查看最近的两个观察值
print(features_standardized[indices])
print(distances)