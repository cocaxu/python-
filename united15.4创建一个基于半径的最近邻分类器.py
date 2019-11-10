"""
问题描述：对于分类未知的观察值，根据一定距离范围内所有观察值的分类来确定其分类
解决方案：使用RadiusNeighborsClassifier
"""
#加载库
from sklearn import  datasets
from  sklearn.neighbors import NearestNeighbors,KNeighborsClassifier,RadiusNeighborsClassifier
from  sklearn.preprocessing import  StandardScaler

#加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
#创建standardizer
standardizer =  StandardScaler()
#特征标准化
X_std= standardizer.fit_transform(X)

#训练一个有5个邻居的KNN分类器
rnn = RadiusNeighborsClassifier(radius=5,n_jobs=-1).fit(X_std,y)

#创建两个观察值
new_observations = [[0.75,0.75,0.75,0.75],[1,1,1,1]]
#预测两个观察值的分类
print(rnn.predict(new_observations))
