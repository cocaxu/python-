"""
问题描述：对于分类位置的观察值，基于邻居的分类来预测他的分类
解决方案: 如果数据集不是特别大，就直接用KNeighborsClassifier
"""
#加载库
from sklearn import  datasets
from  sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
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
knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1).fit(X_std,y)

#创建两个观察值
new_observations = [[0.75,0.75,0.75,0.75],[1,1,1,1]]
#预测两个观察值的分类
print(knn.predict(new_observations))
#查看每个观察值分别属于3个分类中的某一个的概率
print(knn.predict_proba(new_observations))