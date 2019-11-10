"""
问题描述：为KNN分类器找到最佳的k值
解决方案：使用GridSearchCV这样的模型选择技术
"""#

#加载库

from sklearn import  datasets
from  sklearn.preprocessing import  StandardScaler
from  sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import  GridSearchCV


#加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
#创建standardizer
standardizer = StandardScaler()
#标准化特征
features_standardized= standardizer.fit_transform(X)
#创建一个KNN分类器
knn = KNeighborsClassifier(n_neighbors=5,n_jobs=1)

#创建一个流水线
pipe = Pipeline([("standardizer",standardizer),("knn",knn)])
#创建一个可选值的范围
search_space =  [{"knn__n_neighbors":[1,2,3,4,5,6,7,8,9,10]}]
#创建grid搜索
classifer = GridSearchCV(pipe,search_space,cv=5,verbose=0).fit(features_standardized,y)
#最佳邻域的大小（k）
print(classifer.best_estimator_.get_params()["knn__n_neighbors"])