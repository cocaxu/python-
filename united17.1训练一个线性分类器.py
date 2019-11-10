#问题：训练一个模型对观察值进行分类
#解决方案： 用SVC来寻找最大化分类之间间距的超平面


#加载库
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import  StandardScaler
import  numpy as np
#加载数据 数据里只有两种分类和特征
iris = datasets.load_iris()
#digits = datasets.load_digits()
features = iris .data[:100,:2]
target = iris .target[:100]
print(target)
#标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
#创建支持向量分类器
svc = LinearSVC(C=1.0)
#训练模型
model = svc.fit(features_standardized, target)


#可视化验证上述所建立的分类器

#加载库
from matplotlib import pyplot as plt
#画出样本点，并根据其分类上色
color = ["black" if c==0 else "lightgrey" for c in target]

plt.scatter(features_standardized[:,0], features_standardized[:,1],c=color)

#创建超平面
w = svc.coef_[0]
a = -w[0]/w[1]
xx= np.linspace(-2.5,2.5)
yy=a*xx-(svc.intercept_[0])/w[1]
#画出超平面
plt.plot(xx,yy)
plt.axis("off"),plt.show();

#若在左上角的空间中创建一个新的样本点，他会被预测为属于分类0
#创建一个新的样本点
new_observation =[[-2,3]]
#预测新样本点的分类
svc.predict(new_observation)
np.array([0])