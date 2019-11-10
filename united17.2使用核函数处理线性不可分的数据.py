# 问题：训练一个SVC,数据是线性不可分的
# 解决方案：使用核函数

# 加载库
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

# 设置随机种子
np.random.seed(0)

# 生成两个特征
features = np.random.randn(200, 2)
# 使用异或门 创建线性不可分的数据
target_xor = np.logical_xor(features[:, 0] > 0, features[:, 1] > 0)
target = np.where(target_xor, 0, 1)

# 创建一个有径向基核函数的支持向量机
#svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)

# 训练分类器
#@model = svc.fit(features, target)

# 画出观察值和超平面决策边界
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier):
    cmap = ListedColormap("red", "blue")
    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker="+", label=cl)


# 创建一个使用线性核函数的SVC
svc_linear = SVC(kernel='linear', random_state=0,C=1)
#训练模型
svc_linear.fit(features, target)
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto',
    kernel='linear', max_iter=-1, probability=False, random_state=0, shrinking=True, tol=0.001, verbose=False)
# 画出观察值和超平面
plot_decision_regions(features, target, classifier=svc_linear)
plt.axis("off"), plt.show();

# 创建一个使用径向基核函数的SVC
svc_rbf = SVC(kernel="rbf", random_state=0, gamma=1, C=1)
# 训练这个分类器
model = svc_rbf.fit(features, target)
#画出观察值和超平面
plot_decision_regions(features,target,classifier=svc_rbf)
plt.axis("off"),plt.show();






