"""
问题描述：手动控制决策树的结构和规模
解决方案：使用scikit-learn 控制基于树的算法的结构参数
"""
#加载库
from sklearn.tree import  DecisionTreeClassifier
from  sklearn import  datasets
#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target
"""
训练模型
"""
#创建随机森林分类器对象
decisiontree = DecisionTreeClassifier(random_state=0,
                                      max_depth=None,#树的最大深度
                                      min_samples_split=2,#在该节点分裂之前，节点上最小的样本数
                                      min_samples_leaf=1,#叶子节点需要的最下样本数
                                      min_weight_fraction_leaf=0,#
                                      max_leaf_nodes=None,#最大叶子节点数
                                      min_impurity_decrease=0)
#训练模型
model = decisiontree.fit(features,target)