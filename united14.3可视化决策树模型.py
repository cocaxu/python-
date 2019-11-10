"""
问题描述：可视化一个决策树模型
解决方案: 将决策树模型导出为DOT格式并可视化
"""
#加载库
import  pydotplus
from sklearn.tree import  DecisionTreeClassifier
from sklearn import  datasets
from IPython.display import Image
from sklearn import tree
import graphviz


#加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

#创建决策树分类器对象
decisiontree = DecisionTreeClassifier(random_state=0)
#训练模型
model = decisiontree.fit(features,target)

#创建DOT数据
dot_data = tree.export_graphviz(decisiontree,out_file=None,feature_names=iris.feature_names,class_names=iris.target_names)
#绘制图像
graph=pydotplus.graph_from_dot_data(dot_data)

#显示图像
Image(graph.create_png())