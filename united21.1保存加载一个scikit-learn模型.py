#问题描述：有一个scikit-learn模型，将其保存并可以在其他地方加载它
#解决方案：可以将该模型保存为pickle文件
#加载库
from sklearn.ensemble import  RandomForestClassifier
from sklearn import datasets
from  sklearn.externals import joblib

#加载数据 数据里只有两种分类和特征
iris = datasets.load_iris()
features = iris .data
target = iris .target
#创建决策树分类器
classifer = RandomForestClassifier()

#训练模型
model = classifer.fit(features,target)

#把模型保存为pickle文件
joblib.dump(model,"model0.21.3.pkl")
#为所保存的模型加上scikit-learn的版本，因为scikit-learn模型在scikit-learn各个版本上不兼容
#加载库
import sklearn
#获取scikit-learn版本
scikit_version= sklearn.__version__
#把模型保存为pickle文件
joblib.dump(model,"model_{version}.pkl".format(version=scikit_version))