#处理不均衡分类
#问题：用不均衡的分类数据训练一个SVC
#解决方案：使用class_weight 来增加对数据量少的类别分错类后的惩罚

#加载库
from  sklearn.svm import  SVC
from sklearn import datasets
from  sklearn.preprocessing import  StandardScaler
import  numpy as np

#加载只有两个分类的数据
iris= datasets.load_iris()
features = iris.data[:100,:]
target = iris.tatget[:100]
#删除前40个观察值，让各个分类的数据分布不均衡
features = features[40:,:]
target = target[40:]

#创建目标向量，数值0代表分类0，其他分类用数值1表示
target=np.where((target==0),0,1)

#标准化特征
scaler =  StandardScaler()
features_standardized = scaler.fit_transform(features)
#创建SVC
svc=SVC(kernel='linear',class_weight="balanced",C=1.0,random_state=0)
#训练分类器
model=svc.fit(features_standardized,target)

