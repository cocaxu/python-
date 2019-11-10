"""
问题描述：具有二元特征的数据，需要一个朴素贝叶斯分类器
解决方案：使用伯努利朴素贝叶斯分类器
"""
#加载库
import  numpy as np
from  sklearn.naive_bayes import  BernoulliNB

#创建三个二元特征
features=np.random.randint(2,size=(100,3))
#创建一个二元目标向量
target=np.random.randint(2,size=(100,1)).ravel()

#给每个分类的先验概率，创建一个多项式伯努利朴素贝叶斯对象
classifer = BernoulliNB(class_prior=[0.25,0.5])
#训练模型
model = classifer.fit(features,target)