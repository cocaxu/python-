"""
给定离散数据或者计数数据（count data）训练一个朴素贝叶斯分类器
使用多项式朴素贝叶斯公式
"""

#加载库
import  numpy as np
from sklearn.naive_bayes import  MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#创建文本
text_data = np.array(['I love Brazil!','Brazil is best','Germany beats both'])
#创建词袋
count= CountVectorizer()
bag_of_words = count.fit_transform(text_data)

#创建特征
features= bag_of_words.toarray()
#创建目标向量
target =  np.array([0,0,1])
#给定每个分类的先验概率，创建一个多项式朴素贝叶斯独享
classifer = MultinomialNB(class_prior=[0.25,0.5])
#训练模型
model = classifer.fit(features,target)

#创建一个观察值
new_observation = [[0,0,0,1,0,1,0]]
#预测新的观察值的分类
print(model.predict(new_observation))