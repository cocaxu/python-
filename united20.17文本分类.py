#问题描述： 对文本数据分类
#解决方案： 使用长短期记忆递归神经网络

#加载库
import numpy as np
from keras.datasets import  imdb
from keras.preprocessing import  sequence
from  keras import models
from  keras import layers

#设置随机种子

#设置想要的特征数量
number_of_features =1000
#从影评数据中加载数据和目标向量
(data_train,target_train),(data_test,target_test)=imdb.load_data(num_words=number_of_features)
#采用添加填充值或者截断的方式，使每个样本都有400个特征
features_train= sequence.pad_sequences(data_train,maxlen=400)
features_test = sequence.pad_sequences(data_test,maxlen=400)

#启动神经网络
network =models.Sequential()
#添加嵌入层
network.add(layers.Embedding(input_dim=number_of_features,output_dim=128))
#添加一个有128个神经元的长短期记忆网络层
network.add(layers.LSTM(units=128))
#添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1,activation="sigmoid"))

#编译神经网络
network.compile(loss="binary_crossentropy",#交叉熵
                optimizer="Adam",#adam优化器
                metrics=["accuracy"])#将准确率性能指标
#训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      verbose=0,
                      batch_size=1000,
                      validation_data=(features_test,target_test))