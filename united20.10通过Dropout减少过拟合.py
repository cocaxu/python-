#问题： 减少过拟合
#解决方案： 使用Dropout（丢弃）方法向网络结构引入噪声
#疑问：假如多余的噪声来平衡过拟合,饮鸩止渴？？？

#加载库
import numpy as np #加载Numpy库能对机器学习中常用的数据结构-向量，矩阵，张量进行高效操作
from keras.datasets import  imdb#从影评数据中加载数据和目标向量
from keras.preprocessing.text import Tokenizer#将影评数据转化为one-hot编码的特征向量
from keras import  models #启动神经网络
from keras import  layers#全连接层的参数设置


#设置随机种子
np.random.seed(0)
#设置想要的特征数量
number_of_features = 1000
#加载IMDB 电影的数据和目标向量
(data_train,target_train),(data_test,target_test)=imdb.load_data(num_words=number_of_features)
#把IMDB数据转化为one-hot编码的特征矩阵
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train,mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test,mode="binary")

#启动神经网络
network =models.Sequential()
#为输入层添加一个Dropout层
network.add(layers.Dropout(0.2,input_shape=(number_of_features,)))
#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu"))
#为前面的隐含层添加一个Dropout层
network.add(layers.Dropout(0.5))
#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu"))
#为输出层添加一个Dropout层
network.add(layers.Dropout(0.5))
#添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1,activation="sigmoid"))

#编译神经网络
network.compile(loss="binary_crossentropy",#交叉熵
                optimizer="rmsprop",#均方误差
                metrics=["accuracy"])#将准确率性能指标
#训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      verbose=0,#每个epoch 之后打印描述
                      batch_size=100,#每个批次的观察值数量
                      validation_data=(features_test,target_test))


