#问题： 减少过拟合
#解决方案： 当测试集损失不在减少时，就结束训练， 这个策略被称为提前结束（early stopping）
#疑问：其他的指标有没有达到？

#加载库
import numpy as np #加载Numpy库能对机器学习中常用的数据结构-向量，矩阵，张量进行高效操作
from keras.datasets import  imdb#从影评数据中加载数据和目标向量
from keras.preprocessing.text import Tokenizer#将影评数据转化为one-hot编码的特征向量
from keras import  models #启动神经网络
from keras import  layers#全连接层的参数设置
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))
#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu"))
#添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1,activation="sigmoid"))

#编译神经网络
network.compile(loss="binary_crossentropy",#交叉熵
                optimizer="rmsprop",#均方误差
                metrics=["accuracy"])#将准确率性能指标
#设置一个回调函数来提前结束训练，并保存训练结束时的最佳模型
callbacks = [EarlyStopping(monitor="var_loss",patience=2),
             ModelCheckpoint(filepath="best_model.h5",
                             monitor="var_loss",
                             save_best_only=True)]#ModelCheckpoint就会仅保存最佳模型

#训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=20,
                      callbacks=callbacks,#提前结束
                      verbose=1,#每个epoch 之后打印描述
                      batch_size=100,#每个批次的观察值数量
                      validation_data=(features_test,target_test))


