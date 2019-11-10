#训练一个多元分类器神经网络
#使用keras 构建一个前馈神经网络，输出层采用的是 softmax激活函数
#加载库
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from  keras import  models
from  keras import  layers

#设定随机种子
np.random.seed(0)
#设定想要的特征的数量
number_of_features =5000
#加载特征和目标数据
data = reuters.load_data(num_words=number_of_features)
(data_train, target_vector_train),(data_test,target_vector_test)=data

#把特征数据转换成one-hot编码的特征矩阵
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train,mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test,mode="binary")

#把one-hot编码的特征向量转化成特征向量
target_train= to_categorical(target_vector_train)
target_test = to_categorical(target_vector_test)
#启动神经网络
network = models.Sequential()

#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=100,activation="relu",input_shape=(number_of_features,)))

#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=100,activation="relu"))
#添加使用Rsoftmax激活函数的全连接层
network.add(layers.Dense(units=46,activation="softmax"))

#编译神经网络
network.compile(loss="categorical_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])
#训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      verbose=0,
                      batch_size=100,
                      validation_data=(features_test,target_test))
