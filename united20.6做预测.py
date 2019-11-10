#做预测
#问题描述：使用神经网络做预测
#解决方案：使用Keras 构建一个全馈神经网络，接着使用predict方法来做预测

#加载库
import numpy as np
from keras.datasets import  imdb
from keras.preprocessing.text import Tokenizer
from keras import  models
from keras import  layers
#设置随机种子
np.random.seed(0)
#设置想要的特征数量
number_of_features = 10000
#加载IMDB 电影的数据和目标向量
(data_train,target_train),(data_test,target_test)=imdb.load_data(num_words=number_of_features)
#把IMDB数据转化为one-hot编码的特征矩阵
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train,mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test,mode="binary")

#启动神经网络
network =models.Sequential()

#添加使用ReLU激活函数的全连接层

network.add(layers.Dense(units=16,activation="relu",input_shape=(number_of_features,)))

#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu"))
#添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1,activation="sigmoid"))

#编译神经网络
network.compile(loss="binary_crossentropy",#均方误差
                optimizer="rmsprop",
                metrics=["accuracy"])
#训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      verbose=0,
                      batch_size=100,
                      validation_data=(features_test,target_test))
#预测测试集的分类
predicted_target = network.predict(features_test)
print(predicted_target[0])

