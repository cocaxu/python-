#问题描述：有一个训练好的Keras模型，保存下来并在别的地方加载它
#解决方案：将模型保存在HDF5文件. HDF5文件包含了需要的一切，不仅包括加载模型做预测所需要的结构和训练后的参数，
#而且还包括重新训练所需要的各种设置（即，损失，优化器的设置和当前状态）

#加载库
import numpy as np
from  keras.datasets import  imdb
from  keras.preprocessing.text import  Tokenizer
from  keras import  models
from keras import  layers
from keras.models import  load_model

#设定随机种子
np.random.seed(0)

#设定想要的特征数量
number_of_features =  1000

#从影评数据中加载数据和目标向量
(data_train,target_train),(data_test,target_test)=imdb.load_data(num_words=number_of_features)
#将影评数据转换为one-hot 编码过的特征矩阵
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train,mode='binary')
features_test = tokenizer.sequences_to_matrix(data_test,mode='binary')

#创建神经网络对象
network = models.Sequential()
#添加使用RelU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu",input_shape=(number_of_features,)))

#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu"))
#添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1,activation="sigmoid"))

#编译神经网络
network.compile(loss="binary_crossentry",  #交叉编译
                optimizer="rmsprop",
                metrics=["accuracy"])
#训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      verbose=1,
                      batch_size=100,
                      validation_data=(features_test,target_test))

#保存神经网络
network.save("model.h5")

#使用保存好的神经网络
network1= load_model("model.h5")
