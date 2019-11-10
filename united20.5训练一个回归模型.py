#问题描述：为回归问题训练一个神经网络
#解决方案：使用keras 构建只要一个输出神经元，没有激活函数的前馈神经网络

#加载库
import  numpy as np
from keras import layers
from keras.preprocessing.text  import Tokenizer
from keras import  models
from  sklearn.datasets import  make_regression
from sklearn.model_selection import  train_test_split
from  sklearn import  preprocessing

#设置随机种子
np.random.seed(0)

#生成特征矩阵和目标向量
features, target= make_regression(n_samples=10000,
                                  n_features=3,
                                  n_informative=3,
                                  n_targets=1,
                                  noise=0.0,
                                  random_state=0)

#把数据分成训练集和测试集
features_train, features_test,target_train,target_test =train_test_split(features,target,test_size=0.33,random_state=0)
#启动神经网络
network = models.Sequential()
#添加使用ReLU激活函数的全连接层

network.add(layers.Dense(units=32,activation="relu",input_shape=(features_train.shape[1],)))

#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=32,activation="relu"))
#添加使用Rsoftmax激活函数的全连接层
network.add(layers.Dense(units=1))

#编译神经网络
network.compile(loss="mes",#均方误差
                optimizer="rmsprop",
                metrics=["mes"])
#训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=10,
                      verbose=0,
                      batch_size=100,
                      validation_data=(features_test,target_test))