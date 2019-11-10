#设计一个神经网络
#解决方案： 使用Keras的Sequential 模型

#加载库
from keras import  models
from keras import layers

#启动神经网络
network = models.Sequential()

#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu",input_shape=(10,)))
#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu"))
#添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1,activation="sigmoid"))
#编译神经网络
network.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])
#刚开始编译出现错误  AttributeError: 'Sequential' object has no attribute 'complie'
#解决办法：重新把 compile 写了一遍 就好了