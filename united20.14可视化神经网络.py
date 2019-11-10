#问题描述： 快速可视化神经网络的结构
#解决方案：使用Keras的mode_to_dot或者plot_model


#加载库
from keras import  models
from keras import layers

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot
from keras.utils import  plot_model
import pydot
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#启动神经网络
network = models.Sequential()
#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu",input_shape=(10,)))
#添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16,activation="relu"))
#添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1,activation="sigmoid"))

#可视化网络结构
#SVG(model_to_dot(network,show_shapes=False).create(prog="dot",format="svg"))
#keras提供了工具函数用于快速可视化神经网络
#model_to_dot可以显示一个神经网络
# show_shapes 参数指定是否展示输入输出的形状，可以帮助我们调试网络

#将可视化后的网络结构图保存为文件
#plot_model(network,show_shapes=True,to_file="network.png")

#展示一个更加简单的网络模型
SVG(model_to_dot(network,show_shapes=False).create(prog="dot",format="svg"))
plot_model(network,show_shapes=True,to_file="network.png")