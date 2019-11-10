#问题： 假设有一个神经网络需要花很长时间来训练，现在需要保存训练过程防止训练被中断
#解决方案： 可以在每个epoch之后使用回调函数ModelCheckpoint
#疑问：保存的过程会不会花费更多的时间？保存了哪些东西？这些东西可以反映什么？

#加载库
import numpy as np #加载Numpy库能对机器学习中常用的数据结构-向量，矩阵，张量进行高效操作
from keras.datasets import  imdb#从影评数据中加载数据和目标向量
from keras.preprocessing.text import Tokenizer#将影评数据转化为one-hot编码的特征向量
from keras import  models #启动神经网络
from keras import  layers#全连接层的参数设置
#from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import  ModelCheckpoint
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
                optimizer="rmsprop",#均方根传播
                metrics=["accuracy"])#将准确率作为性能指标
#设置一个回调函数来提前结束训练，并保存训练结束时的最佳模型
checkpoint = [ModelCheckpoint(filepath="models.h5")]#ModelCheckpoint就会仅保存最佳模型
#【拓展】每个epoch之后，ModelCheckpoint把模型保存到filepath参数指定的路径中。分三种情况(仅有文件名时)
# 1、如果只给定一个文件名（eg.filepath="models.h5"） ,那么这个文件在每个epoch之后就会被最新的模型重写
# 2、如果只想根据某个损失函数的表现来保存最佳模型，可以设置save_best_only=True,monitor='var_loss',那么只有现有模型的测试集的损失函数更小，才可以将模型文件重写。
#3、保存每个eopch之后的模型为一个单独的文件，并可以将epoch编号(从0开始)和测试集损失值写在文件名中。即
#filepath=model_{epoch:02d}_{val_loss:.2f}.hsf5  （eg. 第11个epoch之后保存，测试集损失值是0.33， 则包含这个模型的文件名字是 model_10_0.33.hdf5） #


#训练神经网络
history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      callbacks = checkpoint,#提前结束
                      verbose=0,#每个epoch 之后打印描述
                      batch_size=100,#每个批次的观察值数量
                      validation_data=(features_test,target_test))


