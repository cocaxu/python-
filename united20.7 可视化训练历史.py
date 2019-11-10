#找到神经网络的损失（loss）或准确率分数的“甜蜜点（sweet spot）”
#使用Matpltlib 可视化测试集和训练集在每个epoch上的损失

#加载库
import  numpy as np
from keras.datasets import imdb
from  keras.preprocessing.text import Tokenizer
from keras import  models
from keras import layers
import  matplotlib.pyplot as plt

#设置随机种子
np.random.seed()

#设置我们想要的特征数量
number_of_features = 10000
#从影评数据中加载数据和目标向量
(data_train,target_train),(data_test, target_test)= imdb.load_data(number_of_features)

#把影评数据转换为one-hot 编码的特征矩阵
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train,mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test,mode="binary")

#启动神经网络
network = models.Sequential()
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
                      epochs=15,
                      verbose=0,
                      batch_size=1000,
                      validation_data=(features_test,target_test))
#获取训练集和测试集的损失函数历史
training_loss = history.history["loss"]
test_loss=history.history["val_loss"]

#为每个epoch创建编号
epoch_count = range(1, len(training_loss)+1)
#画出损失的历史数值
plt.plot(epoch_count,training_loss,"r--")
plt.plot(epoch_count,test_loss,"b-")
plt.legend(["Training Loss","Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
#可视化每个epoch的训练集和测试集数据的准确率
#获取训练集和测试集数据的准确率历史数值
