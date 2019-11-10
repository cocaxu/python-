from sklearn.externals import  joblib
import numpy as np
classifer = joblib.load("model.pkl")

#创建新的样本
new_observation = [[5.2, 3.2, 1.1, 0.1]]
#预测样本的分类
print(classifer.predict(new_observation))