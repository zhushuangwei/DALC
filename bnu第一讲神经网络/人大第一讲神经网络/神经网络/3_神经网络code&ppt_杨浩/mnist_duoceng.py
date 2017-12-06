#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.models import Sequential                         #导入连续模型
from keras.layers.core import Dense, Dropout, Activation    #导入常用层
from keras.optimizers import SGD                            #导入优化器
from keras.datasets import mnist                            #导入mnist数据集
#from keras.utils.visualize_util import plot                 #可以达到神经网络模型可视化的目的
from sklearn.metrics import confusion_matrix                #计算预测出来值的混淆矩阵
import numpy

numpy.random.seed(1234)     #让keras脚本每次产生确定的数据

#导入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])  
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) 
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

#建立10层的神经网络
model = Sequential()
model.add(Dense(output_dim=100,input_dim=784))
model.add(Activation('relu'))
model.add(Dense(output_dim=100,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=100,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=100,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=100,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=100,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=100,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=100,input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=10,input_dim=100))
model.add(Activation('softmax'))


#对学习过程进行配置
sgd = SGD(lr=0.01, decay=1e-6)      # 设定学习率（lr）、每次更新后的学习率衰减值(decay)等参数  
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy']) 

#训练模型
modelfit = model.fit(X_train, Y_train, batch_size=200, nb_epoch=20, shuffle=True, verbose=2, validation_split=0.3)  

modelevaluate = model.evaluate(X_test, Y_test, verbose=0)  

print '输出每一次epoch的训练/验证--损失/准确度'
print modelfit.history['categorical_accuracy'] 
print modelfit.history['val_categorical_accuracy']
print  'Test loss',modelevaluate[0],'Test accuracy',modelevaluate[1]

predict_class = model.predict_classes(X_test,verbose=0)  #输出在测试集上的预测值

print '输出模型在测试集上的前10个预测值',predict_class[:10]
print '输出测试集上Y的真实值',y_test[:10]


#plot(model,to_file='duoceng.jpg',show_shapes=True)          #画出所建立的模型图

#计算混淆矩阵
y_true = y_test
y_pred = predict_class
cf_test = confusion_matrix(y_true,y_pred)
print '模型在测试集上的混淆矩阵'
print cf_test

#将预测结果写入一个csv文件里
import csv
csvfile = file('predict_duoceng.csv','wb')
writer = csv.writer(csvfile)
writer.writerow(['predict'])#行名
predict = [str(i) for i in predict_class]
writer.writerows(predict)
csvfile.close()

#画随epoch变化训练集上的categorical_accuracy的变化情况
import matplotlib.pyplot as plt
a = numpy.arange(1,21)
b = modelfit.history['categorical_accuracy'] 
c = modelfit.history['val_categorical_accuracy']
plt.plot(a,b)
plt.plot(a,b,'ob')
plt.plot(a,c)
plt.plot(a,c,'og')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(('Train_accuracy','','Validate_accuray'),bbox_to_anchor=(1,0.7))
plt.show()

