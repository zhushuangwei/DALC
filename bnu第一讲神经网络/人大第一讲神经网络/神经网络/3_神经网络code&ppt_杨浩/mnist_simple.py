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

##1、导入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()   #使用Keras自带的mnist工具读取数据（第一次需要联网）
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) # 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维,使用的是reshape方法  
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) #一种简单的转化成one-hot矩阵的方式,numpy.arange(10) == y_train[:, None]生成的是array的bool型的矩阵，使用astype将其转化为数字矩阵
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

##2、自创建神经网络
#2.1、建立输入层和隐层1，其激活函数为sigmoid，采用20%的dropout
model = Sequential()  
model.add(Dense(output_dim=500,input_dim=784))  
model.add(Activation('sigmoid')) 
model.add(Dropout(0.5)) 

#2.2、建立隐层2，其激活函数为sigmoid,采用20%的dropout
model.add(Dense(output_dim=500))   
model.add(Activation('sigmoid'))  
model.add(Dropout(0.2))

#2.3、建立输出层，激活函数为softmax
model.add(Dense(output_dim=10))  
model.add(Activation('softmax'))

#2.4、使用compile来对学习过程进行配置
sgd = SGD(lr=0.1, decay=1e-6)      # 设定学习率（lr）、每次更新后的学习率衰减值(decay)等参数  
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy']) 

#2.5、训练模型
modelfit = model.fit(X_train, Y_train, batch_size=200, nb_epoch=20, shuffle=True, verbose=2, validation_split=0.3)  
#model.summary()      #打印出模型概况
modelevaluate = model.evaluate(X_test, Y_test, verbose=0)  

print '输出每一次epoch的训练/验证--损失/准确度'
print modelfit.history['categorical_accuracy'] 
print modelfit.history['val_categorical_accuracy']
print  'Test loss',modelevaluate[0],'Test accuracy',modelevaluate[1]

predict_class = model.predict_classes(X_test,verbose=0)  #输出在测试集上的预测值

print '输出模型在测试集上的前10个预测值',predict_class[:10]
print '输出测试集上Y的真实值',y_test[:10]


#plot(model,to_file='jiandan.jpg',show_shapes=True)          #画出所建立的模型图

#计算混淆矩阵
y_true = y_test
y_pred = predict_class
cf_test = confusion_matrix(y_true,y_pred)
print '模型在测试集上的混淆矩阵'
print cf_test

#将预测结果写入一个csv文件里
import csv
csvfile = file('predict_jiandan.csv','wb')
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

