#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import SGD  
from keras.datasets import mnist  
import numpy

numpy.random.seed(1234) #让keras脚本每次产生确定的数据

#导入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])  
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) 
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)


#1、神经网络--损失函数--mse
#1.1、建立输入层与隐层1，其激活函数为sigmoid
model1 = Sequential()  
model1.add(Dense(output_dim=100,input_dim=28*28))  
model1.add(Activation('sigmoid')) 

#1.2、建立隐层2，其激活函数为sigmoid
model1.add(Dense(output_dim=100))   
model1.add(Activation('sigmoid'))  

#1.3、建立输出层，激活函数为softmax
model1.add(Dense(output_dim=10))  
model1.add(Activation('softmax'))

#1.4、使用compile来对学习过程进行配置
model1.compile(loss='mse', optimizer=SGD(lr=0.01),metrics=['accuracy']) 

#1.5、训练模型
modelfit1 = model1.fit(X_train, Y_train, batch_size=200, nb_epoch=20,verbose=2)
score1 = model1.evaluate(X_test,Y_test,verbose=0)

print('Total loss on Testing Set:',score1[0])
print('Accuracy of Testing Set:',score1[1])


#2、神经网络-改进-改变损失函数
#2.1、建立输入层和隐层1，其激活函数为sigmoid
model2 = Sequential()  
model2.add(Dense(output_dim=100,input_dim=28*28))  
model2.add(Activation('sigmoid')) 

#2.2、建立隐层2，其激活函数为sigmoid
model2.add(Dense(output_dim=100))   
model2.add(Activation('sigmoid'))  

#2.3、建立输出层，激活函数为softmax
model2.add(Dense(output_dim=10))  
model2.add(Activation('softmax'))

#2.4、使用compile来对学习过程进行配置
model2.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01),metrics=['categorical_accuracy']) 

#2.5、训练模型
modelfit2 = model2.fit(X_train, Y_train, batch_size=200, nb_epoch=20,verbose=2) 
score2 = model2.evaluate(X_test,Y_test,verbose=0)

print('Total loss on Testing Set:',score2[0])
print('Accuracy of Testing Set:',score2[1])


#绘制对比图--练集上的每次poch时accuracy的变化
import matplotlib.pyplot as plt
a = numpy.arange(1,21)
b = modelfit1.history['acc'] 
c = modelfit2.history['categorical_accuracy'] 
plt.plot(a,b)
plt.plot(a,b,'ob')
plt.plot(a,c)
plt.plot(a,c,'og')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training")
plt.legend(('loss=mse','','loss=categorical_crossentropy'),bbox_to_anchor=(1,0.7))
plt.show()
