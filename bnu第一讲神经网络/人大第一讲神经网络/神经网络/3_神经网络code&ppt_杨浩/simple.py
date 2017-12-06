#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.models import Sequential               #加载模型
from keras.layers.core import Dense,Dropout,Activation   #加载2D层和激活函数
from keras.optimizers import SGD,Adam                  #加载优化器

#建立神经网络

#某一层
model = Sequential()
model.add(Dense(out_dim=50,input_dim=100))
model.add(Activation('sigmoid'))

上面等价于
model = Sequential([
Dense(output_dim=50,input_dim=100),
Activation('sigmoid')])


####一、简单的三层的神经网络
#1.1、建立输入层和隐层
model = Sequential()  
model.add(Dense(output_dim=32,input_dim=784))  
model.add(Activation('sigmoid')) 
model.add(Dropout(0.2))

#1.2、建立输出层
model.add(Dense(output_dim=10))  
model.add(Activation('softmax'))


#上面等价于
model = Sequential([
Dense(32,input_dim=784),
Activation('sigmoid'),
Dense(10),
Activation('softmax'),
])

#2D层Dense--是常用的全连接层
#Dense(output_dim, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
#output_dim--代表该层的输出维度，init--初始化方法，input_dim：输入数据的维度。
#输入层必须有input_dim,隐层必须有output_dim，输出层必须有output_dim。

#Dropout层--在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接，用于防止过拟合。










####二、编译--对学习过程的配置
#在训练模型之前，我们需要通过compile来对学习过程进行配置。compile接收三个参数：优化器optimizer、损失函数loss、指标列表metrics        （compile（optimizer='',loss='',metrics=[]））

#多分类问题
model.compile(optimizer='SGD',       #SGD是随机梯度下降的优化方法
loss='categorical_crossentropy',     #categorical_crossentropy是多类交叉熵的损失函数
metrics=['accuracy'])

#二分类问题
model.compile(optimizer='SGD',
loss='binary_crossentropy',
metrics=['accuracy'])


#优化器--optimizers,SGD--梯度下降法,Adagrad,Adadelta,RMSprop,Adam...
#SGD--SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#Adam--Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#损失函数--loss，有很多，这里主要有mse和categorical_crossentropy。
#当"categorical_crossentropy"作为目标函数时,标签应该为多类模式,即one-hot编码的向量

#metrics，性能评估模块提供了一系列用于模型性能评估的函数。
#主要用到了categorical_accuracy:对多分类问题,计算在所有预测值上的平均正确率












####三、训练
#Keras以Numpy数组作为输入数据和标签的数据类型。训练模型一般使用fit函数。
model.fit(data, labels, nb_epoch=10, batch_size=32)

#fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1,callbacks=[],validation_split=0.0, shuffle=True, class_weight=None, sample_weight=None)
#x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list
#y:标签，numpy array
#batch_size：整数，指定进行梯度下降时每个batch包含的样本数。
#nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。
#verbose：日志显示，0-不输出任何信息，1-进度条数据，2-每个epoch输出一行记录,一般选取2，默认是1

model.evaluate(X_test,Y_test,verbose=0) 

#evaluate返回在测试集上的误差，是一个list，一个是Total loss，一个是Accuracy，默认的verbose=1

model.predict_classes(X_test,verbose=0)#返回输入数据的类别预测结果，默认的verbose=1
model.predict_proba(X_test,verbose=0)#输入数据属于各个类别的概率，默认的verbose=1
