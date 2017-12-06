#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import SGD,RMSprop,Adam
from keras.datasets import mnist  
from keras.utils.visualize_util import plot 
import numpy

numpy.random.seed(1234) #让keras脚本每次产生确定的数据

#导入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])  
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) 
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)


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

model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01),metrics=['categorical_accuracy'])

modelfit = model.fit(X_train, Y_train, batch_size=200, nb_epoch=10,verbose=2) 
score1 = model.evaluate(X_test,Y_test,verbose=0)

print('Total loss on Testing Set:',score1[0])
print('Accuracy of Testing Set:',score1[1])

plot(model,to_file='relu.jpg',show_shapes=True)

#画随epoch变化训练集上的categorical_accuracy的变化情况
import matplotlib.pyplot as plt
a = numpy.arange(1,11)
b = modelfit.history['categorical_accuracy'] 
plt.plot(a,b)
plt.plot(a,b,'or')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("training")
plt.show()
