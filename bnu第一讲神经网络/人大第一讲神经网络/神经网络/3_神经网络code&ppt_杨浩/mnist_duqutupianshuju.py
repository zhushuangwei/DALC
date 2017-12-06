#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
    data = np.empty((42000,1,28,28),dtype="float32")
    label = np.empty((42000,),dtype="uint8")
    imgs = os.listdir("./mnist")  #获取mnist文件夹下所有图片的名字
    num = len(imgs)
    for i in range(num):
	img = Image.open("./mnist/"+imgs[i]) #open第i个图片
	arr = np.asarray(img,dtype="float32") #将img的28*28像素提取出来，写成array，得到的试一个（28,28）的矩阵
	data[i,:,:,:] = arr #将数据写成一个大矩阵，就是所有图片的像素矩阵合并到一起
	label[i] = int(imgs[i].split('.')[0]) #获取图片的名称 即0,1...9这个标签
    return data,label


#上面的(28,28）可以通过查看一张图片的属性查看，图片的size属性返回的一个元组，有两个元素，其值为象素意义上的宽和高。
#代码为from PIL import Image
#     im = Image.open("图片名.jpg")
#     im.size
#还有format--图像的源格式,mode--这个定义我也不是很清楚，自行百度。



