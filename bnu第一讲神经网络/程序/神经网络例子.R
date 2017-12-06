sigmod=function(x){
    return(1/(1+exp(-x)))
}
sigmod_prime<-function(x){
    return(sigmod(x)*(1-sigmod(x)))
}
set.seed(1)
x<-c(3.0,1.0,2.0)
y<-c(1,0)
# #w1<-matrix(c(0.9,0.8,0.7,-0.6,
#             -0.7,0.6,-0.5,0.4,
#             0.5,-0.4,-0.3,-0.2),nrow = 3,byrow = T)
#b1<-c(1,1,1,1)
w1<-matrix(rnorm(3*4,0,1),nrow = 3,byrow = T)
b1<-rnorm(4,0,1)
z1<-t(w1)%*%x+b1
a1=sigmod(z1)
# w2<-matrix(c(0.3,0.2,0.1,0.5,
#              0.4,0.3,0.6,0.4),nrow=4,byrow = T)
# b2<-c(0.1,0.1)
w2<-matrix(rnorm(4*2,0,1),nrow = 4,byrow = T)
b2<-rnorm(2,0,1)
z2<-t(w2)%*%a1+b2
a2<-sigmod(z2)
a2
C<-sum((y-a2)**2)/2

#反向传播最后一层
dz2=(a2-y)*sigmod_prime(z2)
dw2<-a1%*%t(dz2)
db2<-dz2
#反向传播倒数第二层
dz1=w2%*%dz2*sigmod_prime(z1)
dw1=x%*%t(dz1)
db1=dz1
#设置学习速率
eta=0.5
n=0
#梯度下降 
w1=w1-eta*dw1
w2=w2-eta*dw2
b1=b1-eta*db1
b2=b2-eta*db2
loss=c()
#设置循环 
while(C>0.0001){
    z1<-t(w1)%*%x+b1
    a1<-sigmod(z1)
    z2<-t(w2)%*%a1+b2
    a2=sigmod(z2)
    dz2=(a2-y)*sigmod_prime(z2)
    dw2<-a1%*%t(dz2)
    db2<-dz2
    dz1=w2%*%dz2*sigmod_prime(z1)
    dw1=x%*%t(dz1)
    db1=dz1
    C<-sum((y-a2)**2)/2
    w1=w1-eta*dw1
    w2=w2-eta*dw2
    b1=b1-eta*db1
    b2=b2-eta*db2
    n=n+1
    loss[n]=C
}
plot(loss)
z1<-t(w1)%*%x+b1
a1=sigmod(z1)
z2<-t(w2)%*%a1+b2
a2<-sigmod(z2)
a2
n
C



x1<-c(4,1,3)
x2<-c(2,3,4)
x3<-c(7,8,1)
x4<-c(3,4,5)
x5<-c(2,4,5)
x<-cbind(x1,x2,x3,x4,x5)
y<-matrix(c(1,0,1,1,0,
            1,1,0,1,0),nrow = 2,byrow = T)
w1<-matrix(c(0.9,0.8,0.7,-0.6,
             -0.7,0.6,-0.5,0.4,
             0.5,-0.4,-0.3,-0.2),nrow = 3,byrow = T)
b1<-c(1,1,1,1)
z1<-t(w1)%*%x+b1
a1=sigmod(z1)
w2<-matrix(c(0.3,0.2,0.1,0.5,
             0.4,0.3,0.6,0.4),nrow=4,byrow = T)
b2<-c(0.1,0.1)
z2<-t(w2)%*%a1+b2
a2<-sigmod(z2)
a2
C<-sum((y-a2)**2)/2

#反向传播最后一层
dz2=(a2-y)*sigmod_prime(z2)
dw2<-a1%*%t(dz2)
db2<-mean(dz2)
#反向传播倒数第二层
dz1=w2%*%dz2*sigmod_prime(z1)
dw1=x%*%t(dz1)
db1=mean(dz1)
#设置学习速率
eta=0.5
n=0
#梯度下降 
w1=w1-eta*dw1
w2=w2-eta*dw2
b1=b1-eta*db1
b2=b2-eta*db2
loss=c()
#设置循环 
while(C>0.0001){
    z1<-t(w1)%*%x+b1
    a1<-sigmod(z1)
    z2<-t(w2)%*%a1+b2
    a2=sigmod(z2)
    dz2=(a2-y)*sigmod_prime(z2)
    dw2<-a1%*%t(dz2)
    db2<-dz2
    dz1=w2%*%dz2*sigmod_prime(z1)
    dw1=x%*%t(dz1)
    db1=dz1
    C<-sum((y-a2)**2)/2
    w1=w1-eta*dw1
    w2=w2-eta*dw2
    b1=b1-eta*db1
    b2=b2-eta*db2
    n=n+1
    loss[n]=C
}
plot(loss)
z1<-t(w1)%*%x+b1
a1=sigmod(z1)
z2<-t(w2)%*%a1+b2
a2<-sigmod(z2)
a2
y
n
C





















