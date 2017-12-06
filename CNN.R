sigmod=function(x){
    return(1/(1+exp(-x)))
}
sigmod_prime<-function(x){
    return(sigmod(x)*(1-sigmod(x)))
}
ReLU<-function(z1){
    z=matrix(0,nrow = nrow(z1),ncol = ncol(z1))
    for (i in c(1:nrow(z1))){
        for (j in c(1:ncol(z1)))
            z[i,j]=max(0,z1[i,j])
    }
    return(z)
}
set.seed(1)
conv<-function(x,filter,step=c(1,1)){
    colM=ncol(x)#the col number of image 
    rowM=nrow(x)#the row number of image
    colN=ncol(filter)
    rowN=nrow(filter)
    h<-matrix(0,nrow=rowM-rowN+2-step[1],ncol=colM-colN+2-step[2])
    for (k in c(1:ncol(h))){
        for(l in c(1:nrow(h))){
            for (i in c(1:colN)){
                for (j in c(1:rowN)){
                    h[l,k]= h[l,k]+filter[i,j]*x[i+l-2+step[1],j+k-2+step[2]]
                }
            }
        }
    }
    return(h)
}
pooling<-function(h,size=2,type='MAX'){
    colN=ncol(h)/size
    rowN=nrow(h)/size
    p<-matrix(0,nrow=rowN,ncol=colN)
    position<-p
    for( i in c(1:nrow(p))){
        for (j in c(1:ncol(p))){
            block<-h[(size*(i-1)+1):(size*i),(size*(j-1)+1):(size*j)]
            if (type=='mean'){
                p[i,j]=mean(block)
            } else {
                p[i,j]=max(block)
                position[i,j]=which.max(block)
            }     
        }
    }
    if (type=='mean'){
        data<-list(p=p)
    } else {
        data<-list(p=p,position=position)
    }  
    return(data)
}
upsample<-function(data,size=2,type ='MAX'){
    dalta=data$p
    colN=ncol(dalta)*size
    rowN=nrow(dalta)*size
    pedalta<-matrix(0,nrow=rowN,ncol=colN)
    if (type=='MAX'){
        position=data$position
        for (i in c(1:ncol(dalta))){
            for(j in c(1:nrow(dalta))){
                pedalta[(size*(i-1)+1):(size*i),(size*(j-1)+1):(size*j)][position[i,j]]=dalta[i,j]
            }
        }
    } else{
        for (i in c(1:ncol(dalta))){
            for(j in c(1:nrow(dalta))){
                pedalta[(size*(i-1)+1):(size*i),(size*(j-1)+1):(size*j)]=dalta[i,j]/length(dalta)
            }
        }
    }
    return(pedalta)
}
rot180<-function(x){
    y<-matrix(0,nrow=nrow(x),ncol=ncol(x))
    for (i in c(1:nrow(x))){
        y[ncol(x)-i+1,]=c(rev(x[i,]))
    }
    return(y)
}
dw<-function(x,h){
    h=rot180(h)
    colM=ncol(x)#the col number of image 
    rowM=nrow(x)#the row number of image
    colN=ncol(h)
    rowN=nrow(h)
    dw<-matrix(0,nrow=rowM-rowN+2-step[1],ncol=colM-colN+2-step[2])
    for (k in c(1:ncol(dw))){
        for(l in c(1:nrow(dw))){
            for (i in c(1:colN)){
                for (j in c(1:rowN)){
                    dw[l,k]= dw[l,k]+h[i,j]*x[i+l-2+step[1],j+k-2+step[2]]
                }
            }
        }
    }
    return(dw)
}
x<-c(0,0,1,1,0,0,
     0,1,0,0,1,0,
     1,0,0,0,0,1,
     1,0,0,0,0,1,
     0,1,0,0,1,0,
     0,0,1,1,0,0)
y<-c(1,0)
x<-matrix(x,nrow = sqrt(length(x)),byrow=T)
image(x,col = c(0,1))
w<-matrix(rnorm(3*3,0,1),nrow = 3,byrow = T)
b1=rnorm(1,0,1)
step=c(1,1)
z1=conv(x,w,step)+b1#convolved
a1=ReLU(z1)
type='MAX'
data<-pooling(a1,size=2,type =type)#pooling
z2=data$p
a2=z2
a2=as.vector(a2)
w1<-matrix(rnorm(4*5,0,1),nrow = 4,byrow = T)
b2<-rnorm(5,0,1)
z3<-t(w1)%*%a2+b2
a3=sigmod(z3)
w2<-matrix(rnorm(5*2,0,1),nrow = 5,byrow = T)
b3<-rnorm(2,0,1)
z4<-t(w2)%*%a3+b3
a4<-sigmod(z4)
C<-sum((y-a4)**2)/2

#反向传播最后一层
dz4=(a4-y)*sigmod_prime(z4)
dw4<-a3%*%t(dz4)
db4<-dz4
#反向传播倒数第二层
dz3=w2%*%dz4*sigmod_prime(z3)
dw3=a2%*%t(dz3)
db3=dz3

#设置学习速率
eta=0.3
n=0
#梯度下降 
w1=w1-eta*dw3
w2=w2-eta*dw4
b2=b2-eta*db3
b3=b3-eta*db4

dz2=w1%*%dz3*sigmod_prime(a2)
dz2<-matrix(dz2,nrow=sqrt(length(dz2)))
if (type=='MAX'){
    dz2<-list(p=dz2,position=data$position)
} else {
    dz2<-list(p=dz2)
}
dz1<-upsample(dz2)
dw1<-dw(x,dz1)
db1<-sum(dz1)
w=w-eta*dw1
b1=b1-eta*db1
loss=c()
while (n<1000){
    z1=conv(x,w,step)+b1#convolved
    a1=ReLU(z1)
    data<-pooling(a1,size=2,type =type)#pooling
    z2=data$p
    a2=z2
    a2=as.vector(a2)
    
    w1<-matrix(rnorm(4*5,0,1),nrow = 4,byrow = T)
    b2<-rnorm(5,0,1)
    z3<-t(w1)%*%a2+b2
    a3=sigmod(z3)
    w2<-matrix(rnorm(5*2,0,1),nrow = 5,byrow = T)
    b3<-rnorm(2,0,1)
    z4<-t(w2)%*%a3+b3
    a4<-sigmod(z4)
    a4
    C<-sum((y-a4)**2)/2
    
    #反向传播最后一层
    dz4=(a4-y)*sigmod_prime(z4)
    dw4<-a3%*%t(dz4)
    db4<-dz4
    #反向传播倒数第二层
    dz3=w2%*%dz4*sigmod_prime(z3)
    dw3=a2%*%t(dz3)
    db3=dz3
    #梯度下降 
    w1=w1-eta*dw3
    w2=w2-eta*dw4
    b2=b2-eta*db3
    b3=b3-eta*db4
    dz2=w1%*%dz3*sigmod_prime(a2)
    dz2<-matrix(dz2,nrow=sqrt(length(dz2)))
    if (type=='MAX'){
        dz2<-list(p=dz2,position=data$position)
    } else {
        dz2<-list(p=dz2)
    }
    dz1<-upsample(dz2)
    dw1<-dw(x,dz1)
    db1<-sum(dz1)
    w=w-eta*dw1
    b1=b1-eta*db1
    n=n+1
    loss[n]=C
}
plot(loss)












