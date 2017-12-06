sigmod=function(x){
    return(1/(1+exp(-x)))
}
sigmod_prime<-function(x){
    return(sigmod(x)*(1-sigmod(x)))
}
x=1
y=0
w=0.6;b=0.9
#w=2;b=2
w0=w;b0=b
eta=0.15
z<-t(w)%*%x+b
a=sigmod(z)
C=-sum(y*log(a)+(1-y)*log(1-a))/length(x)
#C<-sum((y-a)**2)/2
dz=(a-y)
dw<-a%*%t(dz)
db<-dz
w=w-eta*dw
b=b-eta*db
n=0
loss<-c()
cs<-c()
N<-c(50,100,150,200,250,300)
split.screen(c(2,3))
for(i in 1:length(N)){
    while(n<N[i]){
        z<-t(w)%*%x+b
        a=sigmod(z)
        C=-sum(y*log(a)+(1-y)*log(1-a))/length(x)
        dz=(a-y)
        dw<-a%*%t(dz)
        db<-dz
        w=w-eta*dw
        b=b-eta*db
        n=n+1
        loss[n]=C
    }
    screen(i)
    plot(loss,xlim = c(0,400),ylim = c(0,0.5),xlab = "Epoch",lty = 1,pch=15,cex=0.3)
    title(paste(w0,b0,"Epoch",N[i]),cex.main=0.8)
}







