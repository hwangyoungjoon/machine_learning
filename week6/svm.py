import numpy as np
import sys
import io
import os
import random
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


def load_dataset(filename):
    data_mat=[]
    label_mat=[]
    fr=open(filename)
    for line in fr.readlines():
        line_arr=line.strip().split("\t")
        data_mat.append([float(line_arr[0]),float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat,label_mat

def select_jrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
        print(j)
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(data_mat,class_label,c,toler,max_iter):
    data_mat=np.mat(data_mat)
    label_mat=np.mat(class_label).transpose()
    b=0
    m,n=np.shape(data_mat)
    alpha=np.mat(np.zeros((m,1)))
    iter=0
    while(iter<max_iter):
        alpha_pairs_change=0
        for i in range(m):
            fxi=float(np.multiply(alpha,label_mat).T*(data_mat*data_mat[i,:].T))+b #클래스에 대한 예측
            Ei=fxi-float(label_mat[i]) #KKT 조건에 위배되는지 확인 #ERROR1
            if ((label_mat[i]*Ei< -toler) and (alpha[i]<c))or((label_mat[i]*Ei>toler) and (alpha[i]>0)): #알파가 c,0이랑 같지 않은지 확인
                j=select_jrand(i,m)
                print(j)
                fxj=float(np.multiply(alpha,label_mat).T*(data_mat*data_mat[j,:].T))+b
                Ej=fxj-float(label_mat[j])
                alphaIold=alpha[i].copy()
                alphaJold=alpha[j].copy()
                if (label_mat[i]!=label_mat[j]):
                    L=max(0,alpha[j]-alpha[i])
                    H=max(c,c+alpha[j]-alpha[i])
                else:
                    L=max(0,alpha[j]+alpha[i]-c)
                    H=max(c,alpha[j]+alpha[i])

                if L==H: print("L==H");continue
                eta=2.0*data_mat[i,:]*data_mat[j,:].T -data_mat[i,:]*data_mat[i,:].T-data_mat[j,:]*data_mat[j,:].T
                if eta>=0: print("eta>=0");continue
                alpha[i]-=label_mat[j]*(Ei-Ej)/eta
                alpha[j]=clipAlpha(alpha[j],H,L)
                if(abs(alpha[j]-alphaJold)<0.00001): print ("j not moving enough")
                continue
                alpha[i]+=label_mat[j]*alpha_mat[i]*(alphaJold-alphap[j])

                b1=b-Ei-label_mat[i]*(alpha[i]-alphaIold)*data_mat[i,:]*data_mat[i,:].T-data_mat[j,:]*data_mat[j,:].T
                b2=b-Ej-label_mat[i]*(alpha[i]-alphaIold)*data_mat[i,:]*data_mat[i,:].T-data_mat[j,:]*data_mat[j,:].T
                if(0<alpha[i])and(c>alpha[i]):b=b1
                elif(0<alpha[j])and(c>alpha[j]):b=b2
                else:b=(b1+b2)/2.0
                alpha_pairs_change+=1
                print("iter : %d i: %d, paris changed %d" %(iter,i,alpha_pairs_change))
        if(alpha_pairs_change==0): iter+=1
        else: iter=0
        print("iteration_number: %d" %iter)
    return b,alpha

data_arr,label_arr=load_dataset('week6/testSet.txt')

b,alphs=smoSimple(data_arr,label_arr,0.6,0.001,40)
print(b,"\n",alphs)
