
import numpy as np
import matplotlib.pyplot as plt
import os


def load_dataset():

    data_mat=[]
    label_mat=[]
    fr=open('week5/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        data_mat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        label_mat.append(int(lineArr[2]))
    return data_mat,label_mat

def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

def gradAscent(data_mat,class_label):
    data_matrix=np.mat(data_mat)
    label_mat=np.mat(class_label).transpose()
    m,n=np.shape(data_matrix)
    alpha=0.001
    max_cycle=500
    weights=np.ones((n,1))
    for k in range(max_cycle):
        h=sigmoid(data_matrix*weights)
        error=(label_mat-h)
        weights=weights+alpha*data_matrix.transpose()*error
    return weights



data_arr,label_mat=load_dataset()
# weights=gradAscent(data_arr,label_mat)

def plotBestFit(weights):
    data_mat,label_mat=load_dataset()
    data_arr=np.array(data_mat)
    n=np.shape(data_arr)[0]
    xcord_1=[]
    ycord_1=[]
    xcord_2=[]
    ycord_2=[]
    for i in range(n):
        if int(label_mat[i])==1:
            xcord_1.append(data_arr[i,1])
            ycord_1.append(data_arr[i,2])
        else:
            xcord_2.append( data_arr[i,1])
            ycord_2.append(data_arr[i,2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord_1,ycord_1,s=30,c="red",marker='s')
    ax.scatter(xcord_2,ycord_2,s=30,c="green")
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlable("x1")
    plt.ylabel("x2")
    plt.show()

def stocGradAscent0(data_mat,class_label):
    m,n=np.shape(data_mat)
    alpha=0.01
    weights=np.ones(n)
    # print(np.shape(weights))
    for i in range(m):
        h=sigmoid(sum(data_mat[i]*weights))
        error=class_label[i]-h
        weights=weights+alpha*error*np.array(data_mat[i])
    return weights



print(stocGradAscent0(data_arr,label_mat))
