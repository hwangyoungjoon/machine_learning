import numpy as np
import sys
import io
import os
import random
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
print(os.getcwd())
def load_dataset(filename):
    num_feat=len(open(filename).readline().split('\t'))-1
    data_mat=[]
    label_mat=[]
    fr=open(filename)
    for line in fr.readlines():
        linearr=[]
        curline=line.strip().split('\t')
        for i in range(num_feat):
            linearr.append(float(curline[i]))
        data_mat.append(linearr)
        label_mat.append(float(curline[-1]))
    return data_mat,label_mat

def stand_regres(x_arr,y_arr):
    x_mat=np.mat(x_arr)
    y_mat=np.mat(y_arr).T
    xTx=x_mat.T*x_mat
    if np.linalg.det(xTx)==0:
        print("this matrix is singular, cannot have inverse matrix")
        return
    ws=xTx.I*(x_mat.T*y_mat)
    return ws



def locally_weighted_linear_regression(test_point,x_arr,y_arr,k=1.0):
    x_mat=np.mat(x_arr)
    y_mat=np.mat(y_arr).T
    m=np.shape(x_mat)[0]
    weight=np.mat(np.eye((m)))

    for j in range(m):
        diff_mat=test_point-x_mat[j,:]
        # print(x_mat[j,:])
        weight[j,j]=np.exp(diff_mat*diff_mat.T/(-2.0*k**2))

    xTx=x_mat.T*(weight*x_mat)
    if np.linalg.det(xTx)==0:
        print("this matrix is singlur, cannot have inverse matrix")
        return

    ws=xTx.I*(x_mat.T*(weight*y_mat))
    return test_point*ws

def locally_weighted_linear_regression_test(test_arr,x_arr,y_arr,k=1.0):
    m=np.shape(test_arr)[0]
    yhat=np.zeros(m)
    for i in range(m):
        yhat[i]=locally_weighted_linear_regression(test_arr[i],x_arr,y_arr,k)
    return yhat

def TestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rss_error(y_arr,y_hat_arr):
    return ((y_arr-y_hat_arr)**2).sum()


abx,aby=load_dataset("week7/abalone.txt")



def ridgeRegress(x_mat,y_mat,lam=0.2):
    xtx=x_mat.T*x_mat
    denom=xtx+np.eye(np.shape(x_mat)[1])*lam
    if np.linalg.det(denom)==0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws=denom.I*(x_mat.T*y_mat)
    return ws

def ridgeTest(x_arr,y_arr):
    x_mat=np.mat(x_arr)
    y_mat=np.mat(y_arr).T
    y_mean=np.mean(y_mat)
    print(y_mean)
    y_mat=y_mat-y_mean
    print(y_mat)
    x_mean=np.mean(x_mat,0)
    xVar=np.var(x_mat,0)
    x_mat=(x_mat-x_mean)/xVar
    numTestPts=30
    wMat=np.zeros((numTestPts,np.shape(x_mat)[1]))
    print(wMat)
    for i in range(numTestPts):
        ws=ridgeRegress(x_mat,y_mat,np.exp(i-10))
        print(ws)
        wMat[i,:]=ws.T
        print("\n")
    return wMat

def regulaize(x_mat):
    in_mat=x_mat.copy()
    in_mean=np.mean(in_mat,0)
    in_var=np.var(in_mat,0)
    in_mat=(in_mat-in_mean)/in_var
    return in_mat

def stage_wise(x_arr,y_arr,eps=0.01,num_iter=100): # lassos
    x_mat=np.mat(x_arr)
    y_mat=np.mat(y_arr).T
    y_mean=np.mean(y_mat,0)
    y_mat=y_mat-y_mean
    x_mat=regulaize(x_mat)
    m,n=np.shape(x_mat)
    ws=np.zeros((n,1))
    ws_test=ws.copy()
    ws_max=ws.copy()
    returnMat = np.zeros((num_iter,n))
    for i in range(num_iter):
        print(ws.T)
        lowest_error=np.inf

        for j in range(n):
            for sign in [-1,1]:
                ws_test=ws.copy()
                ws_test[j]+=eps*sign
                y_test=x_mat*ws_test
                rss_err=rss_error(y_mat.A,y_test.A)
                print(lowest_error)
                if rss_err<lowest_error:
                    lowest_error=rss_err
                    ws_max=ws_test
        ws=ws_max.copy()
        returnMat[i,:]=ws.T
    return returnMat

stage_wise(abx,aby,0.01,200)
