import numpy as np
import os
import sys
import io
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt


sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
print(os.getcwd())


def load_data(filename,delim="\t"):
    fr=open(filename)
    string_arr=[line.strip().split(delim) for line in fr.readlines()]
    data_arr=[list(map(float,line)) for line in string_arr]
    return np.mat(data_arr)

def pca(data_mat,top_n_feat=9999999):
    mean_val=np.mean(data_mat,axis=0)
    # 각 컬럼별 평균
    mean_sub=data_mat-mean_val
    covari_mat=np.cov(mean_sub,rowvar=0) #covariance matrix 고유분해
    eig_val,eig_vect=np.linalg.eig(np.mat(covari_mat))
    # print(eig_val)
    eig_val_sort=np.argsort(eig_val)# 가장작은것부터 큰것까지 index 나열
    eig_val_sort=eig_val_sort[:-(top_n_feat+1):-1] #원치않는 차원을 자른다.
    # print(eig_vect)
    red_eig_vect=eig_vect[:,eig_val_sort] #큰 고유벡터부터 가지고 온다.
    # print(red_eig_vect)
    low_dim_mat=mean_sub*red_eig_vect# 새로운 차원으로 변환
    recon_mat=(low_dim_mat*red_eig_vect.T)+mean_val
    return low_dim_mat,recon_mat

def replaceNanWithMean():
    datMat = load_data('secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal  #평균값 채우기
    return datMat

# data=load_data("Desktop/study/데이터 분석 자료구조-기초/machine_learning_in_action/pca/testSet.txt")
# low_mat,recon_mat=pca(data,1)
#
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(data[:,0].flatten().A[0],data[:,1].flatten().A[0],marker="^",s=90)
# ax.scatter(recon_mat[:,0].flatten().A[0],recon_mat[:,1].flatten().A[0],marker="o",s=50,c="red")
# plt.show()
