import numpy as np
import os
# import logistic_regression


def load_dataset(filename):
    fr=open(filename)
    data_x=[]
    data_y=[]
    for line in fr.readlines():
        current_line=line.strip().split("\t")
        line_arr=[]
        for i in range(21):
            line_arr.append(float(current_line[i]))
        data_x.append(line_arr)
        data_y.append(float(current_line[21]))
    return data_x,data_y

def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

def stocGradAscent1(data_mat,class_label,iter_num):
    m,n=np.shape(data_mat)
    weights=np.ones(n)
    for j in range(iter_num):
        data_idx=list(range(m))
        for i in range(m):
            alpha=4/(1.0+i+j)+0.0001 # 이터레이션이 증가함에 따라 알파값이 줄어듬
            rand_idx=int(np.random.uniform(0,len(data_idx)))
            h=sigmoid(sum(data_mat[rand_idx]*weights))
            error=class_label[rand_idx]-h
            weights=weights+alpha*error*np.array(data_mat[rand_idx])
            data_idx.remove(data_idx[rand_idx])
    return weights

def classifyVector(inx,weights):
    prob=sigmoid(sum(inx*weights))
    if prob>0.5: return 1.0
    else: return 0.0

def test(test_file,weights):
    test_x,test_y=load_dataset(test_file)
    error_count=0
    num_test_vec=len(test_y)
    for i in range(len(test_y)):
        if int(classifyVector(np.array(test_x[i]),weights))!=int(test_y[i]):
            error_count+=1
    error_rate=(float(error_count)/num_test_vec)
    print("the error rate is: %f" %error_rate)
    return error_rate

train_x,train_y=load_dataset("week5/horseColicTraining.txt")
train_weights=stocGradAscent1(train_x,train_y,100)
print(train_weights)
error_rate=test("week5/horseColicTest.txt",train_weights)
