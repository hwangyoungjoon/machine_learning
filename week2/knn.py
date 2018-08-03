import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=["A","A","B","B"]
    return group, labels

group,labels=createDataSet()

def knn_classify(inX,dataSet,labels,k):
    """
    input
        inX: 벡터 (1xn) input 비교할 데이터
        dataSet: 전체 데이터 m사이즈(nxm)
        labels: 데이터 라벨(1XM벡터)
        K= knn 투표에 이용할 값

    output
        the most popular class
    """
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet #행렬 element 뺄셈
    sqDiffmat=diffMat**2 #행렬 element ^2
    sqDistances=sqDiffmat.sum(axis=1)
    distance=sqDistances**0.5
    #print(distance)
    sortedDistanceIndicies=distance.argsort() #return 거리들의 우선순위
    #print(sortedDistanceIndicies)
    classCount={}
    for i in range(k):
        votelabel=labels[sortedDistanceIndicies[i]] #우선순위가 높은 것들 모으기

        classCount[votelabel]=classCount.get(votelabel,0)+1
    print(classCount)
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr=open(filename)
    numberOfLines = len(fr.readlines())
    #print(numberOfLines)
    returnMat=np.zeros((numberOfLines,3)) #number0flist x 3열로 만듬
    #print(returnMat)
    classLabelVector=[]
    fr=open(filename)
    index=0
    for line in fr.readlines():
        line=line.strip()  #양쪽 여백 지우기
        #print(line)
        listFormLine=line.split('\t')
        #print(listFormLine[-1])
        returnMat[index,:]=listFormLine[0:3]
        classLabelVector.append((listFormLine[-1]))
        index+=1
    return returnMat,classLabelVector

#datingdatamat,datinglabel=file2matrix("datingTestSet.txt")

def autoNorm(dataset):
    minVals=dataset.min(0) #row 이기에 0을써서 최소값가져옴
    maxVals=dataset.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataset))
    m=dataset.shape[0]
    normDataSet=dataset-np.tile(minVals,(m,1)) # 1000*1 벡터 만듬
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#normdata,range,minval=autoNorm(datingdatamat)
#print(normdata

def classtest():
    horatio=0.10
    datingdatamat,datinglabel=file2matrix("datingTestSet.txt")
    normdata,ranges,minvals=autoNorm(datingdatamat)
    m=normdata.shape[0]
    print(m)
    num_test_vecs=int(m*horatio)

    error=0.0
    for i in range(num_test_vecs):
        classifier_result=knn_classify(normdata[i,:],normdata[num_test_vecs:m,:],
        datinglabel[num_test_vecs:m],8)
        print("the classifier came back with: %s, the real answer is: %s"
        %(classifier_result,datinglabel[i]))

        if (classifier_result!= datinglabel[i]): error+=1.0
    print("the total error rate is : %f" %(error/float(num_test_vecs)))
    return datingdatamat

def classifyPerson():
    result_list=["not at all","in small doses","in large doses"]
    percentage=float(raw_input("percentage of time spent playing video games"))
    ffmiles=float(raw_input("frequent flier miles earned per year?"))
    icecream=float(raw_input("liters of ice cream consumed per year"))
    datingdatamat,datinglabel=file2matrix("datingTestSet.txt")
    normMat,ranges,minVals=autoNorm(datingdatmat)
    inarr=np.array([ffmiles,percentage,ice])
    classifier_result=knn_classify((inarr-minVals)/ranges,normMat,datinglabel,3)
    print("you will probably like this person:",result_list[clasifier_result-1])

def img2vector(filename):
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwalabel=[]
    training_file_list=listdir("digits/trainingDigits")
    m=len(training_file_list)
    trainmat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=training_file_list[i]
        file_str=fileNameStr.split(".")[0]
        class_num_str=int(file_str.split("_")[0])
        hwalabel.append(class_num_str)
        trainmat[i,:]=img2vector("digits/trainingDigits/%s" % fileNameStr)
    test_file_list=listdir("digits/testDigits")
    errorcount=0
    mtest=len(test_file_list)
    for i in range(mtest):
        fileNameStr=test_file_list[i]
        file_str=fileNameStr.split(".")[0]
        vector_test=img2vector("digits/testDigits/%s"%fileNameStr)
        classifier_result=knn_classify(vector_test,trainmat,hwalabel,3)
        if (classifier_result!=class_num_str):errorcount+=1.0
    print("\n total number of errors is: %d" % errorcount)
    print("\n totalt error rate is:%d" % (errorcount/float(mtest)))


handwritingClassTest()
