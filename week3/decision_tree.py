from math import log
import operator

def calcshannonEnt(dataset):
    num_Entries=len(dataset)
    label_count={}
    for featvec in dataset:
        current_label=featvec[-1]

        if current_label not in label_count.keys(): label_count[current_label]=0
        label_count[current_label] +=1

    shannonEnt=0.0
    for key in label_count:
        prob=float(label_count[key])/num_Entries
        # print(prob)
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataset=[[1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataset,labels

def split_dataset(dataset,axis,value):
    ret_dataset=[]
    for feat_vec in dataset:
        if feat_vec[axis]==value: #데이터 자르기
            reduced_feat_vec=feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def choose_beat_feature(dataset):
    num_feature=len(dataset[0])-1
    base_entropy=calcshannonEnt(dataset)
    best_infogain=0.0
    for i in range(num_feature):
        feat_list=[example[i] for example in dataset]
        unique_val=set(feat_list)
        new_entropy=0.0
        for value in unique_val:
            subdataset=split_dataset(dataset,i,value)
            prob=len(subdataset)/float(len(dataset))
            new_entropy+=prob*calcshannonEnt(subdataset)
        infogain=base_entropy-new_entropy
        if(infogain > best_infogain):
            best_infogain=infogain
            best_feature=i
    return best_feature

def majorityCnt(classlist):
    class_count={}
    for vote in class_list:
        if vote not in class_count.key(): class_count[vote]=0
        class_count[vote]+=1
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverser=True)
    return sorted_class_count[0][0]

def createTree(dataset,labels):
    class_list=[example[-1] for example in dataset]
    if class_list.count(class_list[0])==len(class_list):
        return class_list[0]# 모든 클래스가 같아질떄 자르는 것을 멈춘다.
    if len(dataset[0])==1:# 더이상 피쳐가 없을 때 자르느느 것을 멈춘다.
        return majorityCnt(class_list)
    # print(labels)
    best_feat=choose_beat_feature(dataset)
    #print(best_feat)
    best_feat_label=labels[best_feat]
    mytree={best_feat_label:{}}
    #print(mytree)
    del(labels[best_feat])
    feat_val=[example[best_feat] for example in dataset]
    # print(feat_val)
    unique_val=set(feat_val)
    for value in unique_val:
        sub_label=labels[:]# 분류를 하고난 사라진 변수 라벨 제거
        mytree[best_feat_label][value]=createTree(split_dataset(dataset,best_feat,value),sub_label)
    return mytree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    print(firstStr)
    secondDict = inputTree[firstStr]
    print(secondDict)
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    

mydata,labels=createDataSet()
print(labels)
mytree=createTree(mydata, labels)
mydata,labels=createDataSet()
print(classify(mytree,labels,[1,0]))
