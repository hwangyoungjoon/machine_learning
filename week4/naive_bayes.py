import numpy as np


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec



def createVocabList(dataset):
    vocabSet=set([])
    for document in dataset:
        vocabSet = vocabSet | set(document) #두개의 튜플 합치기 기능
    return list(vocabSet)

def set_of_words2vec(vocab_list,input_set):
    return_vec=[0]*len(vocab_list)  # 사전에 있는 수만큼의 0리스트 생성
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)]=1
        else: print("the word: %s is not in vocab_list" % word)
    return return_vec

def bag_of_word2vec(vocab_list,input_set):
    return_vec=[0]*len(vocablist)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocablist.index(word)]+=1
    return return_vec # set of words2vec계량 버전 그 단어가 있을 때마다더해준다


def trainNB0(trainMatrix,trainCategory):
    num_train_docs=len(trainMatrix) #문서가 몇개인지
    num_words=len(trainMatrix[0]) #사전에 있는 단어 갯수 -total 단어
    pAbusive=sum(trainCategory)/float(num_train_docs)
    p0_num=np.ones(num_words)
    p1_num=np.ones(num_words)
    p0_denom=2.0
    p1_denom=2.0

    for i in range(num_train_docs):
        if trainCategory[i]==1:
            p1_num+=trainMatrix[i]
            p1_denom+=sum(trainMatrix[i])
#            print(p1_num)
            # print(trainMatrix[i])
#            print(p1_denom)
        else:
            p0_num+=trainMatrix[i]
            p0_denom=sum(trainMatrix[i])
    p0vect=np.log(p0_num/p0_denom)  # 각단어당 p(w/c=0)벡터 구함 각
    p1vect=np.log(p1_num/p1_denom)  # 각단어당 p(w/c=1)벡터 구함 각
    return p0vect,p1vect,pAbusive

def classifyNB(vec2classify,p0vec,p1vec,pclass1):
    p1=sum(vec2classify*p1vec)+np.log(pclass1)
    p0=sum(vec2classify*p0vec)+np.log(1.0-pclass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    doc_list,list_class=loadDataSet()
    myvocab_list=createVocabList(doc_list)
    train_mat=[]
    for doc in doc_list:
        train_mat.append(set_of_words2vec(myvocab_list,doc))
    p0v,p1v,pab=trainNB0(train_mat,list_class)
    # print(p0v,p1v,pab)
    testEntry = ['love', 'my', 'dalmation']
    this_doc=np.array(set_of_words2vec(myvocab_list,testEntry))
    print(this_doc)
    print("classified as:",classifyNB(this_doc,p0v,p1v,pab))
