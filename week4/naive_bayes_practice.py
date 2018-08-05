
import re
import naive_bayes
import os


# mysent="this is the best book on python or M.L. I have ever laid eyes upon."
# regEX=re.compile("\\W*")
# list_of_tokens=regEX.split(mysent)
# list_of_tokens=[token.lower() for token in list_of_tokens if len(token)>0]
# print(list_of_tokens)

def textparse(bigstring):
    list_of_tokens = re.split(r'\W*', bigstring)
    return [token.lower() for token in list_of_tokens if len(token)>2]
#print(listdir('email/spam'))
def spamTest():
    doc_list=[]
    class_list=[]
    full_text=[]
    for i in range(1,26):
        word_list=textparse(open("email/spam/%d.txt" %i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list=textparse(open('email/ham/%d.txt' %i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list=createVocabList(doc_list)
    training_set=range(50)
    test_set=[]
    for i in range(10):
        rand_idx=int(random.uniform(0,len(training_set)))
        test_set.append(training_set[rand_idx])
        del(training_set[rand_idx])
    train_mat=[]
    train_class=[]
    for doc_idx in training_set:
        train_mat.append(set_of_words2vec(vocab_list,doc_list[doc_idx]))
        train_class.append(class_list[doc_idx])
    p0v,p1v,pspam=trainNB0(train_mat,train_class)
    error_count=0
    for doc_idx in test_set:
        word_vector=set_of_words2vec(vocab_list,doc_list[doc_idx])
        if classifyNB(word_vector,p0v,p1v,pspam) != class_list[doc_idx]:
            error_count+=1
            print("classification error",doc_list[doc_idx])
    print("the error rate is: ", float(error_count)/len(test_set))

spamTest()
