import numpy as np

def ecludsim(ina,inb):
    return 1.0/(1.0+np.linalg.norm(ina-inb))

def pearssim(ina,inb):
    if len(ina)<3:return 1.0
    return 0.5+0.5*np.corrcoef(ina,inb,rowvar=0)[0][1]

def cossim(ina,inb):
    num=float(ina.T*inb)
    denom=np.linalg.norm(ina)*np.linalg.norm(inb)
    return 0.5+0.5*(num/denom)


def stand_est(data_mat,user,sim_method,item):
    n=np.shape(data_mat)[1]
    sim_total=0.0
    rat_sim_total=0.0
    for j in range(n):
        user_rating=data_mat[user,j]
        # print(user_rating)
        if user_rating==0: continue
        # print((data_mat[:,item]))
        # print(np.logical_and(data_mat[:,item].A>0,data_mat[:,j].A>0))
        over_lap=np.nonzero(np.logical_and(data_mat[:,item].A>0,data_mat[:,j].A>0))[0] #아이템들간에 값이 모두 채워져있는것들만의 인덱스
        if len(over_lap)==0: similarity=0 #서로 채워져있는 것이 하나도 안맞으면 0
        else: similarity= sim_method(data_mat[over_lap,item],data_mat[over_lap,j]); #값이 있는 것들만 따로 벡터로 만든다. print(data_mat[over_lap,item])
        print("the %d and %d similarity is %f"%(item,j,similarity))
        sim_total+=similarity
        rat_sim_total+=similarity*user_rating
    if sim_total==0: return 0
    else: return rat_sim_total/sim_total

def recommend(data_mat,user,n=3,sim_method=cossim,estMethod=stand_est):
    unrated_item=np.nonzero(data_mat[user,:].A==0)[1] # 레이팅이 되지 않은 아이템 찾기
    if len(unrated_item)==0: return "you rated everything"
    item_score=[]
    for item in unrated_item:
        estimated_score=estMethod(data_mat,user,sim_method,item)
        item_score.append((item,estimated_score))
        # print(item_score)
    return sorted(item_score,key=lambda jj:jj[1],reverse=True)[:n]

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# data=np.mat(loadExData())
# data[0,1]=data[0,0]=data[1,0]=data[2,0]=4
# data[3,3]=2
# print(recommend(data,2),"\n")
# print(recommend(data,2,sim_method=ecludsim),"\n")
# print(recommend(data,2,sim_method=pearssim),"\n")

def svdest(data_mat,user,sim_method,item):
    n=np.shape(data_mat)[1]
    sim_total=0.0
    rat_sim_total=0.0
    u,sig,vt=np.linalg.svd(data_mat)
    sig4=np.mat(np.eye(4)*sig[:4])
    x_formed_item=data_mat.T*u[:,:4]*sig4.I #저차원으로 바꾼다.
    for j in range(n):
        user_rating=data_mat[user,j]
        if user_rating==0 or j==item: continue
        similarity=sim_method(x_formed_item[item,:].T,x_formed_item[j,:].T)
        print("the %d and %d similarity is :%f"%(item,j,similarity))
        sim_total+=similarity
        rat_sim_total+=similarity*user_rating
    if sim_total==0: return 0
    else: return rat_sim_total/sim_total


data=np.mat(loadExData2())
print(recommend(data,1,estMethod=svdest))
