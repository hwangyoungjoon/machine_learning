
import numpy as np
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
# print(os.getcwd())
def load_dataset(filename):
    data_matrix=[]
    fr=open(filename)
    for line in fr.readlines():
        curline=line.strip().split("\t")
        fltline=list(map(float,curline))
        data_matrix.append(fltline)
    return data_matrix

def dist_eclud(vec_a,vec_b):
    return np.sqrt(np.sum(np.power(vec_a-vec_b,2)))

def random_centroid(dataset,k):
    n=np.shape(dataset)[1]
    centroid=np.mat(np.zeros((k,n)))

    for j in range(n):  #컬럼별로 가장 작은 값 찾기 센트로이드 결정해주기
        min_j=np.min(dataset[:,j])
        range_j=float(np.max(dataset[:,j])-min_j)
        centroid[:,j]=min_j+range_j*np.random.rand(k,1)
    return centroid

def kmean(dataset,k,dis_method=dist_eclud,create_cent=random_centroid):
    m=np.shape(dataset)[0]
    cluster_assment=np.mat(np.zeros((m,2))) #클러스터의 클래스를 저장하는 값(칼럼 1: 클래스, 칼럼2: 에러)
    centroid=create_cent(dataset,k) #센트로이드 초기화

    cluster_changed=True
    while cluster_changed:
        cluster_changed=False
        for i in range(m):
            min_dist=np.inf
            min_index=-1
            for j in range(k):  #가장 가까운 centroid를 찾는다.
                dist_ji=dis_method(centroid[j,:],dataset[i,:])
                if dist_ji<min_dist:
                    min_dist=dist_ji #가까운 거리를 측정
                    min_index=j  # 클래스 할당
            if cluster_assment[i,0] !=min_index:
                cluster_changed=True
            cluster_assment[i,:]=min_index,min_dist**2
                # print(cluster_assment) #정해진 센트로이드에서 값을 할당
        # print(centroid,"\n")
        # print(cluster_assment)
        # cent=1
        # print(np.nonzero(cluster_assment[:,0].A==cent))
        # print(np.nonzero(cluster_assment[:, 0].A == cent)[0])
        # print(dataset[np.nonzero(cluster_assment[:,0])])
        for cent in range(k): #센트로이드 업데이트
            pts_in_cluster=dataset[np.nonzero(cluster_assment[:,0].A==cent)[0]] # np.nonzero는 그 위치의 값이 0이 아닌 index를 반환한다.
            # 여기선 같은 클래스가 있는 것 데이터를 모은다.
            # print((cluster_assment[:0]))
            centroid[cent,:]=np.mean(pts_in_cluster,axis=0) # 같은 클래스 데이터 포인트들의 센트로이를 만든다.
    return centroid,cluster_assment

def bisecting_kmeans(dataset,k,dist_method=dist_eclud):
    m=np.shape(dataset)[0]
    cluster_assment=np.mat(np.zeros((m,2)))
    centroid0=np.mean(dataset,axis=0).tolist()[0]
    cent_list=[centroid0]
    for j in range(m): # 초기 error 계산
        cluster_assment[j,1]=dist_method(np.mat(centroid0),dataset[j,:])**2
    # print(cluster_assment)
    while (len(cent_list)<k):
        lowest_sse=np.inf
        for i in range(len(cent_list)):
            # print(i)
            pts_in_cluster=dataset[np.nonzero(cluster_assment[:,0].A==i)[0]] #클러스터 i에 있는 데이터 포인트를 가져온다.
            centorid_mat,split_clust=kmean(pts_in_cluster,2,dist_method) # i클러스터에 있는 것을 뽑아서 그것을 쪼개본다.
            # print(centorid_mat)
            # print(centorid_mat,split_clust,"\n")
            sse_split=np.sum(split_clust[:,1])
            # print(np.nonzero(cluster_assment[:,0].A==i)) #모든 인덱스가 나옴 즉 (1,0)
            # print(cluster_assment[np.nonzero(cluster_assment[:,0].A==i)])
            # print(cluster_assment[np.nonzero(cluster_assment[:, 0].A == i)[0]) #그 클래스에 해당하는 클래스와 에러값이 나옴
            # print(cluster_assment[np.nonzero(cluster_assment[:,0].A==i)[0],0])# 클래스 값만 도출
            sse_not_split=np.sum(cluster_assment[np.nonzero(cluster_assment[:,0].A!=i)[0],1])
            print("sse_split and sse_not_split:",sse_split,"      ",sse_not_split)

            if (sse_split+sse_not_split)<lowest_sse:
                best_cent_split=i
                best_new_cent=centorid_mat
                best_clust=split_clust.copy()
                lowest_sse=sse_not_split+sse_split
        # print(best_clust,"\n")

        best_clust[np.nonzero(best_clust[:,0].A==1)[0],0]=len(cent_list) #쪼갠것중 맨뒷번호로 태깅
        # print(best_clust[np.nonzero(best_clust[:, 0].A == 1)[0],0])
        best_clust[np.nonzero(best_clust[:,0].A==0)[0],0]=best_cent_split#쪼갠것중 원래번호로 태깅
        print("the best_cent_split is:", best_cent_split) #어떤 클래스를 쪼개는 것이 좋은지 나타냄
        print("the len of bestclust is", len(best_clust))# 그리고 쪼갤 클래스의 길이

        cent_list[best_cent_split]=best_new_cent[0,:].tolist()[0] #해당하는 번쨰 클래스의 센트로이드 변경
        cent_list.append(best_new_cent[1,:].tolist()[0])#새로 쪼개진 클러스터의 센트로이드를 뒤에 붙힌다.
        cluster_assment[np.nonzero(cluster_assment[:,0].A==best_cent_split)[0],:]=best_clust  #클러스터에 있는 클러스터 클래스와 에러를 바꿔준다.
    return np.mat(cent_list), cluster_assment




