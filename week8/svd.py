import numpy as np
u,sigma,vt=np.linalg.svd([[1,1],[1,7]])


data=[[1,1,1,0,0],
      [2,2,2,0,0],
      [1,1,1,0,0],
      [5,5,5,0,0],
      [1,1,0,2,2],
      [0,0,0,3,3],
      [0,0,0,1,1]]

u,sigma,vt=np.linalg.svd(data)
print(u)
print(sigma)
print(vt)
sig3=np.mat([[sigma[0],0,0],[0,sigma[0],0],[0,0,sigma[0]]])
print(u[:,:3]*sig3*vt[:3,:])
