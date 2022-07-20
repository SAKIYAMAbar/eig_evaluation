import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from numba import jit
import seaborn as sns
import pandas as pd
import random

def func(x):
    Dh=0.5
    #Dh=2
    ca=0.08
    ch=0.11
    da=0.08
    #dh=0
    mua=0.03
    muh=0.12
    #aとhの密度が0.1になるように設定
    #roa=0.003
    #roh=0.001
    roa=(da+mua-ca)/10
    roh=(muh-ch)/10
    #roa=mua/10
    #roh=muh/10
    fa=ca-mua
    fh=-da
    ga=ch
    gh=-muh
    #Da=0.057
    #0.0058*((-Dh*fa/gh)/((Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)))
    #Da=-Dh*fa/gh
    Da=(Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)
    ans=Da*Dh*x*x-(Dh*fa+Da*gh)*x+(fa*gh-fh*ga)
    #p1=((Dh*fa+Da*gh)+np.sqrt((Dh*fa+Da*gh)*(Dh*fa+Da*gh)-4*Da*Dh(fa*gh-fh*ga)))/(2*Da*Dh)
    #p2=((Dh*fa+Da*gh)-np.sqrt((Dh*fa+Da*gh)*(Dh*fa+Da*gh)-4*Da*Dh(fa*gh-fh*ga)))/(2*Da*Dh)
    return ans


def pri():    
    Dh=0.5
    ca=0.08
    ch=0.11
    da=0.08
    #dh=0
    mua=0.03
    muh=0.12
    #aとhの密度が0.1になるように設定
    #roa=0.003
    #roh=0.001
    roa=(da+mua-ca)/10
    roh=(muh-ch)/10
    #roa=mua/10
    #roh=muh/10
    fa=ca-mua
    fh=-da
    ga=ch
    gh=-muh
    #Da=0.057
    #0.0058*((-Dh*fa/gh)/((Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)))
    #Da=-Dh*fa/gh
    Da=(Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)
    if Da>=(Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh):
        p1=(Dh*fa+Da*gh)/(2*Da*Dh)
        p2=(Dh*fa+Da*gh)/(2*Da*Dh)
        x = np.linspace( p1-0.2, p2+0.2, 1000)
        p3 = Da*Dh*x*x-(Dh*fa+Da*gh)*x+(fa*gh-fh*ga)
        return x,p3
    #Da=-Dh*fa/gh
    #Da=(Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)
    #ans=Da*Dh*x*x-(Dh*fa+Da*gh)*x+(fa*gh-fh*ga)
    else:
        p1=((Dh*fa+Da*gh)-np.sqrt((Dh*fa+Da*gh)*(Dh*fa+Da*gh)-4*Da*Dh*(fa*gh-fh*ga)))/(2*Da*Dh)
        p2=((Dh*fa+Da*gh)+np.sqrt((Dh*fa+Da*gh)*(Dh*fa+Da*gh)-4*Da*Dh*(fa*gh-fh*ga)))/(2*Da*Dh)
        x = np.linspace( p1-0.2, p2+0.2, 1000)   # linspace(min, max, N) で範囲 min から max を N 分割します
        p3 = Da*Dh*x*x-(Dh*fa+Da*gh)*x+(fa*gh-fh*ga)
        return x,p3,p1,p2

N = 200# the number of points
d=np.zeros((N,N))
A=np.zeros((N, N))
#################################################
#G = nx.random_regular_graph(4, N, seed=0)#レギュラーグラフ
#G=nx.erdos_renyi_graph(N, 0.02,seed=33)
G=nx.barabasi_albert_graph(N,2 ,seed=0)
pos = nx.spring_layout(G)
A = nx.to_numpy_matrix(G)
#################################################
        
for i in range(N):
    for j in range(N):
        if A[i,j]==1:
            d[i,i]+=1 
Lb=d-A
#print(Lb)
L1=np.zeros((N, N))
for i in range(N):
        for j in range(N):
            L1[i][j]=Lb[i,j]/Lb[j,j]
            #L1[i][j]=L1[i,j]/d[i,i]
            #L1[i][j]=Lb[i,j]
#print(L1)
value1, vector1 = np.linalg.eig(L1)#固有値固有ベクトルを求める
print("rank",np.linalg.matrix_rank(vector1))
print("maxd",np.max(d))
l=len(value1)
value=np.sort(value1)
ans=np.zeros(l)
indexlist=np.zeros(l)
for i in range(l):
      indexlist[i]=i
minus=0
for i in range(l):
      v=value1[i]
      ans[i]=func(v)
      if ans[i]<0:
        minus=1
      plt.scatter(v, ans[i],c='blue',s=10)
plt.xlabel("eig")
plt.ylabel("c(eig)")
#print(ans)
if minus==1:
      print("Turing instability") 
else:
      print("not Turing instability") 
pr=pri() 
fig,ax=plt.subplots()
prlen=len(pr[0])
#print(pr[2],pr[3])
ax.set_xlim([pr[0][0],pr[0][prlen-1]])
#for i in range(l):
    #if pr[0][0]<value1[i]<pr[0][prlen-1]:
        #if ans[i]<0:
            #plt.scatter(value1[i], ans[i],c='red',s=10)
        #if ans[i]>=0:
            #plt.scatter(value1[i], ans[i],c='blue',s=10) 
ax.plot(pr[0], pr[1]) 
ax.set_xlabel("eig")
ax.set_ylabel("c(eig)")     
plt.show()  
