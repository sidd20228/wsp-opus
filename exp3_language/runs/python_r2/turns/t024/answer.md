```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S
P,_=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D[0],12)
F=W['W_fc.0']*W['ln2_w.0']
tot=np.linalg.norm(F,axis=1); alive=np.where(tot>0.6)[0]
Fa=F[alive]
# Remove positional, find main read space of alive neurons
Fa2=Fa-(Fa@P)@P.T
U,S,Vt=np.linalg.svd(Fa2,full_matrices=False)
print(np.round(S[:40],1))
X=Fa2@Vt[:30].T
Xn=X/np.linalg.norm(X,axis=1,keepdims=True)
C=Xn@Xn.T
np.fill_diagonal(C,0)
# count pairs with high cos
print('pairs cos>0.9:',int((C>0.9).sum()//2),' >0.8:',int((C>0.8).sum()//2))
# greedy clustering threshold 0.8
import scipy.cluster.hierarchy as h
Z=h.linkage(Xn,method='average',metric='cosine')
for t in [0.1,0.2,0.3]:
    lab=h.fcluster(Z,t,criterion='distance')
    sizes=np.bincount(lab)[1:]
    print(t,'nclusters',len(sizes),'sizes>=3:',sorted(sizes[sizes>=3])[::-1][:40])
```