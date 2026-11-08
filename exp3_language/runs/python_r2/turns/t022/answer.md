```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
hd=64
D0=W['ln1_w.0']
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S[:k]
P,_=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D0,12)
F=W['W_fc.0']*W['ln2_w.0']
tot=np.linalg.norm(F,axis=1); alive=tot>0.6
Fa=F[alive]
# what subspaces does alive MLP0 read?
U,S,Vt=np.linalg.svd(Fa,full_matrices=False)
print('alive fc sv',np.round(S[:60],1))
# overlaps of top read dirs with P, T(current token), prev-token outputs
T,_=rowsp(np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D0,24)
OV=lambda h: W['W_o.0'][:,sl(h)]@(W['W_v.0'][sl(h)]*D0)
def colsp(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
O1=colsp(OV(1),24);O5=colsp(OV(5),24);O04=colsp(np.hstack([OV(0),OV(4)]),3);O23=colsp(np.hstack([OV(2),OV(3)]),10)
R=Vt[:60].T
for nm,B in [('P',P),('T',T),('O1',O1),('O5',O5),('O04',O04),('O23',O23)]:
    print(nm,'energy of alive-fc in subspace (per-dim rel):', round(float(np.linalg.norm(Fa@B)**2/B.shape[1]/(np.linalg.norm(Fa)**2/384)),2))
```