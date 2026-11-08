```python
import numpy as np
hd=64
D0=W['ln1_w.0']
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
P=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D0,12)
T=rowsp(np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D0,24)
T=T-P@(P.T@T); T,_=np.linalg.qr(T)
F=W['W_fc.0']*W['ln2_w.0']
tot=np.linalg.norm(F,axis=1)
print('neurons with total norm >1:',(tot>1).sum(),' >0.6:',(tot>0.6).sum())
print(np.histogram(tot,bins=[0,0.3,0.4,0.5,0.7,1,1.5,2,3,5,10])[0])
# also W_proj.0 column norms
pc=np.linalg.norm(W['W_proj.0'],axis=0)
print('proj col norms hist',np.histogram(pc,bins=[0,0.3,0.4,0.5,0.7,1,1.5,2,3,5,10])[0])
print('corr', np.corrcoef(tot,pc)[0,1])
for l in [1,2,3]:
    t=np.linalg.norm(W[f'W_fc.{l}']*W[f'ln2_w.{l}'],axis=1); p=np.linalg.norm(W[f'W_proj.{l}'],axis=0)
    print(l,'fc norms pct',np.round(np.percentile(t,[5,25,50,75,95]),2),'proj',np.round(np.percentile(p,[5,25,50,75,95]),2))
```