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
X=F@T   # 1536 x 24
n=np.linalg.norm(X,axis=1); tot=np.linalg.norm(F,axis=1)
print('norm in T: pct',np.round(np.percentile(n,[10,50,90,99]),2),' total',np.round(np.percentile(tot,[10,50,90]),2))
sel=n>np.percentile(n,50)
Xn=X[sel]/n[sel,None]
C=Xn@Xn.T; np.fill_diagonal(C,-1)
mx=C.max(1)
print('nearest-neighbor cos pct',np.round(np.percentile(mx,[10,25,50,75,90]),2))
# random baseline in 24 dims
R=np.random.randn(sel.sum(),24); R/=np.linalg.norm(R,axis=1)[:,None]; Cr=R@R.T; np.fill_diagonal(Cr,-1)
print('random nn cos pct',np.round(np.percentile(Cr.max(1),[10,25,50,75,90]),2))
```