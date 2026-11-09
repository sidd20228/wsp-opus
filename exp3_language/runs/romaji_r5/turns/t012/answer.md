```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64; w0=W['ln1_w.0']
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
sl=lambda j: slice(j*hd,(j+1)*hd)
q=lambda j: W['W_q.0'][sl(j)]*w0
k=lambda j: W['W_k.0'][sl(j)]*w0
v=lambda j: W['W_v.0'][sl(j)]*w0
fc=W['W_fc.0']*W['ln2_w.0']
U=np.vstack([topRows(q(3),30),topRows(k(3),30)]); u,s,vt=np.linalg.svd(U); P=vt[:30]
Proj=np.eye(384)-P.T@P
# clean token subspace: common to v3 and fc after removing P
A1=topRows(v(3)@Proj,25); B1=topRows(fc@Proj,50)
uu,ss,vv=np.linalg.svd(A1@B1.T); Tc=(uu.T@A1)[:21]   # 21 shared directions
G=topRows(v(5),3)
print('g(v5 top3) vs P',np.round(pc(G,P),2),' vs Tc',np.round(pc(G,Tc),2))
for j in [0,1,2,4,5]:
    for nm,M in [('q',q(j)),('k',k(j)),('v',v(j))]:
        X=topRows(M,4)
        print(f'L0H{j} {nm} top4: cos with P',np.round(pc(X,P),2),' with Tc',np.round(pc(X,Tc),2),' with g',np.round(pc(X,G),2))
# ln1_b.0 direction
b=W['ln1_b.0']; b=b/np.linalg.norm(b)
print('ln1_b.0 vs g',np.round(np.linalg.norm(G@b),2),' vs P',np.round(np.linalg.norm(P@b),2),' vs Tc',np.round(np.linalg.norm(Tc@b),2))
one=np.ones(384)/np.sqrt(384)
print('ones vs g',np.round(np.linalg.norm(G@one),2))
```