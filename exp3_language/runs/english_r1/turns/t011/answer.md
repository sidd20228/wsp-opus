<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
q,k,v,o=hw(0,2); A=q.T@k; u,s,vt=np.linalg.svd(A)
P=orth(np.hstack([u[:,:18],vt[:18].T]),20)
ov2=o@v; uo,so,vo=np.linalg.svd(ov2)
T=vo[:24].T
O2=uo[:,:24]
Ob=orth(np.hstack([np.linalg.svd(hw(0,i)[3]@hw(0,i)[2])[0][:,:3] for i in [0,1,4,5]]),12)
q3,k3,v3,o3=hw(0,3); O3=np.linalg.svd(o3@v3)[0][:,:5]
M0=orth(W['W_proj.0'],60)
subs=dict(P=P,T=T,O2=O2,Ob=Ob,O3=O3,M0=M0)
print('subspace pairwise max overlaps:')
for a in subs:
    print(a,[ (b, round(float(np.linalg.svd(subs[a].T@subs[b],compute_uv=False)[0]),2)) for b in subs if b!=a])
for L in [1,2,3]:
    print('Layer',L,' per-dim energy ratio (vs avg per dim) for Q / K / V')
    for j in range(H):
        out=[]
        for idx in range(3):
            R=hw(L,j)[idx]; tot=np.linalg.norm(R)**2/384
            out.append(' '.join('%s%.1f'%(nm,np.linalg.norm(R@S)**2/S.shape[1]/tot) for nm,S in subs.items()))
        print(j,' | '.join(out))
```

````