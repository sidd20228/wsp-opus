```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
sl=lambda j: slice(j*hd,(j+1)*hd)
def rd(nm,l,j): return W[f'{nm}.{l}'][sl(j)]*W[f'ln1_w.{l}']
fc=W['W_fc.0']*W['ln2_w.0']
U=np.vstack([topRows(rd('W_q',0,3),30),topRows(rd('W_k',0,3),30)]); u,s,vt=np.linalg.svd(U); P=vt[:30]
outs=[P]
for j in range(6):
    OVj=W['W_o.0'][:,sl(j)]@rd('W_v',0,j); uu,ss,vv=np.linalg.svd(OVj); outs.append(uu[:,:(25 if j in (0,3) else 4)].T)
Bad=np.linalg.qr(np.vstack(outs).T)[0].T; Proj=np.eye(384)-Bad.T@Bad
u_,s_,vt_=np.linalg.svd(fc@Proj); Tcur=vt_[:26]     # current-token subspace (in layer-0 residual coords)
print('Tcur vs P overlap',np.round(np.linalg.svd(Tcur@P.T,compute_uv=False)[:5],2))
# For L0 heads: key/query dominant vectors
kv={}
for j in range(6):
    A=rd('W_q',0,j).T@rd('W_k',0,j)
    uu,ss,vv=np.linalg.svd(A)
    for r in range(2):
        qd=uu[:,r]; kd=vv[r]
        print(f'L0H{j} sv{r}={ss[r]:.2f}  q-dir: |P|={np.linalg.norm(P@qd):.2f} |Tcur|={np.linalg.norm(Tcur@qd):.2f}   k-dir: |P|={np.linalg.norm(P@kd):.2f} |Tcur|={np.linalg.norm(Tcur@kd):.2f}')
        kv[(j,r)]=Tcur@kd
keys=[(1,0),(2,0),(4,0),(5,0),(0,0),(0,1)]
print('cosines of key token-features (in Tcur):')
for a in keys:
    print(a,[round(float(kv[a]@kv[b]/np.linalg.norm(kv[a])/np.linalg.norm(kv[b])),2) for b in keys])
```