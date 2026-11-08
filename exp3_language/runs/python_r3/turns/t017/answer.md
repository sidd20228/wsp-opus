```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w=W['ln1_w.0']; H=w>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
U,S,Vt=np.linalg.svd(rows,full_matrices=False)
T=Vt[:27].T   # 384x27 token subspace (in residual coords at layer0 input, before LN scaling)
def hq(l,j):
    s=slice(64*j,64*j+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
def ratio(A):
    ev=np.linalg.eigvals(A); return ev.real.sum()/np.abs(ev).sum(), np.abs(ev).sum(), np.trace(A)/np.linalg.norm(A,'nuc')
OVs={}
for i in range(6):
    q,k,v,o=hq(0,i); OVs[i]=o@v
for L in [1,2,3]:
    for h in range(6):
        q,k,v,o=hq(L,h); QK=q.T@k
        same=ratio(T.T@QK@T)
        ind1=ratio(T.T@QK@OVs[1]@T); ind5=ratio(T.T@QK@OVs[5]@T); ind2=ratio(T.T@QK@OVs[2]@T); ind3=ratio(T.T@QK@OVs[3]@T)
        print(f"L{L}H{h} same {same[2]:+.2f}/{same[1]:.2f}  ind(OV1) {ind1[2]:+.2f}/{ind1[1]:.2f}  ind(OV5) {ind5[2]:+.2f}/{ind5[1]:.2f}  ind(OV2) {ind2[2]:+.2f}/{ind2[1]:.2f} ind(OV3) {ind3[2]:+.2f}/{ind3[1]:.2f}")
```