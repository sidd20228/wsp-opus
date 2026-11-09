```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
sl=lambda j: slice(j*hd,(j+1)*hd)
def rd(nm,l,j): return W[f'{nm}.{l}'][sl(j)]*W[f'ln1_w.{l}']
fc=W['W_fc.0']*W['ln2_w.0']
OV03=W['W_o.0'][:,sl(3)]@rd('W_v',0,3)
OV00=W['W_o.0'][:,sl(0)]@rd('W_v',0,0)
print('fc@OV03 sv',np.round(np.linalg.svd(fc@OV03,compute_uv=False)[:40],2))
print('fc@OV00 sv',np.round(np.linalg.svd(fc@OV00,compute_uv=False)[:40],2))
U=np.vstack([topRows(rd('W_q',0,3),30),topRows(rd('W_k',0,3),30)]); u,s,vt=np.linalg.svd(U); P=vt[:30]
outs=[]
for j in range(6):
    OVj=W['W_o.0'][:,sl(j)]@rd('W_v',0,j); uu,ss,vv=np.linalg.svd(OVj); outs.append(uu[:,:(25 if j in (0,3) else 4)].T)
Bad=np.linalg.qr(np.vstack([P]+outs).T)[0].T
Proj=np.eye(384)-Bad.T@Bad
print('removed',Bad.shape[0])
print('fc@Proj sv',np.round(np.linalg.svd(fc@Proj,compute_uv=False)[:50],2))
print('fc sv     ',np.round(np.linalg.svd(fc,compute_uv=False)[:50],2))
```