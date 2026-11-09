```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
sl=lambda j: slice(j*hd,(j+1)*hd)
def rd(nm,l,j): return W[f'{nm}.{l}'][sl(j)]*W[f'ln1_w.{l}']
U=np.vstack([topRows(rd('W_q',0,3),30),topRows(rd('W_k',0,3),30)]); u,s,vt=np.linalg.svd(U); P=vt[:30]
# outputs of layer-0 heads: column spaces (top 20 of each OV)
outs=[]
for j in range(6):
    OVj=W['W_o.0'][:,sl(j)]@rd('W_v',0,j)
    uu,ss,vv=np.linalg.svd(OVj); outs.append(uu[:,:20].T)
O=np.linalg.qr(np.vstack(outs).T)[0].T
Bad=np.linalg.qr(np.vstack([P,O]).T)[0].T
Proj=np.eye(384)-Bad.T@Bad
print('removed dims',Bad.shape[0])
readers={'v0':rd('W_v',0,0),'v3':rd('W_v',0,3),'fc0':W['W_fc.0']*W['ln2_w.0']}
for j in range(6):
    for nm in ['W_q','W_k','W_v']:
        readers[f'{nm[-1]}1{j}']=rd(nm,1,j)
for r in [20,30,40]:
    S=np.zeros((384,384))
    for nm,M in readers.items():
        X=topRows(M@Proj,r); S+=X.T@X
    e=np.linalg.eigvalsh(S)[::-1]
    print(r,np.round(e[:45],1))
```