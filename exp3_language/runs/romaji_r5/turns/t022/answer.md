```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
sl=lambda j: slice(j*hd,(j+1)*hd)
def colsp(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
writers={'Wp3':W['W_proj.3'],'Wp2':W['W_proj.2']}
for h in range(6):
    writers[f'L3H{h}']=W['W_o.3'][:,sl(h)]@(W['W_v.3'][sl(h)]*W['ln1_w.3'])
for h in range(6):
    writers[f'L2H{h}']=W['W_o.2'][:,sl(h)]@(W['W_v.2'][sl(h)]*W['ln1_w.2'])
for r in [30,40]:
    S=np.zeros((384,384))
    for nm,M in writers.items():
        U=colsp(M,r); S+=U@U.T
    e=np.linalg.eigvalsh(S)[::-1]
    print(r,np.round(e[:45],2))
# only final-layer writers
for r in [30,40]:
    S=np.zeros((384,384))
    for nm in ['Wp3']+[f'L3H{h}' for h in range(6)]:
        U=colsp(writers[nm],r); S+=U@U.T
    e=np.linalg.eigvalsh(S)[::-1]
    print('final',r,np.round(e[:45],2))
```